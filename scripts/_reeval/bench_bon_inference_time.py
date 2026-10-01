"""Experiment #1: BoN deployment latency vs N.

Times the FULL candidate-generation + critic-scoring block -- one call to
`BestofNLearner.sample_actions` -- at env_num=1, for N in 1,2,4,8,16,32. This is
the per-decision cost at deployment, so env_num is 1 and the environment is
never stepped: no rendering, no physics, only policy + critic.

Why not read this off the eval logs: at N=1 the eval path still runs the critic
and discards the score, so eval wall time at N=1 is not the policy-only latency.

MEASUREMENT PROTOCOL. A first version measured each N once, in ascending order,
after 3 warmup calls, and produced N=2 SLOWER than N=4 -- with per-N spreads of
only 1-4 ms, so it was systematic, not noise. Whatever its cause (clock ramp,
lingering async compile, allocator state), a latency curve that is not monotone
in N is not a usable measurement. This version removes the possibility:

  1. COMPILE     every N is compiled and exercised before anything is timed, so
                 no timed call can absorb a JIT.
  2. SETTLE      a fixed-duration burst cycling through all N, discarded. Brings
                 GPU clocks and power state to steady state before timing.
  3. INTERLEAVE  ROUNDS passes over the N values, each pass rotated by one, with
                 K timed calls per N per pass. Any residual drift is spread
                 across all N instead of landing on whichever went first.
  4. CHECK       the per-round medians are reported, and a non-monotone final
                 curve is flagged loudly rather than quietly returned.

Reports the median over all ROUNDS*K samples per N, with IQR.

Run (needs a GPU):
    sbatch scripts/_reeval/bench_bon_inference_time.sbatch.sh
"""
import dataclasses, json, logging, os, platform, statistics, time
from pathlib import Path

import jax
import numpy as np

from src.training import config as _config
from src.envs import make_env
from src.rl.best_of_n.best_of_n_learner import BestofNLearner
from src.rl.filtered_sft_agent.filtered_sft_learner import filtered_sft_wrap_env
from src.training.utils import init_logging

N_VALUES = (1, 2, 4, 8, 16, 32)
WARMUP = int(os.environ.get("BENCH_WARMUP", 5))     # compile + exercise, per N
SETTLE_S = float(os.environ.get("BENCH_SETTLE_S", 30))
ROUNDS = int(os.environ.get("BENCH_ROUNDS", 6))
K = int(os.environ.get("BENCH_K", 5))               # timed calls per N per round
OUT = Path("/capstor/scratch/cscs/ralfroemer/bon_latency")


def main(config: _config.OnlineTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}  devices={jax.devices()}")
    assert config.collect.eval_env_num == 1, "deployment latency must be measured at env_num=1"

    agent = BestofNLearner(config)
    assert agent._resuming, "refusing to benchmark an un-restored agent"
    logging.info(f"restored at training_steps={int(agent.training_steps)}")

    task = config.collect.eval_tasks[0]
    env = filtered_sft_wrap_env(make_env(config, [task], num_devices=1),
                                config=config, env_num=1)
    obs, info = env.reset()
    prompt = info["task_description"] if isinstance(info, dict) else info[0]["task_description"]
    logging.info(f"observation from {task}: {sorted(obs)} prompt={prompt!r}")

    def set_n(n):
        agent._config = dataclasses.replace(
            agent._config, rl=dataclasses.replace(agent._config.rl, n_samples=n))
        assert agent._config.rl.n_samples == n

    def call():
        jax.block_until_ready(agent.sample_actions(obs, task_description=prompt))

    samples = {n: [] for n in N_VALUES}
    round_medians = {n: [] for n in N_VALUES}
    try:
        # 1. COMPILE -------------------------------------------------------
        t0 = time.perf_counter()
        for n in N_VALUES:
            set_n(n)
            for _ in range(WARMUP):
                call()
        logging.info(f"compile+warmup done for all N in {time.perf_counter()-t0:.1f}s")

        # 2. SETTLE --------------------------------------------------------
        t0 = time.perf_counter(); c = 0
        while time.perf_counter() - t0 < SETTLE_S:
            set_n(N_VALUES[c % len(N_VALUES)]); call(); c += 1
        logging.info(f"settled: {c} discarded calls over {SETTLE_S:.0f}s")

        # 3. INTERLEAVED MEASUREMENT --------------------------------------
        for r in range(ROUNDS):
            order = N_VALUES[r % len(N_VALUES):] + N_VALUES[:r % len(N_VALUES)]
            for n in order:
                set_n(n)
                ts = []
                for _ in range(K):
                    t = time.perf_counter(); call(); ts.append(time.perf_counter() - t)
                samples[n].extend(ts)
                round_medians[n].append(statistics.median(ts))
            logging.info(f"round {r+1}/{ROUNDS} ({order[0]} first): "
                         + "  ".join(f"N={n}:{round_medians[n][-1]*1000:.1f}" for n in N_VALUES))
    finally:
        env.close()

    results = {}
    for n in N_VALUES:
        ts = sorted(samples[n]); q = statistics.quantiles(ts, n=4)
        results[n] = dict(median_s=statistics.median(ts), p25_s=q[0], p75_s=q[2],
                          min_s=ts[0], max_s=ts[-1], n_timed=len(ts),
                          round_medians_s=round_medians[n], raw_s=ts)

    base = results[1]["median_s"]
    logging.info("=== BoN deployment latency (env_num=1) ===")
    for n in N_VALUES:
        r = results[n]
        logging.info(f"  N={n:<3d} {r['median_s']*1000:8.1f} ms  "
                     f"IQR=[{r['p25_s']*1000:.1f}, {r['p75_s']*1000:.1f}]  "
                     f"{r['median_s']/base:5.2f}x vs N=1   "
                     f"rounds={[round(x*1000,1) for x in r['round_medians_s']]}")

    meds = [results[n]["median_s"] for n in N_VALUES]
    bad = [(N_VALUES[i], N_VALUES[i+1]) for i in range(len(meds)-1) if meds[i] > meds[i+1]]
    if bad:
        logging.error(
            f"NON-MONOTONE: {bad} -- latency must not fall as N grows. Check, in order: "
            "(a) more than one visible device, which shards a tiny batch raggedly and "
            "makes e.g. N=2 cost more than N=4 -- a deployment latency must be measured "
            "on ONE device; (b) an unsettled machine, which the per-round medians above "
            "would show as drift. Do not use these numbers until it is monotone.")
    else:
        logging.info("monotone in N: OK")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "bon_latency_env1.json"
    out_path.write_text(json.dumps(
        {"host": platform.node(), "devices": [str(d) for d in jax.devices()],
         "env_num": 1, "warmup": WARMUP, "settle_s": SETTLE_S, "rounds": ROUNDS, "k": K,
         "task": task, "monotone": not bad, "non_monotone_pairs": bad,
         "checkpoint_base_dir": str(config.checkpoint_base_dir),
         "training_steps": int(agent.training_steps),
         "results": {str(k2): v for k2, v in results.items()}}, indent=2))
    logging.info(f"wrote {out_path}")


if __name__ == "__main__":
    main(_config.cli())
