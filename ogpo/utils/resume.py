"""Resumable round-based training: replay-buffer chunks + agent checkpoints mirrored to S3 (or a directory).

A batch service that repairs a failed node by restarting the container from scratch needs the state of a long run on S3:

  * a replay-buffer *chunk* per round (the rows that round's collection added, so a round uploads only new data),
  * an agent checkpoint (trainable params + optimiser states) every `ckpt_interval` updates,
  * `MANIFEST.json`, rewritten once both halves of a consistent pair are on S3.

Resume point = (R, s): rounds 0..R are collected into the buffer and the agent is at update step s with
R * updates_per_round <= s <= (R + 1) * updates_per_round. Uploads run serially in a background thread; `commit`
only advances the manifest to a pair that satisfies the inequality.
"""
import json
import os
import pickle
import queue
import shutil
import subprocess
import threading
import time

import jax
import numpy as np

MANIFEST = "MANIFEST.json"


# ----------------------------------------------------------------------------- storage backends
def _is_s3(uri):
    return str(uri).startswith("s3://")


def _run(cmd):
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(f"{' '.join(cmd[:4])} ... failed ({out.returncode}): {out.stderr[-500:]}")
    return out.stdout


def _put(src, dst):
    """Copy a local file or directory to `dst` (an s3:// URI or a path)."""
    if _is_s3(dst):
        _run(["aws", "s3", "cp", "--only-show-errors"] + (["--recursive"] if os.path.isdir(src) else []) + [src, dst])
    elif os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _get(src, dst, recursive=False):
    if _is_s3(src):
        _run(["aws", "s3", "cp", "--only-show-errors"] + (["--recursive"] if recursive else []) + [src, dst])
    elif recursive:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)


def _rm(uri, recursive=True):
    if _is_s3(uri):
        _run(["aws", "s3", "rm", "--only-show-errors"] + (["--recursive"] if recursive else []) + [uri])
    elif os.path.isdir(uri):
        shutil.rmtree(uri, ignore_errors=True)
    elif os.path.exists(uri):
        os.remove(uri)


def _exists(uri):
    if _is_s3(uri):
        r = subprocess.run(["aws", "s3", "ls", uri], capture_output=True, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    return os.path.exists(uri)


# ----------------------------------------------------------------------------- pytrees
def restore_tree(cur, new, what="state"):
    """Put the numpy tree `new` on the devices (and in the sharding) of the matching leaves of `cur`."""
    if jax.tree.structure(cur) != jax.tree.structure(new):
        raise ValueError(f"resume: {what} tree does not match the agent (same agent flags as the run being resumed?)")

    def put(c, n):
        n = np.asarray(n)
        if tuple(c.shape) != tuple(n.shape):
            raise ValueError(f"resume: {what} leaf shape {tuple(n.shape)} != the agent's {tuple(c.shape)}")
        n = n.astype(c.dtype)
        return jax.device_put(n, c.sharding) if hasattr(c, "sharding") else n

    return jax.tree.map(put, cur, new)


def restore_trees(pairs, protect=()):
    """`restore_tree` for several (cur, new, what) triples at once, restoring a live agent in place. Every tree structure and leaf
    shape is validated first (ValueError before anything is touched). Then, leaf by leaf, the current device buffer is deleted
    (unless its id is in `protect`: still referenced by something that stays) and the new value is placed on the same sharding.
    Deleting right before placing lets the new buffer take the freed hole in the device allocator. Placing it next to the live
    copy instead stacks a second copy above the first: the high-water mark of the pool rises, and the next large executable
    allocation no longer finds a contiguous block (mt144 OOM at 44.31 GiB, docs/changes/2026-10-27-pi05-snapshot-resume).
    -> the restored trees, in order."""
    prepared = []
    for cur, new, what in pairs:
        if jax.tree.structure(cur) != jax.tree.structure(new):
            raise ValueError(f"resume: {what} tree does not match the agent (same agent flags as the run being resumed?)")
        leaves = []
        for c, n in zip(jax.tree.leaves(cur), jax.tree.leaves(new)):
            n = np.asarray(n)
            if tuple(c.shape) != tuple(n.shape):
                raise ValueError(f"resume: {what} leaf shape {tuple(n.shape)} != the agent's {tuple(c.shape)}")
            leaves.append((c, n.astype(c.dtype), getattr(c, "sharding", None)))
        prepared.append((jax.tree.structure(cur), leaves))
    freed, out = set(), []
    for treedef, leaves in prepared:
        placed = []
        for c, n, sharding in leaves:
            if isinstance(c, jax.Array) and id(c) not in protect and id(c) not in freed:
                c.delete()
                freed.add(id(c))
            placed.append(jax.device_put(n, sharding) if sharding is not None else n)
        out.append(jax.tree.unflatten(treedef, placed))
    return out


# ----------------------------------------------------------------------------- the store
class ResumeStore:
    """`local_dir`: scratch for files being written / fetched. `remote`: s3://... or a directory (tests)."""

    def __init__(self, local_dir, remote, updates_per_round, ckpt_interval=250, rebase_rounds=False):
        self.local, self.remote = str(local_dir), str(remote).rstrip("/")
        self.upr, self.ckpt_interval = int(updates_per_round), int(ckpt_interval)
        # rebase_rounds: the restored manifest ends a finished round collected with another updates_per_round; the
        # rounds after it use self.upr. Round r then ends at update (r + 1) * upr + step_offset (0 without a rebase).
        self.rebase_rounds, self.step_offset = bool(rebase_rounds), 0
        os.makedirs(self.local, exist_ok=True)
        self._chunks = {}   # round -> {"rows": n, "meta": {...}} committed on the remote
        self._agents = {}   # update step -> remote key committed
        self._q = queue.Queue()
        self._err = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    # ---- remote layout
    def _r(self, *p):
        return "/".join((self.remote,) + p)

    # ---- upload thread
    def _worker(self):
        while True:
            job = self._q.get()
            if job is None:
                self._q.task_done()
                return
            try:
                t0 = time.time()
                kind = job[0]
                if kind == "chunk":
                    _, r, path, rows, meta = job
                    _put(path, self._r("chunks", f"c{r:04d}"))
                    self._chunks[r] = {"rows": rows, "meta": meta}
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    _, s, path, csv_dir = job
                    _put(path, self._r("agent", f"step{s:07d}.pkl"))
                    _put(csv_dir, self._r("csv", f"step{s:07d}"))
                    self._agents[s] = {"csv": f"csv/step{s:07d}"}
                    os.remove(path)
                    shutil.rmtree(csv_dir, ignore_errors=True)
                print(f"[resume] uploaded {kind} {job[1]} in {time.time() - t0:.0f}s", flush=True)
                self._commit()
            except Exception as e:  # keep training; the previous manifest stays valid
                self._err = e
                print(f"[resume] upload failed: {e}", flush=True)
            finally:
                self._q.task_done()

    def _commit(self):
        if not self._chunks or not self._agents:
            return
        R = max(self._chunks)
        ok = [s for s in self._agents if R * self.upr <= s - self.step_offset <= (R + 1) * self.upr]
        if not ok:
            return
        s = max(ok)
        kept, rows = [], 0  # newest chunks until they cover the whole ring
        cap = self._chunks[R]["meta"]["max_size"]
        for r in sorted(self._chunks, reverse=True):
            kept.append(r)
            rows += self._chunks[r]["rows"]
            if rows >= cap:
                break
        manifest = {"round": R, "step": s, "agent": f"agent/step{s:07d}.pkl", "csv": self._agents[s]["csv"],
                    "chunks": sorted(kept), "buffer": self._chunks[R]["meta"], "step_offset": self.step_offset,
                    "utc": time.strftime("%FT%TZ", time.gmtime())}
        tmp = os.path.join(self.local, MANIFEST)
        with open(tmp, "w") as f:
            json.dump(manifest, f)
        _put(tmp, self._r(MANIFEST))
        print(f"[resume] manifest -> round {R}, step {s}, chunks {sorted(kept)}", flush=True)
        for r in [r for r in self._chunks if r not in kept]:
            _rm(self._r("chunks", f"c{r:04d}"))
            del self._chunks[r]
        for old in [a for a in self._agents if a < s]:
            _rm(self._r("agent", f"step{old:07d}.pkl"), recursive=False)
            _rm(self._r(self._agents[old]["csv"]))
            del self._agents[old]

    def wait(self):
        """Block until every queued upload finished."""
        self._q.join()
        return self._err

    def close(self):
        self.wait()
        self._q.put(None)

    # ---- saving
    def save_chunk(self, rb, r, pointer_before, total_before, extra):
        """Queue the rows added to `rb` since (`pointer_before`, `total_before`). `extra`: JSON-able run counters."""
        n = rb.total_added - total_before
        m = min(n, rb.max_size)  # a round larger than the ring only keeps its tail
        p0 = (pointer_before + n - m) % rb.max_size
        segs = [(p0, min(p0 + m, rb.max_size))]
        if p0 + m > rb.max_size:
            segs.append((0, p0 + m - rb.max_size))
        path = os.path.join(self.local, "chunks", f"c{r:04d}")
        shutil.rmtree(path, ignore_errors=True)
        for i, (lo, hi) in enumerate(segs):
            d = os.path.join(path, f"seg{i}")
            os.makedirs(d)
            for k, v in rb._dict.items():
                np.save(os.path.join(d, k + ".npy"), v[lo:hi])
            np.save(os.path.join(d, "_traj_success_mask.npy"), rb._traj_success_mask[lo:hi])
        meta = dict(extra, segments=[list(s) for s in segs], pointer=rb.pointer, size=rb.size,
                    total_added=rb.total_added, max_size=rb.max_size, round=r)
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f)
        self._q.put(("chunk", r, path, m, meta))

    def save_agent(self, state, step, csv_files):
        """`state`: picklable host pytree. `csv_files`: files copied alongside so a resume continues the CSV logs."""
        path = os.path.join(self.local, "agent", f"step{step:07d}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            pickle.dump(state, f)
        os.replace(path + ".tmp", path)
        csv_dir = os.path.join(self.local, "csv", f"step{step:07d}")
        shutil.rmtree(csv_dir, ignore_errors=True)
        os.makedirs(csv_dir)
        for p in csv_files:
            if os.path.exists(p):
                shutil.copy2(p, csv_dir)
        self._q.put(("agent", step, path, csv_dir))

    # ---- restoring
    def manifest(self):
        if not _exists(self._r(MANIFEST)):
            return None
        tmp = os.path.join(self.local, "fetched_" + MANIFEST)
        _get(self._r(MANIFEST), tmp)
        with open(tmp) as f:
            return json.load(f)

    def set_offset(self, man):
        """Round boundaries for a run continuing from `man`; call before the first save. -> step_offset"""
        if "step_offset" in man:  # this run (or a rebase before it) already fixed the boundaries
            self.step_offset = int(man["step_offset"])
        elif self.rebase_rounds:  # first start of a rebased run: the next round begins at the restored step
            self.step_offset = int(man["step"]) - (int(man["round"]) + 1) * self.upr
            print(f"[resume] rebase: round {man['round']} ends at update {man['step']}; rounds after it run {self.upr} "
                  f"updates (step_offset {self.step_offset})", flush=True)
        return self.step_offset

    def fetch(self, man):
        """Download what `man` references. -> (agent state, csv dir, [chunk dirs oldest first])."""
        t0 = time.time()
        agent_path = os.path.join(self.local, "fetched", "agent.pkl")
        _get(self._r(man["agent"]), agent_path)
        csv_dir = os.path.join(self.local, "fetched", "csv")
        _get(self._r(man["csv"]), csv_dir, recursive=True)
        dirs = []
        for r in man["chunks"]:
            d = os.path.join(self.local, "fetched", f"c{r:04d}")
            _get(self._r("chunks", f"c{r:04d}"), d, recursive=True)
            dirs.append(d)
        with open(agent_path, "rb") as f:
            state = pickle.load(f)
        print(f"[resume] fetched step {man['step']} + chunks {man['chunks']} in {time.time() - t0:.0f}s", flush=True)
        # What is on the remote is already committed: later manifests must keep (and eventually prune) it.
        for r, d in zip(man["chunks"], dirs):
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            self._chunks[r] = {"rows": sum(hi - lo for lo, hi in meta["segments"]), "meta": meta}
        self._agents[man["step"]] = {"csv": man["csv"]}
        return state, csv_dir, dirs

    @staticmethod
    def load_buffer(rb, chunk_dirs):
        """Copy the chunk rows (oldest first) into `rb` and restore its counters. -> the newest chunk's meta."""
        for d in chunk_dirs:
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            for i, (lo, hi) in enumerate(meta["segments"]):
                sd = os.path.join(d, f"seg{i}")
                for k, v in rb._dict.items():
                    v[lo:hi] = np.load(os.path.join(sd, k + ".npy"), mmap_mode="r")
                rb._traj_success_mask[lo:hi] = np.load(os.path.join(sd, "_traj_success_mask.npy"))
        rb.pointer, rb.size, rb.total_added = meta["pointer"], meta["size"], meta["total_added"]
        return meta


def truncate_csvs(csv_dir, save_dir, step, keep_at_step):
    """Copy the checkpoint-time CSVs into `save_dir`, dropping rows logged after the resumed state.
    `keep_at_step`: whether a row logged exactly at `step` belongs to it (env.csv rows are logged before the round's updates)."""
    os.makedirs(save_dir, exist_ok=True)
    for name in os.listdir(csv_dir):
        if not name.endswith(".csv"):
            continue
        keep_eq = keep_at_step or name != "env.csv"
        with open(os.path.join(csv_dir, name)) as f:
            lines = f.read().splitlines()
        if not lines:
            continue
        col = lines[0].split(",").index("step")
        rows = [ln for ln in lines[1:] if (float(ln.split(",")[col]) <= step if keep_eq else float(ln.split(",")[col]) < step)]
        with open(os.path.join(save_dir, name), "w") as f:
            f.write("\n".join([lines[0]] + rows) + "\n")


# ----------------------------------------------------------------------------- live-buffer dump (snapshot of a running job)
def _write_npy(path, arr, chunk_bytes):
    """`arr` -> `path` (.npy) in `chunk_bytes` pieces, flushed as it goes, so a dump from inside a training process does not
    pile up dirty pages in the job's memory cgroup."""
    out = np.lib.format.open_memmap(path, mode="w+", dtype=arr.dtype, shape=arr.shape)
    rows = max(1, chunk_bytes // max(1, arr[0].nbytes)) if len(arr) else 1
    for lo in range(0, len(arr), rows):
        out[lo:lo + rows] = arr[lo:lo + rows]
        out.flush()
    del out


def save_buffer_dump(rb, out_dir, extra, chunk_bytes=1 << 30):
    """Write everything of `rb` that is not already a disk memmap to `out_dir`: one `<field>.npy` per RAM field,
    `_traj_success_mask.npy`, and `meta.json` (counters, field names, `extra`), moved into place last. The buffer must not
    change meanwhile (RuntimeError if its counters moved). `load_buffer_dump` is the inverse."""
    out_dir = str(out_dir).rstrip("/")
    tmp = out_dir + ".tmp"
    if os.path.exists(out_dir) or os.path.exists(tmp):
        raise ValueError(f"save_buffer_dump: {out_dir} or {tmp} already exists")
    before = (rb.pointer, rb.size, rb.total_added)
    os.makedirs(tmp)
    fields = [k for k, v in rb._dict.items() if not isinstance(v, np.memmap)]
    for k in fields:
        _write_npy(os.path.join(tmp, k + ".npy"), rb._dict[k], chunk_bytes)
    _write_npy(os.path.join(tmp, "_traj_success_mask.npy"), rb._traj_success_mask, chunk_bytes)
    if (rb.pointer, rb.size, rb.total_added) != before:
        raise RuntimeError(f"save_buffer_dump: the buffer changed during the dump (counters {before} -> "
                           f"{(rb.pointer, rb.size, rb.total_added)}); {tmp} is incomplete")
    meta = dict(extra, fields=fields, pointer=rb.pointer, size=rb.size, total_added=rb.total_added, max_size=rb.max_size)
    with open(os.path.join(tmp, "meta.json"), "w") as f:
        json.dump(meta, f)
    os.rename(tmp, out_dir)
    return meta


def buffer_dump_meta(dump_dir):
    """The `meta.json` of a `save_buffer_dump` directory (counters, `extra`), without reading any array."""
    with open(os.path.join(dump_dir, "meta.json")) as f:
        return json.load(f)


def load_buffer_dump(rb, dump_dir):
    """Fill the RAM fields, success mask and counters of `rb` from a `save_buffer_dump` directory. Fields already on disk
    (`np.memmap`, opened with `ReplayBuffer.create(reuse_disk=True)`) are left as they are. -> the dump's meta."""
    meta = buffer_dump_meta(dump_dir)
    ram = [k for k, v in rb._dict.items() if not isinstance(v, np.memmap)]
    if sorted(ram) != sorted(meta["fields"]):
        raise ValueError(f"{dump_dir}: dump fields {sorted(meta['fields'])} != the buffer's RAM fields {sorted(ram)}")
    if meta["max_size"] != rb.max_size:
        raise ValueError(f"{dump_dir}: dump buffer size {meta['max_size']} != the buffer's {rb.max_size}")
    for k in ram + ["_traj_success_mask"]:
        dst = rb._traj_success_mask if k == "_traj_success_mask" else rb._dict[k]
        src = np.load(os.path.join(dump_dir, k + ".npy"), mmap_mode="r")
        if src.shape != dst.shape or src.dtype != dst.dtype:
            raise ValueError(f"{dump_dir}/{k}.npy is {src.shape} {src.dtype}, the buffer's field is {dst.shape} {dst.dtype}")
        dst[...] = src
    rb.pointer, rb.size, rb.total_added = meta["pointer"], meta["size"], meta["total_added"]
    return meta
