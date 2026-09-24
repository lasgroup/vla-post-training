# Blast radius

## Duplication sweep (CLAUDE.md clone families)

`grep -rn "scores.min(axis=0)" src/rl/` found the defect in **two** copies — the
known `advantage_weighted_sft_learner.py` <-> `best_of_n_learner.py` best-of-N
scoring clone family. Both are fixed. Fixing one and not the other is this
codebase's signature bug class.

| file | line (pre-fix) |
|---|---|
| `src/rl/advantage_weighted_sft/advantage_weighted_sft_learner.py` | 569 |
| `src/rl/best_of_n/best_of_n_learner.py` | 501 |

Other `min(axis=0)` hits in `src/rl/` are unrelated and untouched: the two
`summarize_critic_values` implementations (`advantage_weighted_sft/update_critic.py:84`,
`best_of_n/update_critic.py:176`) already branch on `critic_reduction`, and the
conservative-advantage head gates in `ogpo/update_actor.py:83,300` and
`advantage_weighted_sft/update_actor.py:69` are a different mechanism.

## Inheritance sweep

`AdvantageWeightedSFTLearner.sample_actions` reaches four subclasses. None
overrides `sample_actions`; `BestOfNLearner` carries its own copy of the block
(hence the second fix).

## Behavioural scope

The fix reads `rl.critic.reduction`, whose dataclass default is `"min"`
(`config.py:128`). Every config that leaves it at the default — including
`pi05_libero_online_ogpo_sft` — is **bit-identical** to before. Only configs that
set `reduction="mean"` change, which today is `pi05_libero_online_ogpo_ref`
(`config.py:734`).
