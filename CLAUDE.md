# bipedal-cmaes: notes for Claude

CMA-ES on the weights of a fixed MLP for BipedalWalker-v3. `PLAN.md` = design, `README.md` = commands,
`TRAINING_LOG.md` = every change. Use `.venv/bin/python`.

## Hard requirement
The policy has ≥1 hidden layer: asserted in `train.main`, and tested in `tests/test_common.py`.

## Gotchas (don't regress)
- `common.fresh_reset` makes a new Box2D world each reset; a reused world is NOT reproducible.
- Training fitness is shaped: the env's -100 fall is replaced by `--fall-cost` (20), stall cost 0.
  v0 (raw -100 + stall 100) dived, v1 (fall 0 + stall 100) lunged, v2 (fall 0, stall 0) fell
  early to save torque. Reported scores are always raw, from full episodes.
- Resume uses the CURRENT args/defaults plus `overrides.json`; never pickled args. Changing a
  default changes resumed runs, so pin values in `overrides.json`.
- A popsize override is applied once (`state["popsize_applied"]`); IPOP restarts may double λ later.
- Seeds: training < 1e6, HOF 1e7+, evaluate 2e7+, record 3e7+, compare 4e7+.

## Machine
- i5-8250U 4C/8T, ~4.5 GB free, no GPU. 4 workers ≈ 15.9k steps/s; 6 adds ~2%.
- Never `pkill -f <pattern>` where the pattern appears in your own command line (it kills
  your shell); use `pgrep -f "[t]rain..."` + `kill <pid>`.
