# Training log

Every change to a run, with the generation and the reason. Scores are raw env reward on
held-out terrains (HOF: 20 terrains, seeds 1e7+) unless stated otherwise.

## Setup (2026-10-27)
- bench: 4 workers 15.9k steps/s, 6 workers 16.2k (hyperthreads), ~78 MB RSS/worker. Use 4.
- Found: BipedalWalker on a reused env is **not reproducible** (same seed + policy, 3rd reset
  gave -119.8 vs -118.4). `common.fresh_reset` builds a new Box2D world each reset (no cost);
  tested in `test_episodes_are_reproducible_on_a_reused_env`.

## v0-dive (gens 0-178, stopped)
- 24-32-4 tanh, λ=48, K=5, σ0=0.5, stall cost 100, training fitness = raw reward - stall cost.
- **Stuck in a "dive" optimum**: from gen ~60, 100% of episodes fall at x≈6.7 (start 4.7),
  fitness -98..-101 for every candidate, σ shrinking 0.67 -> 0.17. Cause: the stall cost made
  standing still (≈-108) worse than lunging forward and falling (≈-99), and every lunge scores
  about the same, so there is no slope to follow. NEAT had the opposite trap (standing still).
- Fix for the next run: training fitness swaps the env's -100 for `--fall-cost` (0). Fitness is
  then distance walked minus torque, so getting further before falling always helps, and not
  falling is still rewarded through the progress it keeps earning. Reported scores stay raw.

## baseline (from gen 0)
- 24-32-4 tanh (932 params), λ=48, K=5, σ0=0.5, stall cost 100, fall cost 0.
- **Stuck again (gens 0-209, archived as runs/v1-lunge)**: 100% falls at x≈7.4, fitness ≈+2.5,
  σ collapsed 0.5 -> 0.084. A side run with σ0=0.1 (runs/s01, 38 gens) behaved the same.
  Cause: the stall cost (100). The obvious stepping stone, "lunge, catch yourself and stand", is cut
  by the stall detector and scores ≈ -90, worse than lunging and falling (+2.5). So the walker never
  learns to recover from the first lunge.

## v2 (from gen 0)
- Same as v1 but stall cost 0 (fall cost 0). Then "lunge and stand" (≈+8) beats "lunge and fall"
  (+2.5), and every further step adds progress reward: a monotone path. Standing can't beat walking,
  because nothing pays for not falling except the progress it keeps earning. Early stop is still on (it saves steps).
- **Stuck again (archived as runs/v2-torque, 92 gens)**: still 100% falls. Cause: the torque cost.
  A saturated random net pays ≈0.11/step. Falling at step ~50 ends that bill, while catching
  yourself and standing until the stall detector fires (150+ steps) costs ≈17. With fall cost 0,
  falling early is the cheapest thing to do.

## v3: fall cost 20 vs 40 (side by side, 4 workers each)
- Fall cost must exceed the torque saved by falling early, but stay far below -100 so standing
  still doesn't become a dead end (NEAT's trap). Tested 20 and 40, ~45 gens each:
  both left the fall regime (falls 100% -> ~12%, stalls ~88%). f20 was ahead (best train fitness
  -5.8 vs -7.8, champion -78 vs -112), so f40 was stopped at gen 48 and **v3-f20 continues as the main run**.
- train.py default fall cost is now 20 (resume uses current defaults: it would otherwise
  silently have switched v3-f20 back to 0). Also pinned in runs/v3-f20/overrides.json.
- **Breakthrough at gen ~60**: best training fitness jumped from ≈0 to +120-135 and mean x
  from 6 to 16 within a few generations (the first real steps). At gen ~70 the champion,
  which was the **CMA mean**, scored **170 ±12 on the 20 held-out tracks with 0 falls**,
  already level with NEAT's final 173 (12 h). Gens now take ~20-30 s (longer episodes).
- Ops fix: v3-f40 kept running after "stopping" it. Background jobs started from a
  non-interactive shell have SIGINT ignored, so `kill -INT` did nothing and 8 workers shared
  the CPU for ~20 min. train.py now turns SIGTERM/SIGINT into a clean checkpointed stop
  (`test_sigterm_saves_checkpoint`). f40 was killed at gen 115 (stalls 94%, never walked).

## Preliminary check, v3-f20 at gen ~105 (champion = CMA mean, 194 on 20 HOF tracks)
- 100 unseen tracks (seeds 2e7+): **204.5 ± 17.1**, 1 fall, 69 finished, 30 timeouts at x ≈ 87.2
  (finish line 88.7), 0 runs ≥ 300.
- Paired vs NEAT champion on 100 comparison tracks (4e7+): CMA 202.7 vs NEAT 179.5, diff +23.2,
  CMA better on 63/100; falls 1 vs 10.
- **The ceiling is effort, not falls or speed**: it finishes, but finishing (~360 progress) only
  nets ~204, so ≈150 goes to the torque cost (max 179). Raw reward already includes effort, so
  training keeps optimising it; watch whether |action| falls over the coming generations.
- Fixed: `run_episode` returned a numpy bool for `fell` (JSON dump in evaluate.py failed).

## Score analysis (failure-analysis-report method), champion at gen ~110, 200 analysis tracks
`experiments/record_episodes.py` + `analyze_falls.py` (3 falls in 200: too few to conclude) +
new `analyze_score.py` (this walker's real failure is a low score, not falling):
- finished 122, timed out 75 (ending at x ≈ 87.2, only 1.5 short of the line), fell 3.
- earns ≈358 from progress, spends ≈154 on effort. **43% of the effort (≈67 points) goes to
  joints that aren't moving** (|speed| < 5% of max): hip 2 wastes 39 of 43 (held against a limit
  at full power), knee 2 20 of 34. Leg 2 looks used as a stiff prop.
- speed separates finishes from time-outs (AUC 0.02), but a time-out only costs ~6 points.
  Hull-angle variability correlates −0.89 with the score.
- Lead: the main lever toward 300 is wasted effort on leg 2 (~60 points). Raw reward already
  includes effort; if training plateaus, try a training-only effort weight > 1 (paired comparison).
