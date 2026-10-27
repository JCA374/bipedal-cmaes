# Next project: CMA-ES on BipedalWalker-v3

A guide for starting a new project (suggested folder: `../bipedal-cmaes`). It covers what to build, how to build it, and what the NEAT project (`../bipedal-neat`) taught us. Copy this file into the new folder as `PLAN.md`.

## Goal
- Evolve the weights of a **fixed** neural network with CMA-ES so that it scores **≥ 300 averaged over 100 unseen terrains**. That is the "solved" level. NEAT reached 173–179; PPO reached 272; TD3 reached 308–319.
- It stays gradient-free, which keeps it a fair comparison with NEAT/PPO/TD3.
- **Hard requirement (same as NEAT): the policy has ≥ 1 hidden layer.** With a fixed architecture this holds by construction. Assert it in code and in a test anyway.

## Why CMA-ES (and not more NEAT)
NEAT plateaued for these reasons (details in `../bipedal-neat/TRAINING_LOG.md`):
1. **The time limit, not just falls.** No episode without a fall reached the end of the course. All of them timed out at x ≈ 74 of ~88 after 1600 steps. Scoring 300 requires finishing, which means walking faster.
2. **Speed and falls trade off.** Faster walkers pitch forward. A fall penalty halved falls (17% → 9%) but did not change the score (paired −5.4, 95% CI [−26.5, +16.0]).
3. **Noisy selection.** 3–5 terrains per genome against a per-terrain std of ±63–75. The "200" champion scored 180 on 20 terrains and 173–179 on 100.
4. **Tiny, memoryless networks.** The final champion had 4 hidden nodes and 30 connections, feed-forward. Random structural mutation also pruned capacity away.
5. **Sample-hungry.** 414M env steps over ~12 h.

CMA-ES fixes the architecture at a useful size, learns which weights should move together (the covariance), and adapts its step size by itself. It works well up to a few thousand parameters.

## Environment and stack
- `gymnasium[box2d]==0.29.1`, `numpy<2`, `cma` (pip), `pytest`, `opencv-python` (videos). Python venv in `.venv/`.
- `BipedalWalker-v3`: 24 observations, 4 actions in [−1, 1], 1600-step limit, −100 on a fall.
- Reuse from `../bipedal-neat/common.py`: `OBS_SCALE` + `scale_obs` (clip to [−1, 1]), the `run_episode` structure (reward, steps, fell), the hull-x stall detector.

## Design

### Policy
- An MLP with **one hidden layer, 24 → 32 → 4, tanh everywhere**: 932 parameters. A deeper variant to try later: 24 → 24 → 24 → 4 (1,324 params).
- A pure numpy forward pass on a flat parameter vector: `unflatten(θ) → (W1, b1, W2, b2)`. No torch.
- Optional later: **action repeat** (`--action-repeat 2`) and a **small recurrent state** (an Elman hidden layer). Both target the time-out and gait-rhythm problems.

### Optimiser
- `cma.CMAEvolutionStrategy(x0=zeros(n), sigma0=0.5, {"popsize": λ, "seed": ...})`.
- **λ = 32–48** (the default for n = 932 is ~24; bigger is better for noisy fitness). Use `ask()` → evaluate in parallel → `tell(solutions, -fitness)`. CMA minimises.
- When n ≥ ~1,000 and generations get slow, switch on `"CMA_diagonal": True` for the first N iterations, or use sep-CMA.
- Restarts: if σ collapses or there's no progress for ~150 gens, do an IPOP restart (double λ). Log every restart.

### Fitness (training only)
- **Mean over K terrains, K = 5**, with the same seeds for every candidate in a generation (common random numbers). This removes terrain luck from comparisons inside a generation.
- **Early stop: Δx < 1 over 150 steps** saves steps. But in NEAT, gens 0–69 got stuck at a "stand still" optimum of −7…−9, because standing still beats a −100 fall. Fix: on a stall, add a stall cost equal to the fall penalty (or count it as a fall). Only training fitness changes.
- Optional speed shaping if it times out again: reward the x reached at step 1600. Remember the score ceiling is set by finishing the course.
- **Report raw environment reward only**, from full episodes without early stop.

### Selection of the champion (the noise lesson)
- Every 10 generations, re-score the **CMA mean vector** plus the best ~3 candidates on **20 held-out terrains**, full episodes. The champion is the best on those 20. The CMA mean is often more robust than any single sample.
- Before claiming a result, run the final check: **100 unseen terrains**, mean ± std, fall count, and the number of runs ≥ 300.
- When comparing two policies, always use the **paired** comparison on the same 100 new terrains with a bootstrap CI (port `experiments/compare.py`).

### Seeds (no overlap)
- Training < 1e6 (`rng(base_seed + gen)`), hall of fame 1e7+, `evaluate.py` 2e7+, episode recording 3e7+, paired comparisons 4e7+.

## Files to build
```
common.py       env, scale_obs, MLP (flat params), run_episode, seed ranges
train.py        CMA loop, multiprocessing pool, stats.csv, checkpoint, HOF, overrides.json
evaluate.py     100 episodes, mean/std/falls/≥300 count, writes runs/<name>/eval/
play.py         window render of best/
record_video.py 600x400 50fps, stream frames to ffmpeg (never keep all frames in RAM)
bench.py        steps/s per worker count + RSS per worker
experiments/    record_episodes.py, analyze_falls.py, compare.py (port from bipedal-neat)
tests/          see Tests
```
Output per run, `runs/<name>/`: `stats.csv` (gen, best/mean/median fitness, σ, λ, hof_mean, env_steps_total, steps_saved, gen_seconds, wall_seconds, mem_avail_mb), `best/params.npy + meta.json`, `milestones/{first_positive,100,200,300}/`, `checkpoints/latest.pkl.gz` (pickle the whole `CMAEvolutionStrategy` plus the state dict), `run_config.json` with a `history` of every override.

## Step-by-step plan
1. **Phase 0, setup and benchmark.** Venv, `bench.py`. Check `uptime` first (other projects use all 8 threads). Go when ≥ 10k steps/s in total and ≥ 1.5 GB RAM free.
2. **Phase 1, smoke run.** `train.py --generations 3 --name smoke` creates every output file; the tests pass.
3. **Phase 2, baseline.** 24→32→4, λ = 32, K = 5, σ0 = 0.5, 2 h. Expect ~30–40 s per generation on 4 workers (~260k steps/gen at worst, less with early stop).
4. **Phase 3, overnight.** Resume in 1 h chunks (port `overnight.sh`, `DEADLINE`, `--stop-at 300`). Change settings only via `overrides.json`, one change at a time, logged in `TRAINING_LOG.md`.
5. **Phase 4, if it plateaus.** Try these in order, one per run: (a) λ 32 → 64; (b) action repeat 2; (c) a deeper net; (d) a recurrent hidden layer; (e) speed shaping. Use a paired comparison against the previous champion every time.
6. **Phase 5, analysis.** 100-episode eval, the fall analysis with the `failure-analysis-report` skill, comparison with NEAT/PPO/TD3 on score, env steps, wall clock and parameter count.

## Tests (write them first; they caught real bugs in NEAT)
- `unflatten(flatten(params))` round-trips; the forward pass matches a hand-written reference.
- The policy has ≥ 1 hidden layer (this asserts the hard requirement).
- Smoke train for 2 gens → checkpoint → **resume applies the CURRENT config/overrides** (never a pickled config) and continues the generation counter and the CMA state.
- `run_episode` with `early_stop=False` returns the raw total reward. Early stop never happens in evaluate/play/video.
- The seed ranges don't overlap.

## Lessons from bipedal-neat (don't repeat these)
- **Resume must use the current config.** Pickled configs silently override the edits you make.
- **Adaptive controllers wind up.** NEAT's compat threshold went 0.6 → 38. Clamp anything adaptive. For CMA, log σ and set `"tolx"`/`"tolfun"` sensibly; let restarts handle collapse.
- **10 evaluation terrains lie.** Use at least 20 for selection and 100 for claims. Use paired comparisons with CIs; a difference within ±20 is often noise.
- **Progress comes in bursts.** Don't kill a run after one flat hour; NEAT jumped at gens 99, 238 and 520.
- **Check the ceiling early.** Record the x reached at timeout. If good runs time out short of the end of the course, the problem is speed, not falls.
- **Fall analysis was worth it.** 90% forward pitch-overs after speed-ups; terrain was not the cause; the inputs were enough. Reuse `experiments/`.
- **Memory.** ~50 MB per worker, ~4 GB free. Stream video frames to ffmpeg (the site build was OOM-killed once). Run `free -h` during long runs; swap must not grow.
- **Shared CPU.** `--workers 4` when other jobs run, 6 when idle (8 is only ~9% faster; hyperthreads). Set `OMP_NUM_THREADS=1` in workers.
- **A fast inner loop matters.** NEAT got a 12× speed-up from a numpy forward pass. Keep the MLP numpy-only and test it.
- **Log everything that changes** in `TRAINING_LOG.md` with the generation and the reason, so the curve stays interpretable.

## Done when
- Champion ≥ 300 mean on 100 unseen terrains (seeds 2e7+), with ≥ 1 hidden layer, raw reward, full episodes.
- Or: a documented plateau with a paired-comparison history and a fall/timeout analysis that explains it.
