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

## Gait diagnosis and the queued "real walking" run (v4-walk)
- The v3 champion (gen ~220, 271 held-out) steps alternately (85-90% of touchdowns alternate)
  but **the feet never pass each other**: foot 2 is in front 100% of the time, 0 front-foot changes
  per episode on 3 tracks; hip 2 stays forward (+0.03..+1.2). A skipping gait with leg 2 as a
  forward prop. Nothing in the reward asks for crossing steps, and CMA-ES (σ ≈ 0.17 then) won't
  cross the clumsy in-between gaits on its own.
- Added (backward compatible, checked by resuming a copy of v3 for one generation):
  `--policy symclock` (one controller shared by both legs, leg 2 with legs swapped and the clock
  shifted by π; learned period; 931 params), gait tracking in every episode (crossings,
  front-foot changes, foot-2-in-front share; `gait.csv` per run), `--cross-bonus` (training
  only), resume drops stats/gait rows past the checkpoint after a crash.
- `next_run.sh` waits for v3 + finalize.sh, then trains v4-walk (symclock, cross bonus 0.5) for
  8 h and compares with v3 on the same 100 tracks, score and gait. Results stay local.

## v3-f20 stopped on a plateau (2026-10-27 19:18)
- v3-f20 gen 538 champion 287.0, +4.5 over 200 gens, sigma 0.01742. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.

## v4-walk stopped on a plateau (2026-10-28 00:44)
- v4-walk gen 779 champion 260.4, +4.6 over 200 gens, sigma 0.00402. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.

## v4-walk (auto-summary from next_run.sh)
- symclock policy (one shared leg controller + learned clock, 931 params), cross bonus 0.5, 8 h; champion gen 669.
- Same 100 comparison tracks: v4-walk 236.2 ± 71.71 (falls 11) vs v3-f20 270.23 ± 63.55 (falls 7);
  paired diff -34.03 (95% CI [-53.72, -14.47]), v4-walk better on 7/100.
- Gait: lead changes per episode 0.03 vs 0.02; crossing steps 0.03 vs 0.02;
  foot 2 in front 0.995 vs 0.997.

## Final results and diagnosis (2026-10-28)
- **v3-f20 (main) on 100 unseen tracks: 264.4 ± 66.7**, 11 falls, 89 finished, 0 timeouts, 0 runs ≥ 300 (not solved).
  Paired vs NEAT on 100 comparison tracks: 270.2 vs 179.5, +90.7 (95% CI 73.0 to 107.8), better on 93/100.
- Ceiling: the best fall-free episode scores 291.1 (median 286.9). Progress ≈ 364 minus effort ≈ 75
  leaves < 300 even without falls, so reaching 300 needs effort below ~60 AND fewer falls.
  Effort fell 159 -> 75 during training (wasted share 43% -> 16%).
- Falls (fall analysis, 23 of 300): 65% happen on uphill terrain (vs 17% of the time spent uphill);
  forward speed starts dropping ~55 steps (~1 s) before the fall, then the hull tips.
- **v4-walk did not learn to walk**: foot 2 leads ~99-100% from gen ~200 on; 236.2 vs 270.2 on the same
  tracks (significantly lower). The clock is essential (frozen clock: -129 vs 257; learned period 29 steps),
  but the shared controller still takes fixed roles because each leg sees where the other leg is
  ("if I'm in front, reach; if behind, push"). The step bonus never got a foothold (best candidate ~0
  crossings). Stopped by plateau_watch at gen 780. Lesson 9 corrected accordingly.

## v5-fresh stopped on a plateau (2026-10-28 06:37)
- v5-fresh gen 491 champion -68.4, +3.5 over 200 gens, sigma 0.05742. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.

## Gait experiments v5 (2026-10-28 morning)
- Fall penalty check (v5-warm search distribution, 180 episodes): the few less-lopsided candidates ALL fell, at
  x ≈ 8. A fall's real cost is the lost progress (median ≈ 272 points), not the training fall cost (20).
- **v5-fresh** (symclock, fixed per-episode gait penalty 40·balance + 20·|offset|, fall cost 20): learned to
  ALTERNATE (foot 2 leads 50%, ~14 lead changes) but fell in 100% of episodes at x ≈ 14 ("stumble forward").
  Bug in my balance term: standing with feet apart counted as fully lopsided (penalty 40 + offset), so standing
  scored worse than stumbling and falling, the same kind of trap as v0-v2. Stopped by plateau rule at gen 491.
- **v5-warm** (from v3, σ0 0.15, same penalty): kept skipping (foot 2 leads 99%, offset 1.9 -> 1.65 in 124 gens),
  improved raw score (284) and paid the penalty. Stopped at gen ~125 to give v6 the CPU.

## v6-fresh
- Gait penalty as a tax on lopsided DISTANCE: (40·balance + 20·|offset|) × share of the course covered
  (--gait-per-distance 1), balance measured only over steps where one foot leads (--balance-leading 1).
  Standing pays nothing; ordering on 3 tracks: lunge -18 < stand -11 < few alternating steps then fall +16 <
  skip full course 210; a full-course alternating walk would be worth ~77 more than the skip.
- symclock, fall cost 20 (the value that worked for v3), λ 48, K 5.

## v6-fresh stopped on a plateau (2026-10-28 07:27)
- v6-fresh gen 458 champion -100.0, +0.0 over 200 gens, sigma 0.00763. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.
- **v6-fresh stuck standing** (x ≈ 5.6, 0 falls from gen ~100, σ 0.5 -> 0.008); plateau stop at gen 459. The
  NEAT trap: from a stable stance every perturbation that starts a step risks a fall. v3 escaped this at gen ~55;
  v1/v2 never did, so escaping is partly chance.

## v7-alt and v6b-fresh (2026-10-28 08:15)
- v7-alt: warm start from the v5-fresh champion, which already alternates (16.8 lead changes, foot-2 lead share
  0.59, offset -0.08) but falls at x ≈ 17; v6 reward (gait tax per distance, fall cost 20), σ0 0.2. Every extra
  unit walked while alternating pays ~+4.3, so learning balance is the path.
- v6b-fresh: same as v6 with seed 1, to test whether "stuck standing" was bad luck.

## Curriculum (user's idea, 2026-10-28 ~08:40): balance -> step -> walk -> walk fast
- Every gait run so far had to find balance, alternation and progress at once, and each settled for a
  shortcut (skip, stumble, stand). The curriculum gives each skill its own training reward and warm-starts
  the next stage from the previous champion (symclock policy throughout):
  1. balance (500-step episodes, no early stop): 100·survival − 100·fell − 50·mean|hull angle| − 0.5·effort.
     Pass: 0 falls on 20 held-out tracks, mean |angle| < 0.15.
  2. march (600 steps): survival − fall + 3·front-foot changes (≤ 40) + alternating touchdowns (≤ 60)
     − 30·|mean foot offset| − 0.5·effort. Pass: ≥ 12 lead changes, ≤ 1 fall, |offset| < 0.5.
  3. walk: the v6 reward (fall cost 20, gait tax per distance). Pass: x ≥ 60, ≤ 4 falls, lopsidedness ≤ 0.3.
  4. walk fast: raw reward (fall 100) + gait tax per distance; runs until plateau / 6 h.
- Each stage advances on pass, plateau or time cap (1 / 3 / 6 / 6 h). v6b-fresh stopped to free the CPU;
  v7-alt keeps running for comparison.

- curriculum c1-balance: passed (2026-10-28 08:35)

## Curriculum attempt 1 found the "tap" shortcut; attempt 2 (prefix d) launched ~10:05
- c1-balance passed; its champion stands with the feet 2.93 units apart (foot 2 always in front).
- c2-march (gen ~372): narrowed the stance to |offset| 0.93 and taps the feet alternately (~70 alternating
  touchdowns, above the 60 cap) with 0 front-foot changes, lopsided 1.0; σ 0.013 (collapsed).
- Attempt 2: stage 1 balance_c = balance − 30·|offset| (pass: |offset| < 0.4); stage 2 step = survival − fall
  + 3·lead changes (≤40) + 2·front-foot touchdowns (≤40; a touchdown counts only if that foot is in front and
  it alternates legs) − 30·|offset| − 0.5·effort. Under the new step reward the c2 tapper scores 47 (was 105),
  the split stander −17, a non-falling stepper up to ~300. Stages 3-4 unchanged. Attempt 1 keeps running.

- curriculum d1-balance: time cap (2026-10-28 10:57)

## e2-march stopped on a plateau (2026-10-28 11:31)
- e2-march gen 517 champion 8.4, +4.8 over 200 gens, sigma 0.00504. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.

- curriculum e2-march: plateau (2026-10-28 11:31)

## e3-walk stopped on a plateau (2026-10-28 12:17)
- e3-walk gen 411 champion -63.4, +2.4 over 200 gens, sigma 0.00985. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.

- curriculum e3-walk: plateau (2026-10-28 12:17)

- curriculum f1-walk-s2: passed (2026-10-28 12:55)

- curriculum f2-walk-s1: passed (2026-10-28 13:15)

- curriculum f3-walk-s05: passed (2026-10-28 13:35)

- curriculum f4-walk: passed (2026-10-28 14:15)

- curriculum f5-fast: plateau (2026-10-28 17:28)

## f5-fast stopped on a plateau (2026-10-28 17:28)
- f5-fast gen 410 champion 276.1, +3.9 over 200 gens, sigma 0.00692. Rule: after 400 gens, < 5 points of held-out gain over 200 gens (plateau_watch.sh). The queue moves on.
