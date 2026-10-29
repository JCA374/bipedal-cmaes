# Continue here (paused 2026-10-29 ~05:20)

Paused at the user's request with a result they're happy with. Everything was stopped cleanly; nothing is running.

## The result
- **Final walker:** `models/final-walker/` (params.npy + meta.json, tracked in git): attempt 11's champion,
  `symclock` policy (one shared leg controller + learned clock), 26 → 64 → 2, 1,859 parameters.
  - 100 comparison tracks (seeds 4e7+): **281.8 ± 38.2, 4 falls**, ≈ 903 steps, ≈ 102 front-foot changes per run,
    foot 2 leads 51% of the time, effort 73.6 (`models/final-walker/eval_compare100.json`).
  - 50-track goal check: 277.0, 4 falls, lopsidedness 0.07: all four goal requirements met (`goal_check_50.json`).
  - The old skipping walker (`runs/v3-f20`) on the same 100 tracks: 270.2 ± 63.6, 7 falls, foot 2 always in front.
- Site: https://jca374.github.io/bipedal-cmaes/ (lessons 1-9, Curriculum log, Podcast). Repo: JCA374/bipedal-cmaes.
- How the work was done (check-in loop, journal, site + podcast): `METHOD.md`.

## Resume a run
All curriculum runs have checkpoints. The last two (same start, fall cost 300 vs 200, mean-only champions):
```bash
cd ~/code/games/bi-pedal/bipedal-CMA-ES
setsid nohup ./curriculum_m.sh >> curriculum_m.log 2>&1 &     # attempt 11 (resumes m1-fast)
setsid nohup ./curriculum_n.sh >> curriculum_n.log 2>&1 &     # attempt 12 (resumes n1-fast)
setsid nohup ./plateau_watch.sh > plateau_watch.log 2>&1 &
setsid nohup ./curriculum_publish.sh > curriculum_publish.log 2>&1 &   # site updates every 15 min (edit its END date)
.venv/bin/python dashboard.py --run auto:curriculum_m.log --port=8771  # local dashboard
```
To start something new from the final walker: `train.py --name X --init-from models/final-walker ...`
(see `docs/data/curriculum_attempts.json` for every attempt's exact settings and `TRAINING_LOG.md` for why).

## Open ideas (not tried yet)
- Falls (about 4%) are now the main loss; a fall analysis (`experiments/record_episodes.py` + `analyze_falls.py`)
  of the final walker would show where they happen (uphill was the main cause for the skipper).
- A recurrent hidden state, or action repeat 2 (both untried). A second hidden layer was tried and was too
  fragile for CMA-ES to fine-tune (`common.deepen`, sep-CMA via `--diagonal`).
- Always verify a new champion on 100 tracks before believing it (the winner's curse bit three times).
