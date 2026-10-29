# bipedal-cmaes

CMA-ES evolves the 932 weights of a fixed 24→32→4 tanh network for **BipedalWalker-v3**,
gradient-free, on a 4-core laptop. The goal is ≥300 on 100 unseen tracks; for comparison,
NEAT reached 173–179, PPO 272 and TD3 308–319.

**Site with the explainer video and interactive lessons:** https://jca374.github.io/bipedal-cmaes/

## Commands
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                              # tests
.venv/bin/python bench.py --workers 1 4 6                  # steps/s
.venv/bin/python train.py --name myrun --hours 1           # train (resume with --resume)
RUN=myrun DEADLINE=08:00 nohup ./train_queue.sh > train_queue.log 2>&1 &   # 1 h chunks
.venv/bin/python evaluate.py --run runs/myrun --episodes 100               # final score
.venv/bin/python experiments/export_episode.py --run runs/myrun --out docs/data/episodes/champion.json
.venv/bin/python make_site.py --run runs/myrun             # site data
.venv/bin/python dashboard.py --run runs/myrun             # live local dashboard: http://localhost:8770
```
Change settings between chunks with `runs/<name>/overrides.json`
(`k`, `popsize`, `hof_seeds`, `hof_top`, `stall_penalty`, `speed_bonus`, `fall_cost`, `reason`).

## Files
- `common.py`: env, observation scaling, the numpy MLP over a flat θ, `run_episode`, seed ranges, save/load.
- `train.py`: CMA-ES loop (pycma ask/tell), multiprocessing, checkpoint/resume, hall of fame, milestones, snapshots.
- `evaluate.py`: 100 full episodes with raw reward: mean/std/falls/≥300/timeouts.
- `make_site.py`, `docs/`: the GitHub Pages site (static HTML + vanilla JS demos).
- `PLAN.md`: the design; `TRAINING_LOG.md`: every change and why (including three failed starts).

## How the work was run

See [METHOD.md](METHOD.md) (2-hour check-ins, one change at a time, verification, the log page and the podcast) and [CONTINUE.md](CONTINUE.md) (the paused state and how to resume).
