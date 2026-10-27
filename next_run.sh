#!/usr/bin/env bash
# The "real walking" experiment, queued after the main run:
#   v4-walk = symclock policy (one shared leg controller + clock) + a small crossing-step bonus.
# Waits for finalize.sh, trains for HOURS, then compares with the v3 champion on the same
# 100 tracks (score + gait). Results stay local (not pushed) for review.
#
#   nohup ./next_run.sh > next_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=${RUN:-v4-walk}
BASE=${BASE:-v3-f20}
HOURS=${HOURS:-8}
BONUS=${BONUS:-0.5}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for the main run's training and finalize.sh"
while pgrep -f "[t]rain_queue.sh" >/dev/null || pgrep -f "[f]inalize.sh" >/dev/null || pgrep -f "[t]rain.py --name $BASE" >/dev/null; do sleep 60; done
log "main run done: $(grep -E 'FINALIZE DONE|failed' finalize.log | tail -1)"

pkill -f "[d]ashboard.py" ; sleep 1
setsid nohup $PY dashboard.py --run runs/$RUN > dashboard.log 2>&1 &
log "dashboard now follows runs/$RUN (http://localhost:8770)"

DL=$(date -d "+$HOURS hours" +%H:%M)
log "training $RUN until $DL: --policy symclock --cross-bonus $BONUS"
RUN=$RUN DEADLINE=$DL TARGET=100000 EXTRA="--policy symclock --cross-bonus $BONUS" ./train_queue.sh > train_queue.log 2>&1
log "$(tail -1 train_queue.log)"

log "evaluation: $RUN on 100 unseen tracks; both champions on the same 100 comparison tracks"
$PY evaluate.py --run runs/$RUN --episodes 100 --out runs/$RUN/eval/best.json | grep -E '"(mean|std|falls|finished|n_ge_300)"'
$PY evaluate.py --run runs/$RUN --episodes 100 --seed-kind compare --out results/${RUN}_compare.json > /dev/null
$PY evaluate.py --run runs/$BASE --episodes 100 --seed-kind compare --out results/${BASE}_compare.json > /dev/null
$PY experiments/compare.py results/${RUN}_compare.json results/${BASE}_compare.json --out results/compare_${RUN}_vs_${BASE}.json \
  | grep -vE '^ *[0-9\[\]-]' | head -40
$PY experiments/export_episode.py --run runs/$RUN --label "$RUN champion" --seed 30000008 --out results/${RUN}_champion.json | tail -1

$PY - <<PYEOF >> TRAINING_LOG.md
import json
c = json.load(open("results/compare_${RUN}_vs_${BASE}.json"))
e = json.load(open("runs/${RUN}/eval/best.json"))
ga, gb = c.get("gait_a", {}), c.get("gait_b", {})
print(f"""
## ${RUN} (auto-summary from next_run.sh, needs review)
- symclock policy (one shared leg controller + learned clock, 931 params), cross bonus ${BONUS}, ${HOURS} h.
- 100 unseen tracks: {e['mean']} ± {e['std']}, falls {e['falls']}, finished {e['finished']}, ≥300: {e['n_ge_300']}.
- Paired vs ${BASE} on 100 comparison tracks: {c['mean_a']} vs {c['mean_b']}, diff {c['diff_mean']:+} (95% CI {c['diff_ci95']}), better on {c['a_better_on']}/100; falls {c['falls_a']} vs {c['falls_b']}.
- Gait: crossing steps per episode {ga.get('crossings')} vs {gb.get('crossings')}; front-foot changes {ga.get('lead_changes')} vs {gb.get('lead_changes')}; foot 2 in front {ga.get('front2_share')} vs {gb.get('front2_share')} of the time.""")
PYEOF
tail -6 TRAINING_LOG.md
log "NEXT RUN DONE (results in results/, not published)"
