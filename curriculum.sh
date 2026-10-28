#!/usr/bin/env bash
# Curriculum: 1 balance -> 2 step (march) -> 3 walk -> 4 walk fast. Each stage starts from the previous
# stage's champion and moves on when it passes experiments/stage_check.py, plateaus (plateau_watch.sh),
# or hits its time cap.     nohup ./curriculum.sh > curriculum.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
P=${PREFIX:-c}
GAIT="--gait-weight 40 --offset-weight 20 --balance-leading 1 --gait-per-distance 1"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# name | check | cap hours | args for the first chunk
STAGES=(
  "${P}1-balance|balance|1|--policy symclock --objective balance --train-steps 500"
  "${P}2-march|march|3|--init-from runs/${P}1-balance --sigma0 0.2 --objective march --train-steps 600"
  "${P}3-walk|walk|6|--init-from runs/${P}2-march --sigma0 0.2 $GAIT"
  "${P}4-fast|none|6|--init-from runs/${P}3-walk --sigma0 0.1 --fall-cost 100 $GAIT"
)
for st in "${STAGES[@]}"; do
  IFS="|" read -r RUN CHECK CAP ARGS <<< "$st"
  log "=== stage $RUN (check: $CHECK, cap ${CAP} h)"
  for p in $(ps -eo pid,args | awk '$3=="dashboard.py" && /--port 8771/{print $1}'); do kill $p; done
  start=$(date +%s); outcome="time cap"
  while :; do
    if [ -f runs/$RUN/checkpoints/latest.pkl.gz ]; then R="--resume"; else R="$ARGS"; fi
    $PY -u train.py --name $RUN $R --hours 0.33 --workers ${WORKERS:-4} >> runs_$RUN.log 2>&1 || { log "train.py failed"; tail -5 runs_$RUN.log; exit 1; }
    pgrep -f "dashboard.py --run runs/$RUN" >/dev/null || (setsid nohup $PY dashboard.py --run runs/$RUN --port 8771 > dashboard_cur.log 2>&1 &)
    log "  $(grep -E '^gen' runs_$RUN.log | tail -1 | cut -c1-120)"
    if [ "$CHECK" != none ]; then
      res=$($PY experiments/stage_check.py $CHECK runs/$RUN); rc=$?
      log "  check: $res"
      [ $rc -eq 0 ] && { outcome="passed"; break; }
    fi
    grep -q "PLATEAU $RUN " plateau_watch.log 2>/dev/null && { outcome="plateau"; break; }
    [ $(( $(date +%s) - start )) -ge $(( CAP * 3600 )) ] && break
  done
  log "=== $RUN done: $outcome"
  printf '\n- curriculum %s: %s (%s)\n' "$RUN" "$outcome" "$(date '+%Y-%m-%d %H:%M')" >> TRAINING_LOG.md
done
log "CURRICULUM DONE"
