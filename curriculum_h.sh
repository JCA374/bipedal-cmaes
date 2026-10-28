#!/usr/bin/env bash
# Curriculum attempt 6: a wider network (64 hidden units, Net2Net-widened from f5-fast), real fall cost, gait tax.
#   nohup ./curriculum_f.sh > curriculum_h.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
P=h
GAIT="--gait-weight 40 --offset-weight 20 --balance-leading 1 --gait-per-distance 1"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# name | check | cap hours | support | args for the first chunk
STAGES=(
  "${P}1-fast|none|10|0|--init-from runs/f5-fast --widen 64 --sigma0 0.05 --popsize 64 --fall-cost 100 $GAIT"
)
for st in "${STAGES[@]}"; do
  IFS="|" read -r RUN CHECK CAP SUP ARGS <<< "$st"
  log "=== stage $RUN (check: $CHECK at support $SUP, cap ${CAP} h)"
  start=$(date +%s); outcome="time cap"
  while :; do
    if [ -f runs/$RUN/checkpoints/latest.pkl.gz ]; then R="--resume"; else R="$ARGS"; fi
    $PY -u train.py --name $RUN $R --hours 0.33 --workers ${WORKERS:-4} >> runs_$RUN.log 2>&1 || { log "train.py failed"; tail -5 runs_$RUN.log; exit 1; }
    log "  $(grep -E '^gen' runs_$RUN.log | tail -1 | cut -c1-120)"
    if [ "$CHECK" != none ]; then
      res=$($PY experiments/stage_check.py $CHECK runs/$RUN $SUP); rc=$?
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
