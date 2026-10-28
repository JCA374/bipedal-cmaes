#!/usr/bin/env bash
# Curriculum attempt 4 ("training wheels"): walk with an upright-torque support that fades out stage by stage,
# starting from the alternating stepper (e2-march), then walk fast on the real game.
#   nohup ./curriculum_f.sh > curriculum_f.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
P=f
GAIT="--gait-weight 40 --offset-weight 20 --balance-leading 1 --gait-per-distance 1"
log() { echo "[$(date '+%H:%M:%S')] $*"; }
# name | check | cap hours | support | args for the first chunk
STAGES=(
  "${P}1-walk-s2|walk|3|2|--init-from runs/e2-march --sigma0 0.2 --support 2 $GAIT"
  "${P}2-walk-s1|walk|3|1|--init-from runs/${P}1-walk-s2 --sigma0 0.15 --support 1 $GAIT"
  "${P}3-walk-s05|walk|3|0.5|--init-from runs/${P}2-walk-s1 --sigma0 0.15 --support 0.5 $GAIT"
  "${P}4-walk|walk|4|0|--init-from runs/${P}3-walk-s05 --sigma0 0.15 $GAIT"
  "${P}5-fast|none|6|0|--init-from runs/${P}4-walk --sigma0 0.1 --fall-cost 100 $GAIT"
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
