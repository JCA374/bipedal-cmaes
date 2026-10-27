#!/usr/bin/env bash
# Train in 1 h resumable chunks until the champion reaches TARGET or DEADLINE passes.
# Settings change between chunks only via runs/$RUN/overrides.json (logged in TRAINING_LOG.md).
#
#   RUN=baseline DEADLINE=08:00 nohup ./train_queue.sh > train_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=${RUN:-baseline}
RUN_DIR=runs/$RUN
TARGET=${TARGET:-300}
DEADLINE=${DEADLINE:-08:00}
WORKERS=${WORKERS:-4}
EXTRA=${EXTRA:-}           # first-chunk-only args, e.g. "--hidden 24 24"
deadline_ts=$(date -d "$DEADLINE" +%s)
[ "$deadline_ts" -lt "$(date +%s)" ] && deadline_ts=$(date -d "tomorrow $DEADLINE" +%s)

log() { echo "[$(date '+%H:%M:%S')] $*"; }
best_score() { $PY -c "import json;print(json.load(open('$RUN_DIR/best/meta.json'))['mean'])" 2>/dev/null || echo -999; }

while :; do
  left=$(( deadline_ts - $(date +%s) ))
  best=$(best_score)
  if $PY -c "import sys; sys.exit(0 if float('$best') >= $TARGET else 1)"; then
    log "target $TARGET reached (champion $best)"; break
  fi
  if [ "$left" -lt 600 ]; then log "deadline $DEADLINE reached (champion $best)"; break; fi
  hours=$($PY -c "print(round(min(1.0, $left/3600 - 0.05), 2))")
  if [ -f "$RUN_DIR/checkpoints/latest.pkl.gz" ]; then R="--resume"; else R="$EXTRA"; fi
  log "chunk: ${hours} h, $WORKERS workers (champion $best)"
  $PY -u train.py --name "$RUN" $R --hours "$hours" --workers "$WORKERS" --stop-at "$TARGET" >> "runs_${RUN}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { log "train.py exited $rc, see runs_${RUN}.log"; tail -5 "runs_${RUN}.log"; break; }
done
log "TRAINING DONE: champion $(best_score)"
