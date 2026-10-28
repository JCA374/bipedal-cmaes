#!/usr/bin/env bash
# Stops whichever run is training once its held-out champion has plateaued, so the queue
# moves on (finalize.sh, then next_run.sh). Rule: after MIN_GEN generations, the champion
# gained less than MIN_GAIN points over the last WINDOW generations.
#   nohup ./plateau_watch.sh > plateau_watch.log 2>&1 &
cd "$(dirname "$0")"
MIN_GEN=${MIN_GEN:-400}; WINDOW=${WINDOW:-200}; MIN_GAIN=${MIN_GAIN:-5}
log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "watching: stop when gen >= $MIN_GEN and champion gain < $MIN_GAIN over $WINDOW gens"
while :; do
  # the training parent process (its workers have a train.py parent)
  for pid in $(pgrep -f "[t]rain.py --name"); do
    kill -0 $pid 2>/dev/null || continue          # gone meanwhile (e.g. workers of a stopped run)
    ppid=$(ps -o ppid= -p $pid 2>/dev/null | tr -d " ") ; [ -n "$ppid" ] || continue
    ppid=$(ps -o ppid= -p $pid | tr -d ' ')
    ps -o args= -p $ppid | grep -q "train.py" && continue
    run=$(ps -o args= -p $pid | sed -E 's/.*--name ([^ ]+).*/\1/')
    verdict=$(.venv/bin/python - "$run" "$MIN_GEN" "$WINDOW" "$MIN_GAIN" <<'PY'
import csv, sys
run, min_gen, window, min_gain = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
rows = list(csv.DictReader(open(f"runs/{run}/stats.csv")))
g = int(rows[-1]["generation"])
champ = lambda upto: max([float(r["best_score"]) for r in rows if r["best_score"] and int(r["generation"]) <= upto] or [-1e9])
now, then = champ(g), champ(g - window)
gain = now - then
print(("PLATEAU" if g >= min_gen and gain < min_gain else "ok") + f" {run} gen {g} champion {now:.1f}, +{gain:.1f} over {window} gens, sigma {rows[-1]['sigma']}")
PY
)
    case "$verdict" in
      PLATEAU*)
        log "$verdict -> stopping $run"
        for q in $(pgrep -f "[t]rain_queue.sh"); do
          tr "\0" "\n" < /proc/$q/environ 2>/dev/null | grep -qx "RUN=$run" && kill $q && log "  stopped train_queue.sh ($q)"
        done
        kill -TERM $pid
        for i in $(seq 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done
        log "  train.py stopped: $(grep -E 'Interrupted|Done at' runs_$run.log | tail -2 | tr '\n' ' ')"
        printf '\n## %s stopped on a plateau (%s)\n- %s. Rule: after %s gens, < %s points of held-out gain over %s gens (plateau_watch.sh). The queue moves on.\n' \
          "$run" "$(date '+%Y-%m-%d %H:%M')" "${verdict#PLATEAU }" "$MIN_GEN" "$MIN_GAIN" "$WINDOW" >> TRAINING_LOG.md
        ;;
      *) log "$verdict" ;;
    esac
  done
  pgrep -f "[n]ext_run.sh" >/dev/null || pgrep -f "[t]rain_queue.sh" >/dev/null || pgrep -f "[c]urriculum" >/dev/null || { log "nothing left to watch"; break; }
  sleep 300
done
