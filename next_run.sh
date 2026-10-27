#!/usr/bin/env bash
# The "real walking" experiment (lesson 8), queued after the main run:
#   v4-walk = symclock policy (one shared leg controller + clock) + a small crossing-step bonus.
# Waits for the main run and finalize.sh, trains for HOURS while publishing its progress to
# lesson 8 hourly, then compares with the final v3 champion on the same 100 tracks (score +
# gait), narrates and renders lesson 8's clip, and publishes the lesson.
#
#   nohup ./next_run.sh > next_run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
V=~/.claude/skills/voice-text/scripts/voice.sh
RUN=${RUN:-v4-walk}
BASE=${BASE:-v3-f20}
HOURS=${HOURS:-8}
BONUS=${BONUS:-0.5}
log() { echo "[$(date '+%H:%M:%S')] $*"; }
publish() {   # $1 = commit message; only the listed paths
  git add "${@:2}" && git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && log "pushed: $1"
}

log "waiting for the main run's training and finalize.sh"
while pgrep -f "[t]rain_queue.sh" >/dev/null || pgrep -f "[f]inalize.sh" >/dev/null || pgrep -f "[t]rain.py --name $BASE" >/dev/null; do sleep 60; done
log "main run done: $(grep -E 'FINALIZE DONE|failed' finalize.log | tail -1)"

pkill -f "[d]ashboard.py" ; sleep 1
setsid nohup $PY dashboard.py --run runs/$RUN > dashboard.log 2>&1 &
log "dashboard now follows runs/$RUN (http://localhost:8770)"

DL=$(date -d "+$HOURS hours" +%H:%M)
log "training $RUN until $DL: --policy symclock --cross-bonus $BONUS"
RUN=$RUN DEADLINE=$DL TARGET=100000 EXTRA="--policy symclock --cross-bonus $BONUS" ./train_queue.sh > train_queue.log 2>&1 &
QPID=$!
sleep 120
while kill -0 $QPID 2>/dev/null; do
  $PY make_site.py --run runs/$RUN --out docs/data/progress_v4.json > /dev/null 2>&1 && \
    publish "Lesson 8: v4-walk training progress $(date '+%H:%M')" docs/data/progress_v4.json
  for i in $(seq 60); do kill -0 $QPID 2>/dev/null || break; sleep 60; done
done
log "$(tail -1 train_queue.log)"

log "evaluation: $RUN on 100 unseen tracks; both champions on the same 100 comparison tracks"
$PY evaluate.py --run runs/$RUN --episodes 100 --out runs/$RUN/eval/best.json | grep -E '"(mean|std|falls|finished|n_ge_300)"'
rm -f results/${BASE}_compare.json results/${RUN}_compare.json   # re-score the FINAL champions
$PY experiments/gait_summary.py $BASE $RUN
$PY make_site.py --run runs/$RUN --out docs/data/progress_v4.json > /dev/null

log "lesson 8 narration, voice, clip"
$PY video_tools/narration8.py > /dev/null
s=lesson-08; f=""; [ -d voice/$s ] && f="--force"
$V new video_tools/narration/$s.md --kind voiceover --slug $s $f > /dev/null 2>&1
cp voice/lesson-01/glossary.md voice/$s/glossary.md
$V scan voice/$s > voice/$s/scan.txt 2>&1
$V prep voice/$s > /dev/null 2>&1
$V narrate voice/$s > voice/$s/narrate.log 2>&1 && $V listen voice/$s > voice/$s/listen.txt 2>&1
$V export voice/$s > voice/$s/export.log 2>&1
log "  voice: $($V status voice/$s 2>&1 | grep -c PASS)/5 stages PASS; listen flags: $(grep -cE 'swapped|missing|extra|garbled' voice/$s/listen.txt)"
$PY video_tools/make_lessons.py 08-two-ways-to-walk 2>&1 | grep -E "Rendering|->"

$PY - <<PYEOF >> TRAINING_LOG.md
import json
g = json.load(open("docs/data/gait.json")); W = {w["run"]: w for w in g["walkers"]}; c = g.get("compare", {})
a, b = W["${RUN}"], W["${BASE}"]
print(f"""
## ${RUN} (auto-summary from next_run.sh)
- symclock policy (one shared leg controller + learned clock, 931 params), cross bonus ${BONUS}, ${HOURS} h; champion gen {a['generation']}.
- Same 100 comparison tracks: ${RUN} {a['mean']} ± {a['std']} (falls {a['falls']}) vs ${BASE} {b['mean']} ± {b['std']} (falls {b['falls']});
  paired diff {c.get('diff_mean')} (95% CI {c.get('diff_ci95')}), ${RUN} better on {c.get('a_better_on')}/100.
- Gait: lead changes per episode {a['lead_changes']} vs {b['lead_changes']}; crossing steps {a['crossings']} vs {b['crossings']};
  foot 2 in front {a['front2_share']} vs {b['front2_share']}.""")
PYEOF
sed -i 's/, wip: true }/ }/' docs/assets/site.js
$PY -m pytest -q 2>&1 | tail -1
publish "Lesson 8: results of the walking experiment (v4-walk vs v3), narrated clip" \
  docs/data/gait.json docs/data/progress_v4.json docs/data/episodes docs/video docs/assets/site.js TRAINING_LOG.md
log "NEXT RUN DONE -> https://jca374.github.io/bipedal-cmaes/lessons/08-two-ways-to-walk/index.html"
