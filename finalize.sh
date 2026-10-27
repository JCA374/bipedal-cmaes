#!/usr/bin/env bash
# Everything after training, end to end: evaluation and analysis data (finish.sh), the
# failure/score analysis reports, narration for lessons 5-7 + the explainer from the final
# numbers, voice (voice-text pipeline), video renders, site, tests, commit and push.
#
#   RUN=v3-f20 nohup ./finalize.sh > finalize.log 2>&1 &      (waits for train_queue.sh to finish)
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
V=~/.claude/skills/voice-text/scripts/voice.sh
RUN=${RUN:-v3-f20}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "waiting for training to finish"
while pgrep -f "[t]rain_queue.sh" >/dev/null || pgrep -f "[t]rain.py --name $RUN" >/dev/null; do sleep 60; done
log "training done: $(tail -1 train_queue.log)"

RUN=$RUN ./finish.sh || { log "finish.sh failed"; exit 1; }

log "failure + score analysis (failure-analysis-report method)"
$PY experiments/record_episodes.py --run runs/$RUN --episodes 300 --name final > /dev/null 2>&1
$PY experiments/analyze_falls.py results/episodes/final.jsonl.gz --out docs/analysis/falls 2>&1 | tail -1
$PY experiments/analyze_score.py results/episodes/final.jsonl.gz --out docs/analysis/score 2>&1 | tail -2

log "narration from the final numbers"
$PY video_tools/narration.py runs/$RUN
for s in lesson-05 lesson-06 lesson-07 explainer; do
  f=""; [ -d voice/$s ] && f="--force"
  $V new video_tools/narration/$s.md --kind voiceover --slug $s $f > /dev/null 2>&1
  cp voice/lesson-01/glossary.md voice/$s/glossary.md
  $V scan voice/$s > voice/$s/scan.txt 2>&1
  $V prep voice/$s > /dev/null 2>&1
  $V narrate voice/$s > voice/$s/narrate.log 2>&1 || { log "narrate $s failed"; continue; }
  $V listen voice/$s > voice/$s/listen.txt 2>&1 || true
  $V export voice/$s > voice/$s/export.log 2>&1 || log "export $s failed"
  log "  $s: $($V status voice/$s 2>&1 | grep -c PASS)/5 stages PASS; listen flags: $(grep -cE 'swapped|missing|extra|garbled' voice/$s/listen.txt)"
done

log "videos"
$PY video_tools/make_lessons.py 05-the-training-run 06-inside-the-champion 07-results explainer 2>&1 | grep -E "Rendering|->"

log "site: lessons final, analysis links"
sed -i 's/, draft: true }/ }/' docs/assets/site.js
$PY make_site.py --run runs/$RUN

log "tests + publish"
$PY -m pytest -q 2>&1 | tail -1
git add -A
git commit -q -m "Final: evaluation, analyses, lessons 5-7 and the narrated explainer

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin main && log "pushed"
for i in $(seq 1 30); do s=$(gh api repos/JCA374/bipedal-cmaes/pages/builds/latest --jq .status 2>/dev/null); [ "$s" = built ] && break; sleep 10; done
log "pages: $s -> https://jca374.github.io/bipedal-cmaes/"
log "FINALIZE DONE"
