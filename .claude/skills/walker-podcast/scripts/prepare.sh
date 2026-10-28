#!/usr/bin/env bash
# Build voice/podcast-NN from podcast/episode-NN.md and run the speech-flow scan.
# Exit 0 = clean, 2 = phrasing candidates to fix in podcast/episode-NN.md (then run again).
set -uo pipefail
cd "$(dirname "$0")/../../../.."          # project root
NN=${1:?usage: prepare.sh NN}
V=~/.claude/skills/voice-text/scripts/voice.sh
SRC=podcast/episode-$NN.md
[ -f "$SRC" ] || { echo "missing $SRC"; exit 1; }
[ -f podcast/episode-$NN.notes.json ] || echo "warning: podcast/episode-$NN.notes.json is missing (show notes)"
F=""; [ -d voice/podcast-$NN ] && F="--force"
$V new "$SRC" --kind prose --slug podcast-$NN $F | tail -1
cp voice/lesson-01/glossary.md voice/podcast-$NN/glossary.md
out=$($V scan voice/podcast-$NN 2>&1)
if echo "$out" | grep -q "([1-9][0-9]* candidates"; then
  echo "$out" | grep -v "(0 candidates"
  echo "=> fix these in $SRC and run prepare.sh $NN again"
  exit 2
fi
echo "scan clean: ready for publish.sh $NN"
