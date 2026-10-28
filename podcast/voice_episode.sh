#!/usr/bin/env bash
# Voice one podcast episode with the voice-text pipeline and publish its MP3 to docs/podcast/.
#   podcast/voice_episode.sh 01
set -uo pipefail
cd "$(dirname "$0")/.."
V=~/.claude/skills/voice-text/scripts/voice.sh
n=$1; d=voice/podcast-$n
nice -n 5 $V narrate $d > $d/narrate.log 2>&1 || { echo "narrate failed"; tail -5 $d/narrate.log; exit 1; }
$V listen $d > $d/listen.txt 2>&1 || true
$V export $d > $d/export.log 2>&1 || { echo "export failed"; exit 1; }
mkdir -p docs/podcast
cp $d/audio/ch01.mp3 docs/podcast/episode-$n.mp3
echo "status: $($V status $d 2>&1 | grep -c PASS)/5 PASS; listen flags:"; grep -E "swapped|missing|extra|garbled" $d/listen.txt
echo "duration: $(ffprobe -v error -show_entries format=duration -of csv=p=0 docs/podcast/episode-$n.mp3) s"
echo "EPISODE DONE"
