#!/usr/bin/env bash
# Narrate, listen-check and export the lesson voiceovers (voice-text pipeline), low priority.
set -uo pipefail
cd "$(dirname "$0")/.."
V=~/.claude/skills/voice-text/scripts/voice.sh
for n in "$@"; do
  d=voice/lesson-$n
  echo "[$(date +%H:%M:%S)] $d narrate"; nice -n 15 $V narrate $d > $d/narrate.log 2>&1 || { echo "narrate failed"; tail -5 $d/narrate.log; continue; }
  echo "[$(date +%H:%M:%S)] $d listen";  nice -n 15 $V listen $d > $d/listen.txt 2>&1 || true
  echo "[$(date +%H:%M:%S)] $d export";  $V export $d > $d/export.log 2>&1 || { echo "export failed"; tail -5 $d/export.log; }
  $V status $d 2>&1 | tail -12
done
echo "[$(date +%H:%M:%S)] VOICE DONE"
