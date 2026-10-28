#!/usr/bin/env bash
# Every 30 min until Friday 17:00: rebuild the Curriculum tab's data and push it.
cd "$(dirname "$0")"
END=$(date -d "2026-10-30 17:15" +%s)
while [ "$(date +%s)" -lt "$END" ]; do
  nice -n 5 .venv/bin/python make_curriculum_page.py > /dev/null 2>&1
  git add docs/data/curriculum.json docs/data/curriculum_journal.json docs/data/curriculum_attempts.json docs/data/goal_*.json docs/data/episodes/cur_*.json docs/data/curriculum_log.json docs/data/episodes/log 2>/dev/null
  git commit -q -m "Curriculum tab update $(date '+%a %H:%M')" 2>/dev/null && git push -q origin main 2>/dev/null && echo "[$(date '+%a %H:%M')] pushed"
  sleep 900
done
