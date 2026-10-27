#!/usr/bin/env bash
# Hourly: refresh docs/data/progress.json from the running training and push it, so the
# site's live curve stays current. Stops when training is over (finalize.sh takes over).
cd "$(dirname "$0")"
RUN=${RUN:-v3-f20}
while pgrep -f "[t]rain_queue.sh" >/dev/null; do
  sleep 3600
  .venv/bin/python make_site.py --run runs/$RUN > /dev/null 2>&1 || continue
  git add docs/data/progress.json && git commit -q -m "Training progress: $(date '+%Y-%m-%d %H:%M')" && git push -q origin main
done
