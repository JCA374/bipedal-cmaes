#!/usr/bin/env bash
# Start attempt 6 when attempt 4's driver (curriculum_f.sh) has finished, and point the dashboards at the live attempts.
cd "$(dirname "$0")"
while ps -eo args | grep -q "^bash ./curriculum_f.sh"; do sleep 60; done
echo "[$(date +%H:%M:%S)] attempt 4 finished; starting attempt 6"
setsid nohup ./curriculum_h.sh > curriculum_h.log 2>&1 &
sleep 5
for p in $(ps -eo pid,args | awk '$3=="dashboard.py"{print $1}'); do kill $p; done
sleep 1
setsid nohup .venv/bin/python dashboard.py --run auto:curriculum_h.log --port=8771 > dashboard_auto.log 2>&1 &
setsid nohup .venv/bin/python dashboard.py --run auto:curriculum_g.log --port=8770 > dashboard_c.log 2>&1 &
echo "[$(date +%H:%M:%S)] dashboards: 8771 = attempt 6, 8770 = attempt 5"
