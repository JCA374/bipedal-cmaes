#!/usr/bin/env bash
# After training: final evaluation, comparison, analysis data and site data. Safe to re-run.
#   RUN=v3-f20 ./finish.sh
set -uo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
RUN=${RUN:-v3-f20}
R=runs/$RUN
W=${WORKERS:-4}
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "final score: champion on 100 unseen tracks (seeds 2e7+)"
$PY evaluate.py --run $R --episodes 100 --workers $W --out $R/eval/best.json | grep -E '"(mean|std|falls|n_ge_300|timeouts|timeout_mean_x|finished)"'
for m in $R/milestones/*/; do
  n=$(basename $m)
  $PY evaluate.py --run $m --episodes 100 --workers $W --out $R/eval/milestone_$n.json > /dev/null && \
    log "  milestone $n: $($PY -c "import json;d=json.load(open('$R/eval/milestone_$n.json'));print(d['mean'],'±',d['std'],'falls',d['falls'])")"
done

log "paired comparison with NEAT on 100 comparison tracks (seeds 4e7+)"
[ -f results/neat_compare.json ] || ../bipedal-neat/.venv/bin/python experiments/eval_neat.py --neat ../bipedal-neat/runs/baseline --out results/neat_compare.json
$PY evaluate.py --run $R --episodes 100 --workers $W --seed-kind compare --out results/cma_compare.json > /dev/null
$PY experiments/compare.py results/cma_compare.json results/neat_compare.json --out docs/data/compare_neat.json | grep -E "mean_|diff|falls|better"

log "analysis data for lessons 5-7"
$PY experiments/inspect_policy.py --run $R --episodes 20 --workers $W | tail -5
$PY experiments/archetypes.py --walker $R > /dev/null
$PY experiments/export_episode.py --run $R --label "CMA-ES champion" --out docs/data/episodes/champion.json | tail -1
for m in $R/milestones/*/; do
  n=$(basename $m)
  $PY experiments/export_episode.py --run $m --label "milestone $n" --out docs/data/episodes/stage_$n.json | tail -1
done
cp $R/eval/best.json docs/data/eval_best.json
$PY -c "
import json,glob,os
out={os.path.basename(p)[10:-5]:{k:v for k,v in json.load(open(p)).items() if k!='episodes_detail'} for p in glob.glob('$R/eval/milestone_*.json')}
json.dump(out,open('docs/data/eval_milestones.json','w'),indent=1)"
$PY make_site.py --run $R
log "FINISH DONE"
