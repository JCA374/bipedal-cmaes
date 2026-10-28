"""The curriculum's definition of done: a FAST walker that puts one foot in front of the other.
On 20 held-out tracks (record range, offset 700), full 1600-step episodes, raw reward.
  python experiments/goal_check.py runs/c4-fast        # prints JSON; exit 0 = goal met
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import common  # noqa: E402

GOAL = {"lopsided_max": 0.3, "lead_changes_min": 20, "x_min": 80, "falls_max": 2, "score_min": 250}


def measure(run, n=20):
    pol, meta = common.load_policy(run)
    env = common.make_env()
    eps = [common.run_episode(pol, env, s, early_stop=False) for s in common.seeds("record", n, offset=700)]
    m = lambda k: float(np.mean([e[k] for e in eps]))
    r = {"run": str(run), "champion_gen": meta.get("generation"), "score": m("reward"), "x": m("x"),
         "falls": int(sum(e["fell"] for e in eps)), "lead_changes": m("lead_changes"),
         "lopsided": float(np.mean([abs(2 * e["front2_lead_share"] - 1) for e in eps])),
         "effort": m("effort"), "steps": m("steps")}
    r["checks"] = {"alternates": r["lopsided"] <= GOAL["lopsided_max"] and r["lead_changes"] >= GOAL["lead_changes_min"],
                   "far": r["x"] >= GOAL["x_min"], "safe": r["falls"] <= GOAL["falls_max"], "fast": r["score"] >= GOAL["score_min"]}
    r["met"] = all(r["checks"].values())
    return r


if __name__ == "__main__":
    r = measure(sys.argv[1])
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}))
    sys.exit(0 if r["met"] else 1)
