"""Curriculum graduation test: the run's champion on 20 held-out tracks, full stage-length episodes.
Exit code 0 = passed.   python experiments/stage_check.py balance runs/c1-balance
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import common  # noqa: E402

STEPS = {"balance": 500, "march": 600, "walk": 1600, "balance_c": 500, "step": 600}


def main():
    stage, run = sys.argv[1], sys.argv[2]
    pol, meta = common.load_policy(run)
    env = common.make_env()
    eps = [common.run_episode(pol, env, s, early_stop=False, max_steps=STEPS[stage]) for s in common.seeds("record", 20, offset=500)]
    m = lambda k: float(np.mean([e[k] for e in eps]))
    falls = int(sum(e["fell"] for e in eps))
    lop = float(np.mean([abs(2 * e["front2_lead_share"] - 1) for e in eps]))
    res = {"stage": stage, "run": run, "champion_gen": meta.get("generation"), "falls": falls, "reward": m("reward"),
           "x": m("x"), "lead_changes": m("lead_changes"), "alt_touchdowns": m("alt_touchdowns"),
           "lopsided": lop, "step_touchdowns": m("step_touchdowns"), "abs_offset": float(np.mean([abs(e["lead_mean"]) for e in eps])), "angle": m("angle_abs_mean")}
    ok = {"balance": falls == 0 and res["angle"] < 0.15,
          "march": falls <= 1 and res["lead_changes"] >= 12 and res["abs_offset"] < 0.5,
          "walk": falls <= 4 and res["x"] >= 60 and lop <= 0.3,
          "balance_c": falls == 0 and res["angle"] < 0.15 and res["abs_offset"] < 0.4,
          "step": falls <= 1 and res["lead_changes"] >= 12 and res["step_touchdowns"] >= 10 and res["abs_offset"] < 0.5 and lop <= 0.4}[stage]
    res["passed"] = bool(ok)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
