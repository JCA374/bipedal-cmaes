"""Dump one episode as compact JSON for the in-browser replay (lessons + site).

  python experiments/export_episode.py --run runs/baseline --out docs/data/episodes/champion.json
  python experiments/export_episode.py --random 0.5 --out docs/data/episodes/random.json
  ../bipedal-neat/.venv/bin/python experiments/export_episode.py --neat ../bipedal-neat/runs/baseline \
      --out docs/data/episodes/neat.json        # (NEAT needs its own venv)

Per kept step: hull + 4 leg polygons (world coords), the 24 scaled inputs, the 4
actions, the running reward. Picks the median-reward terrain among --pick seeds so
the example is typical, not lucky.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import common  # noqa: E402


def load_neat(run_dir):
    neat_root = Path(run_dir).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("neat_common", neat_root / "common.py")
    nc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nc)
    genome, config, _ = nc.resolve_genome(run_dir)
    return nc.make_net(genome, config), 1


def r2(a):
    return [round(float(v), 2) for v in a]


def record(policy, seed, ar, every, support=0.0):
    env = common.make_env()
    frames = []

    def on_step(o, a, total, env_):
        pass

    obs_log = {}
    actions = []

    class Spy:
        """Wrap the policy to keep the latest inputs/outputs for each frame."""
        def reset(self):   # stateful policies (the symclock's clock) must start each episode fresh
            if hasattr(policy, "reset"):
                policy.reset()

        def activate(self, x):
            y = policy.activate(x)
            actions.append(np.asarray(y, dtype=np.float64))
            obs_log["x"], obs_log["y"] = x, y
            return y

    step = {"i": 0}

    def on_frame(env_, total):
        i = step["i"]
        step["i"] += 1
        if i == 0 or "x" not in obs_log or i % every:
            return
        u = env_.unwrapped
        polys = []
        for body in [u.hull] + u.legs:
            for f in body.fixtures:
                polys.append([round(c, 2) for v in f.shape.vertices for c in body.transform * v])
        frames.append({"p": polys, "o": r2(obs_log["x"]), "a": r2(obs_log["y"]), "r": round(total, 1),
                       "x": round(u.hull.position.x, 2)})

    res = common.run_episode(Spy(), env, seed, early_stop=False, on_frame=on_frame, action_repeat=ar, support=support)
    u = env.unwrapped
    terrain = [r2(u.terrain_x), r2(u.terrain_y)]
    return res, terrain, frames, np.array(actions)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--neat")
    g.add_argument("--random", type=float, help="random MLP 24-32-4 with this weight std")
    ap.add_argument("--rng", type=int, default=1, help="weight seed for --random")
    ap.add_argument("--label", default=None)
    ap.add_argument("--pick", type=int, default=9, help="choose the median of this many terrains")
    ap.add_argument("--seed", type=int, default=None, help="use exactly this terrain")
    ap.add_argument("--every", type=int, default=2, help="keep every Nth step")
    ap.add_argument("--out", required=True)
    ap.add_argument("--support", type=float, default=0.0, help="training-wheels torque during the replay (curriculum stages)")
    ap.add_argument("--actions-out", default=None, help="also save every step's action (.npy) for exact replays")
    args = ap.parse_args()

    ar = 1
    if args.run:
        policy, meta = common.load_policy(args.run)
        ar = meta.get("action_repeat", 1)
    elif args.neat:
        policy, ar = load_neat(args.neat)
    else:
        policy = common.MLP((24, 32, 4), np.random.default_rng(args.rng).normal(0, args.random, 932))

    if args.seed is not None:
        seed = args.seed
    else:
        cands = common.seeds("record", args.pick)
        env = common.make_env()
        rs = [common.run_episode(policy, env, s, early_stop=False, action_repeat=ar)["reward"] for s in cands]
        seed = cands[int(np.argsort(rs)[len(rs) // 2])]
        print("terrain rewards:", [round(r) for r in rs], "-> median seed", seed)
    res, terrain, frames, actions = record(policy, seed, ar, args.every, args.support)
    if args.actions_out:
        np.save(args.actions_out, actions)
    out = {"label": args.label, "seed": seed, "support": args.support, "reward": round(res["reward"], 1), "steps": res["steps"],
           "fell": res["fell"], "every": args.every, "terrain": terrain, "frames": frames,
           "input_names": common.INPUT_NAMES, "output_names": common.OUTPUT_NAMES}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"{args.out}: reward {res['reward']:.1f}, {res['steps']} steps, fell={res['fell']}, "
          f"{len(frames)} frames, {Path(args.out).stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
