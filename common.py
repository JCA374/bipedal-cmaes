"""Shared helpers: env creation, static obs scaling, the MLP policy over a flat
parameter vector, episode rollout, seed ranges, saving/loading policies.

A trained policy is params.npy (the flat vector θ) + meta.json (the layer sizes).
There is no running normalisation: OBS_SCALE is fixed.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gymnasium as gym
import numpy as np

ENV_ID = "BipedalWalker-v3"
ROOT = Path(__file__).resolve().parent
N_OBS, N_ACT = 24, 4

# Static per-dimension divisors (ported from ../bipedal-neat, 99th percentile of |obs|).
# 0 hull angle, 1 hull ang. vel, 2 vx, 3 vy,
# 4-7 hip1 angle/speed, knee1 angle/speed, 8 contact1,
# 9-12 hip2 angle/speed, knee2 angle/speed, 13 contact2, 14-23 lidar.
OBS_SCALE = np.array(
    [1.0, 0.25, 0.5, 0.5,
     1.2, 2.0, 1.2, 2.0, 1.0,
     1.2, 2.0, 1.2, 2.0, 1.0]
    + [1.0] * 10,
    dtype=np.float64,
)
INPUT_NAMES = ["hull angle", "hull ang. vel", "vel x", "vel y",
               "hip 1 angle", "hip 1 speed", "knee 1 angle", "knee 1 speed", "leg 1 contact",
               "hip 2 angle", "hip 2 speed", "knee 2 angle", "knee 2 speed", "leg 2 contact"] + \
              [f"lidar {i}" for i in range(10)]
OUTPUT_NAMES = ["hip 1", "knee 1", "hip 2", "knee 2"]

# Early termination (training only): stop if the hull moves less than this over the window.
STALL_WINDOW = 150
STALL_MIN_DX = 1.0

# Seed ranges, never overlapping (tests/test_common.py checks this).
SEED_RANGES = {
    "train": (0, 1_000_000),
    "hof": (10_000_000, 11_000_000),
    "evaluate": (20_000_000, 21_000_000),
    "record": (30_000_000, 31_000_000),
    "compare": (40_000_000, 41_000_000),
}


def seeds(kind, n, offset=0):
    lo, hi = SEED_RANGES[kind]
    assert lo + offset + n <= hi
    return [lo + offset + i for i in range(n)]


def scale_obs(obs):
    return np.clip(np.asarray(obs, dtype=np.float64) / OBS_SCALE, -1.0, 1.0)


def make_env(render_mode=None):
    return gym.make(ENV_ID, render_mode=render_mode)


# ---------------------------------------------------------------- policy
class MLP:
    """tanh MLP, sizes e.g. (24, 32, 4). Parameters live in one flat vector θ, laid out
    layer by layer as [W (out x in, row-major), b (out)]."""

    def __init__(self, sizes, theta=None):
        self.sizes = tuple(int(s) for s in sizes)
        assert self.sizes[0] == N_OBS and self.sizes[-1] == N_ACT
        self.n_params = n_params(self.sizes)
        self.set_params(np.zeros(self.n_params) if theta is None else theta)

    @property
    def n_hidden_layers(self):
        return len(self.sizes) - 2

    def set_params(self, theta):
        self.layers = unflatten(np.asarray(theta, dtype=np.float64), self.sizes)

    def activate(self, x, return_hidden=False):
        hidden = []
        for W, b in self.layers:
            x = np.tanh(W @ x + b)
            hidden.append(x)
        return (x, hidden[:-1]) if return_hidden else x


def n_params(sizes):
    return sum(o * i + o for i, o in zip(sizes[:-1], sizes[1:]))


def unflatten(theta, sizes):
    assert theta.shape == (n_params(sizes),), (theta.shape, n_params(sizes))
    layers, k = [], 0
    for i, o in zip(sizes[:-1], sizes[1:]):
        W = theta[k:k + o * i].reshape(o, i)
        k += o * i
        b = theta[k:k + o]
        k += o
        layers.append((W, b))
    return layers


def flatten(layers):
    return np.concatenate([np.concatenate([W.ravel(), b]) for W, b in layers])


# ---------------------------------------------------------------- rollout
def fresh_reset(env, seed):
    """env.reset with a brand-new Box2D world. A reused world carries contact/broadphase
    state between episodes, so the same seed + policy can give a different reward
    (seen: -118.4 vs -119.8 on the 3rd reset). Fresh worlds make episodes exactly
    reproducible, which common random numbers and paired comparisons rely on."""
    import Box2D
    u = env.unwrapped
    u._destroy()
    u.world = Box2D.b2World()
    return env.reset(seed=int(seed))


def run_episode(policy, env, seed, early_stop=True, max_steps=1600, on_frame=None,
                action_repeat=1, on_step=None):
    """Roll out one episode. Returns dict(reward, steps, saved, fell, x).

    saved = steps cut off by early termination. x = hull x at the end (the course
    finish is at ~88). on_frame(env, total) is called before every step (rendering);
    on_step(obs_scaled, action, reward, env) after choosing each action (analysis).
    """
    obs, _ = fresh_reset(env, seed)
    hull = env.unwrapped.hull
    xs, total, steps, fell, saved = [], 0.0, 0, False, 0
    action = None
    while steps < max_steps:
        if on_frame is not None:
            on_frame(env, total)
        if steps % action_repeat == 0:
            o = scale_obs(obs)
            action = policy.activate(o)
            if on_step is not None:
                on_step(o, action, total, env)
        obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        steps += 1
        total += float(reward)
        if terminated:
            fell = reward <= -100
            break
        if truncated:
            break
        if early_stop:
            xs.append(hull.position.x)
            if len(xs) > STALL_WINDOW and xs[-1] - xs[-1 - STALL_WINDOW] < STALL_MIN_DX:
                saved = max_steps - steps
                break
    if on_frame is not None:
        on_frame(env, total)
    return {"reward": total, "steps": steps, "saved": saved, "fell": fell,
            "stalled": saved > 0, "x": float(hull.position.x)}


# ---------------------------------------------------------------- persistence
def save_policy(dirpath, theta, sizes, meta=None):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "params.tmp.npy"
    np.save(tmp, np.asarray(theta, dtype=np.float64))
    tmp.replace(d / "params.npy")  # atomic: readers never see a half-written file
    with open(d / "meta.json", "w") as f:
        json.dump({**(meta or {}), "sizes": list(sizes)}, f, indent=2)


def load_policy(run_or_dir):
    """Accept runs/<name>, runs/<name>/best or a milestone dir. Returns (MLP, meta)."""
    p = Path(run_or_dir)
    if (p / "best" / "params.npy").exists():
        p = p / "best"
    meta = json.loads((p / "meta.json").read_text())
    return MLP(meta["sizes"], np.load(p / "params.npy")), meta
