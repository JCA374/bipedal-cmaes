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
CROSS_HYST = 0.15     # a lower leg must lead by this much (world units) to count as "in front"
CROSS_MIN_DX = 0.4    # a front-foot change counts as a crossing step only after this much progress

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


class SymClock:
    """One controller shared by both legs, with a clock: the 'symclock' policy.

    Each leg's controller sees [hull (4), own leg (5), other leg (5), lidar (10),
    sin φ, cos φ] = 26 inputs and outputs (hip, knee) for its own leg. Leg 2 uses the
    same weights with own/other swapped and its phase shifted by π, so the legs make the
    same movement in opposite phase and neither can specialise as a permanent prop.
    θ = [W1 (H x 26), b1, (more hidden layers), W_out (2 x H), b_out, p] where the clock
    period is T = 40 · exp(clip(p, -1, 1)) steps (≈15-109), learned like any weight.
    Stateful: reset() before each episode (run_episode does).
    """
    HULL, LEG1, LEG2, LIDAR = slice(0, 4), slice(4, 9), slice(9, 14), slice(14, 24)

    def __init__(self, sizes, theta=None):
        self.sizes = tuple(int(s) for s in sizes)
        assert self.sizes[0] == 26 and self.sizes[-1] == 2
        self.n_params = n_params(self.sizes) + 1
        self.set_params(np.zeros(self.n_params) if theta is None else theta)
        self.t = 0

    @property
    def n_hidden_layers(self):
        return len(self.sizes) - 2

    @property
    def period(self):
        return 40.0 * float(np.exp(np.clip(self.p, -1, 1)))

    def set_params(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        self.layers = unflatten(theta[:-1], self.sizes)
        self.p = theta[-1]

    def reset(self):
        self.t = 0

    def inputs(self, x):
        ph = 2 * np.pi * self.t / self.period
        x1 = np.concatenate([x[self.HULL], x[self.LEG1], x[self.LEG2], x[self.LIDAR], [np.sin(ph), np.cos(ph)]])
        x2 = np.concatenate([x[self.HULL], x[self.LEG2], x[self.LEG1], x[self.LIDAR], [np.sin(ph + np.pi), np.cos(ph + np.pi)]])
        return np.stack([x1, x2])

    def activate(self, x, return_hidden=False):
        h = self.inputs(x)
        self.t += 1
        hidden = []
        for W, b in self.layers:
            h = np.tanh(h @ W.T + b)
            hidden.append(h)
        a = h.reshape(-1)            # [leg1 hip, leg1 knee, leg2 hip, leg2 knee]
        return (a, [hh.reshape(-1) for hh in hidden[:-1]]) if return_hidden else a


POLICIES = {"mlp": MLP, "symclock": SymClock}


def widen(kind, theta, sizes, hidden, rng=None, scale=0.05):
    """Net2Net-style widening of a one-hidden-layer policy to `hidden` units WITHOUT changing what it does:
    new hidden units get small random input weights but ZERO outgoing weights, so the widened network
    computes exactly the same actions; evolution can then put the extra capacity to use.
    Returns (new_theta, new_sizes)."""
    assert len(sizes) == 3 and hidden >= sizes[1], "widen handles one hidden layer, growing only"
    rng = rng or np.random.default_rng(0)
    theta = np.asarray(theta, dtype=np.float64)
    extra = theta[-1:] if kind == "symclock" else theta[:0]        # the clock period parameter
    core = theta[:-1] if kind == "symclock" else theta
    (W1, b1), (W2, b2) = unflatten(core, tuple(sizes))
    n_in, h, n_out = sizes
    add = hidden - h
    W1n = np.vstack([W1, scale * rng.normal(size=(add, n_in))])
    b1n = np.concatenate([b1, np.zeros(add)])
    W2n = np.hstack([W2, np.zeros((n_out, add))])
    new_sizes = (n_in, hidden, n_out)
    return np.concatenate([flatten([(W1n, b1n), (W2n, b2)]), extra]), new_sizes


def policy_sizes(kind, hidden):
    return (N_OBS, *hidden, N_ACT) if kind == "mlp" else (26, *hidden, 2)


def make_policy(kind, sizes, theta=None):
    return POLICIES[kind](sizes, theta)


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


SUPPORT_K_ANGLE, SUPPORT_K_VEL = 200.0, 30.0   # "training wheels": upright torque = support × (k_a·angle + k_v·ang. velocity)


def run_episode(policy, env, seed, early_stop=True, max_steps=1600, on_frame=None,
                action_repeat=1, on_step=None, support=0.0):
    """Roll out one episode. Returns dict(reward, steps, saved, fell, x).

    saved = steps cut off by early termination. x = hull x at the end (the course
    finish is at ~88). on_frame(env, total) is called before every step (rendering);
    on_step(obs_scaled, action, reward, env) after choosing each action (analysis).
    """
    obs, _ = fresh_reset(env, seed)
    if hasattr(policy, "reset"):
        policy.reset()
    u = env.unwrapped
    hull = u.hull
    xs, total, steps, fell, saved = [], 0.0, 0, False, 0
    action = None
    # Gait: which lower leg is in front (with hysteresis), how often that changes, and how
    # many changes are real steps (hull moved >= CROSS_MIN_DX since the last counted one).
    front, lead_changes, crossings, front2, last_cross_x = 0, 0, 0, 0, hull.position.x
    leading = 0      # steps where one foot leads by more than CROSS_HYST
    angle_sum, effort, alt_td, last_td, contact = 0.0, 0.0, 0, 0, (False, False)
    step_td, last_step = 0, 0   # touchdowns by the FRONT foot, alternating legs: what a walking step looks like
    lead_sum = 0.0   # sum of (lower leg 1 x - lower leg 2 x): its mean is how far foot 1 sits ahead on average
    while steps < max_steps:
        if on_frame is not None:
            on_frame(env, total)
        if steps % action_repeat == 0:
            o = scale_obs(obs)
            action = policy.activate(o)
            if on_step is not None:
                on_step(o, action, total, env)
        if support:   # assistive torque that keeps the hull upright (curriculum only; reported results use 0)
            hull.ApplyTorque(-support * (SUPPORT_K_ANGLE * hull.angle + SUPPORT_K_VEL * hull.angularVelocity), True)
        obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        steps += 1
        total += float(reward)
        d = u.legs[1].position.x - u.legs[3].position.x
        lead_sum += d
        f = 1 if d > CROSS_HYST else (-1 if d < -CROSS_HYST else front)
        if f != front:
            if front != 0:
                lead_changes += 1
                if hull.position.x - last_cross_x >= CROSS_MIN_DX:
                    crossings += 1
                    last_cross_x = hull.position.x
            front = f
        front2 += front == -1
        leading += front != 0
        angle_sum += abs(hull.angle)
        effort += 0.00035 * 80 * float(np.sum(np.minimum(np.abs(action), 1)))
        c = (obs[8] > 0.5, obs[13] > 0.5)          # touchdowns, and whether they alternate legs
        for leg in (0, 1):
            if c[leg] and not contact[leg]:
                if last_td and last_td != leg + 1:
                    alt_td += 1
                last_td = leg + 1
                if front == (1 if leg == 0 else -1) and last_step != leg + 1:
                    step_td += 1
                    last_step = leg + 1
        contact = c
        if terminated:
            fell = bool(reward <= -100)
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
            "stalled": saved > 0, "x": float(hull.position.x),
            "crossings": crossings, "lead_changes": lead_changes, "front2_share": front2 / max(1, steps),
            "lead_mean": lead_sum / max(1, steps),
            # share of the LEADING steps in which foot 2 leads (0.5 when no foot ever leads: feet together isn't lopsided)
            "front2_lead_share": front2 / leading if leading else 0.5,
            "angle_abs_mean": angle_sum / max(1, steps), "effort": effort, "alt_touchdowns": alt_td,
            "step_touchdowns": step_td}


# ---------------------------------------------------------------- persistence
def save_policy(dirpath, theta, sizes, meta=None, kind="mlp"):
    d = Path(dirpath)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "params.tmp.npy"
    np.save(tmp, np.asarray(theta, dtype=np.float64))
    tmp.replace(d / "params.npy")  # atomic: readers never see a half-written file
    with open(d / "meta.json", "w") as f:
        json.dump({**(meta or {}), "sizes": list(sizes), "policy": kind}, f, indent=2)


def load_policy(run_or_dir):
    """Accept runs/<name>, runs/<name>/best or a milestone dir. Returns (policy, meta)."""
    p = Path(run_or_dir)
    if (p / "best" / "params.npy").exists():
        p = p / "best"
    meta = json.loads((p / "meta.json").read_text())
    return make_policy(meta.get("policy", "mlp"), meta["sizes"], np.load(p / "params.npy")), meta
