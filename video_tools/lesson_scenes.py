"""Scenes for the lesson clips. Scene names match the `## <scene>` sections of
video_tools/narration/lesson-NN.md; a section `## <scene>.<n>` is beat n of that scene.

Every number shown is computed here or loaded from video_tools/data/ (no hand-typed results).
"""
import json
import os

import cma
import numpy as np
from PIL import Image, ImageDraw

import vcommon as C
from vcommon import (AMBER, BG, BLUE, FG, GREEN, H, MUTED, ORANGE, PANEL, RED, TEAL, W, Scene, axes, bars,
                     blank, bullet, cover_frame, fade, label, paragraph, paste_frame, series, text, title_card)

NEAT_SEED = 30_000_002   # median of 9 record-range terrains for the NEAT champion (export_episode.py)
RANDOM_SEED = 30_000_000


def _cache(fn):
    store = {}

    def get(*a):
        if a not in store:
            store[a] = fn(*a)
        return store[a]
    return get


@_cache
def neat_player():
    return C.Player(C.load_actions("neat"), NEAT_SEED)


@_cache
def random_player():
    return C.Player(C.load_actions("random"), RANDOM_SEED)


@_cache
def neat_trace():
    """x position and running reward per step of the NEAT example episode."""
    import common
    env = common.make_env()
    pol = C.load_actions("neat")
    obs, _ = common.fresh_reset(env, NEAT_SEED)
    xs, rs, obs_l, tot = [], [], [], 0.0
    for a in pol.actions:
        obs_l.append(common.scale_obs(obs))
        obs, r, te, tr, _ = env.step(np.asarray(a, dtype=np.float32))
        tot += r
        xs.append(env.unwrapped.hull.position.x)
        rs.append(tot)
        if te or tr:
            break
    return np.array(xs), np.array(rs), np.array(obs_l)


def beat_alpha(beats, t, k, dur=0.4):
    return fade(t, beats.at(k), dur)


# =============================================================================
# Lesson 1: the task
# =============================================================================

def l1_walker():
    p = neat_player()

    def render(t, b):
        img = blank()
        cover_frame(img, p.at_time(t, 1.0, loop_pause=0))
        d = ImageDraw.Draw(img)
        k, lt, _ = b.local(t)
        label(d, (40, 40), "BipedalWalker-v3", 22)
        if k == 0:
            label(d, (40, 90), "hull (body)", 18)
            label(d, (40, 125), "2 legs × (hip + knee) = 4 motors", 18)
        elif k == 1:
            label(d, (40, 90), "random bumpy track, new every episode", 18)
            label(d, (40, 125), "finish ≈ 84 units to the right", 18)
        else:
            label(d, (40, 90), f"step {p.step} / 1600   (50 steps per second)", 20)
            label(d, (40, 125), "ends on: fall · finish · 1600 steps (32 s)", 18)
        return img
    return Scene("walker", render, [6, 6, 6])


def l1_senses():
    p = neat_player()
    groups = [(0, 4), (4, 14), (14, 24)]
    names = ["body: 4 numbers", "legs: 10 numbers", "lidar: 10 numbers"]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "What it senses: 24 numbers", t)
        paste_frame(img, p.at_time(t, 0.6, start=200, loop_pause=0, length=1300), (40, 130, 660, 440))
        k, _, _ = b.local(t)
        lo, hi = groups[k]
        cols = [C.INPUT_COL[i] if lo <= i < hi else C.mix(BG, C.INPUT_COL[i], .25) for i in range(24)]
        bars(d, (740, 125, 1240, 680), p.x_obs, C.INPUT_SHORT, cols, size=14, label_w=100)
        text(d, (40, 610), names[k], 30, C.INPUT_COL[lo], bold=True)
        text(d, (40, 655), "scaled to −1 … 1, the network never sees a picture", 18, MUTED)
        return img
    return Scene("senses", render, [6, 6, 6])


def heading(d, title, t, sub=None):
    C.heading(d, title, t, sub)


def l1_actions():
    p = neat_player()

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "What it does: 4 numbers", t)
        paste_frame(img, p.at_time(t, 0.5, start=400, loop_pause=0, length=1100), (40, 130, 660, 440))
        bars(d, (740, 200, 1240, 480), p.act, C.OUTPUT_SHORT, [BLUE, BLUE, ORANGE, ORANGE], size=22, label_w=120)
        text(d, (860, 510), "−1", 20, MUTED, anchor="mm")
        text(d, (1180, 510), "+1", 20, MUTED, anchor="mm")
        paragraph(d, (740, 560), "sign = which way to turn the joint, size = how hard to push (max torque 80)", 30, 20, MUTED)
        return img
    return Scene("actions", render, [8])


def l1_reward():
    p, rp = neat_player(), random_player()
    xs, rs, _ = neat_trace()
    items = [("progress", "+130 · Δx / 30  →  ≈ 360 for the whole track", GREEN),
             ("effort + tilt", "−0.028 per unit of motor command per step", AMBER),
             ("fall", "−100, and the episode ends", RED),
             ("solved", "≥ 300 averaged over 100 random tracks", BLUE)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "How it's scored", t)
        k, lt, _ = b.local(t)
        if k == 2:
            fr = rp.at_time(lt, 0.6, loop_pause=1.2)
            paste_frame(img, fr, (40, 130, 600, 400), border=RED)
            label(d, (60, 150), f"random network: reward {rp.total:.0f}" + ("  FELL" if rp.done else ""), 18)
        else:
            fr = p.at_time(t, 1.0, loop_pause=0)
            paste_frame(img, fr, (40, 130, 600, 400))
            box = (60, 570, 620, 690)
            axes(d, box, -50, 250, [0, 100, 200], a=1)
            series(d, box, p.rewards, BLUE, -50, 250, n=1600)
            label(d, (60, 545), f"running reward {p.total:.0f}", 16)
        for i, (name, desc, col) in enumerate(items):
            a = beat_alpha(b, t, i)
            y = 150 + i * 130
            text(d, (690, y), name, 30, col, a, bold=True)
            paragraph(d, (690, y + 42), desc, 40, 20, FG if i <= k else MUTED, a)
        return img
    return Scene("reward", render, [5, 5, 6, 6])


def l1_time():
    xs, _, _ = neat_trace()
    start, finish = 4.67, 88.67

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The hidden constraint: time", t)
        box = (140, 160, 1200, 600)
        x0, y0, x1, y1 = box
        axes(d, box, 0, 95, [0, 20, 40, 60, 80], xlabel="step → 1600")
        u = min(1, t / 6)
        n = int(len(xs) * u)
        d.line([(x0, y1 - (y1 - y0) * start / 95), (x1, y1 - (y1 - y0) * finish / 95)], fill=GREEN, width=3)
        d.line([(x0, y1 - (y1 - y0) * finish / 95), (x1, y1 - (y1 - y0) * finish / 95)], fill=C.mix(BG, AMBER, .5), width=1)
        text(d, (x0 + 10, y1 - (y1 - y0) * finish / 95 - 22), "finish (x ≈ 88.7)", 16, AMBER)
        series(d, box, xs[:max(2, n)], BLUE, 0, 95, n=1600, width=4)
        text(d, (x0 + 160, y1 - (y1 - y0) * 30 / 95 - 40), "needed: 2.6 units/s on average", 18, GREEN)
        a = beat_alpha(b, t, 1)
        if a > 0:
            xe = xs[-1]
            text(d, (x1 - 10, y1 - (y1 - y0) * xe / 95 + 26), f"NEAT walker: x = {xe:.0f} at step 1600, out of time", 20, BLUE, a, bold=True, anchor="ra")
            text(d, (x0, 650), "NEAT: rarely fell (9%), scored ≈ 175. Reaching 300 needs safe AND fast.", 22, FG, a)
        text(d, (x0, 130), "hull x position over one episode", 16, MUTED)
        return img
    return Scene("time", render, [7, 8])


def l1_outro():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 260), "policy π:  24 numbers  →  4 numbers", 44, FG, fade(t, .2), bold=True, anchor="mm")
        text(d, (W // 2, 340), "as much reward as possible, on tracks it has never seen", 24, MUTED, fade(t, .8), anchor="mm")
        text(d, (W // 2, 470), "Next: lesson 2, what that function looks like", 26, AMBER, fade(t, 2.5), anchor="mm")
        return img
    return Scene("outro", render, [6])


def lesson1():
    return [title_card("intro", "LESSON 1", "The task", "what the walker senses, does and is scored on", 6),
            l1_walker(), l1_senses(), l1_actions(), l1_reward(), l1_time(), l1_outro()]


# =============================================================================
# Lesson 2: a policy is just numbers
# =============================================================================

def _theta(seed=7, std=0.5):
    return np.random.default_rng(seed).normal(0, std, 932)


def champion_or_random():
    """The current CMA-ES champion's weights if one exists, else random (labelled)."""
    import glob
    best = sorted(glob.glob(os.path.join(C.REPO_ROOT, "runs", "*", "best", "params.npy")), key=os.path.getmtime)
    if best:
        meta = json.load(open(os.path.join(os.path.dirname(best[-1]), "meta.json")))
        if meta.get("sizes") == [24, 32, 4] and meta.get("mean", -1e9) > 100:
            return np.load(best[-1]), f"CMA-ES champion weights (held-out score {meta['mean']:.0f})"
    return _theta(), "random weights, as at the start of training"


def draw_mlp(d, box, theta, x, a=1.0, hl=None, labels=True):
    import common
    mlp = common.MLP((24, 32, 4), theta)
    (W1, b1), (W2, b2) = mlp.layers
    h = np.tanh(W1 @ x + b1)
    y = np.tanh(W2 @ h + b2)
    x0, y0, x1, y1 = box
    cols = [x0 + 110, (x0 + x1) / 2, x1 - 120]
    P = [[(cols[0], y0 + (y1 - y0) * i / 23) for i in range(24)],
         [(cols[1], y0 + (y1 - y0) * i / 31) for i in range(32)],
         [(cols[2], (y0 + y1) / 2 - 150 + i * 100) for i in range(4)]]
    for (A, Bp, Wm, src) in [(P[0], P[1], W1, x), (P[1], P[2], W2, h)]:
        sig = np.abs(Wm * src[None, :])
        order = np.argsort(sig, axis=None)[-260:]
        for flat in order:
            o, i = np.unravel_index(flat, Wm.shape)
            w = Wm[o, i]
            ea = a * min(.85, .08 + .45 * sig[o, i])
            d.line([A[i], Bp[o]], fill=C.mix(BG, BLUE if w > 0 else ORANGE, ea), width=1 if sig[o, i] < .8 else 2)
    vals = [x, h, y]
    for l, layer in enumerate(P):
        r = 6 if l < 2 else 15
        for i, (px, py) in enumerate(layer):
            v = float(np.clip(vals[l][i], -1, 1))
            fill = C.mix(PANEL, BLUE if v >= 0 else ORANGE, abs(v))
            out = AMBER if (hl is not None and l == 1 and i == hl) else C.mix(BG, FG, a)
            d.ellipse([px - r, py - r, px + r, py + r], fill=C.mix(BG, fill, a), outline=out, width=2)
            if labels and l == 0:
                text(d, (px - 12, py), C.INPUT_SHORT[i], 11, MUTED, a, anchor="rm")
            if labels and l == 2:
                text(d, (px + 24, py), f"{C.OUTPUT_SHORT[i]}  {v:+.2f}", 18, FG, a, anchor="lm")
    text(d, (cols[0], y1 + 22), "24 inputs", 16, MUTED, a, anchor="mm")
    text(d, (cols[1], y1 + 22), "32 hidden (tanh)", 16, MUTED, a, anchor="mm")
    text(d, (cols[2], y1 + 22), "4 outputs", 16, MUTED, a, anchor="mm")
    return y


def l2_neuron():
    ins = [("hull angle", 2.0), ("foot 1 contact", -1.5), ("vel x", 0.4)]
    bias = 0.2

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "One neuron", t)
        k, lt, _ = b.local(t)
        tilt = 0.5 * np.sin(t * 0.8) if k == 2 else 0.3
        foot = 0.0 if (k == 2 and np.sin(t * 0.8) > 0) else 1.0
        xv = [tilt, foot, 0.5]
        z = sum(w * v for (_, w), v in zip(ins, xv)) + bias
        for i, ((name, w), v) in enumerate(zip(ins, xv)):
            y = 220 + i * 110
            text(d, (70, y), f"{name}", 22, FG)
            text(d, (70, y + 30), f"x = {v:+.2f}", 20, MUTED, mono=True)
            d.line([(300, y + 20), (560, 330)], fill=BLUE if w > 0 else ORANGE, width=int(2 + abs(w) * 2))
            text(d, (390, y + 2 + (330 - y) * .35), f"w = {w:+.1f}", 18, BLUE if w > 0 else ORANGE, bold=True)
        d.ellipse([560, 280, 660, 380], outline=FG, width=3, fill=PANEL)
        text(d, (610, 330), "Σ + b", 24, FG, anchor="mm", bold=True)
        text(d, (610, 410), f"z = {z:+.2f}", 22, AMBER, mono=True, anchor="mm")
        a1 = beat_alpha(b, t, 1)
        if a1 > 0:
            box = (760, 200, 1200, 460)
            x0, y0, x1, y1 = box
            d.line([(x0, (y0 + y1) / 2), (x1, (y0 + y1) / 2)], fill=C.mix(BG, PANEL, a1), width=2)
            d.line([((x0 + x1) / 2, y0), ((x0 + x1) / 2, y1)], fill=C.mix(BG, PANEL, a1), width=2)
            zs = np.linspace(-4, 4, 200)
            pts = [(x0 + (x1 - x0) * (zz + 4) / 8, (y0 + y1) / 2 - np.tanh(zz) * (y1 - y0) / 2.2) for zz in zs]
            d.line(pts, fill=C.mix(BG, MUTED, a1), width=3)
            zc = float(np.clip(z, -4, 4))
            px, py = x0 + (x1 - x0) * (zc + 4) / 8, (y0 + y1) / 2 - np.tanh(z) * (y1 - y0) / 2.2
            d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=AMBER)
            text(d, (x0, y0 - 30), "y = tanh(z)  always between −1 and +1", 20, FG, a1)
            text(d, (980, 520), f"output y = {np.tanh(z):+.2f}", 30, AMBER, a1, bold=True, anchor="mm")
        a2 = beat_alpha(b, t, 2)
        if a2 > 0:
            paragraph(d, (70, 580), "fires positive when the body tilts forward AND foot 1 is in the air", 60, 22, GREEN, a2)
        return img
    return Scene("neuron", render, [6, 7, 7])


def l2_network():
    theta, lab = champion_or_random()
    p = neat_player()

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The walker's network: 24 → 32 → 4", t)
        k, lt, _ = b.local(t)
        if k == 0:
            x = np.zeros(24) + 0.3 * np.sin(np.arange(24) + t)
            draw_mlp(d, (60, 130, 1240, 660), theta, x, a=fade(t, .2, .8))
        else:
            p.at_time(lt, 0.5, start=300, loop_pause=0, length=1200)
            paste_frame(img, p.cur, (40, 470, 330, 220))
            draw_mlp(d, (380, 130, 1240, 660), theta, p.x_obs, labels=True)
            text(d, (40, 450), lab, 13, MUTED)
        return img
    return Scene("network", render, [8, 9])


def l2_hidden():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Why a hidden layer?", t)
        text(d, (70, 160), "without one (linear):", 26, MUTED, bold=True)
        text(d, (70, 205), "action = tanh(W · obs + b)", 26, FG, mono=True)
        paragraph(d, (70, 260), "\"push harder the more you tilt\"  ✓", 50, 24, GREEN)
        paragraph(d, (70, 300), "\"if foot 1 is down AND I tilt forward, swing leg 2\"  ✗", 50, 24, RED)
        a = beat_alpha(b, t, 0, .8) * min(1, max(0, (t - 4) / 1))
        text(d, (70, 400), "with hidden neurons:", 26, MUTED, a, bold=True)
        text(d, (70, 445), "action = tanh(W2 · tanh(W1 · obs + b1) + b2)", 26, FG, a, mono=True)
        paragraph(d, (70, 500), "each hidden neuron learns a combination of inputs; the outputs combine those", 60, 22, FG, a)
        a1 = beat_alpha(b, t, 1)
        if a1 > 0:
            d.rectangle([60, 580, 1220, 680], fill=C.mix(BG, PANEL, a1))
            text(d, (80, 600), "hard rule in this project:", 20, AMBER, a1, bold=True)
            text(d, (80, 635), 'assert len(sizes) >= 3, "the policy must have >= 1 hidden layer"', 20, FG, a1, mono=True, bg=PANEL)
        return img
    return Scene("hidden", render, [9, 6])


def l2_count():
    rows = [("W1  input → hidden", "24 × 32", 768), ("b1  hidden biases", "32", 32),
            ("W2  hidden → output", "32 × 4", 128), ("b2  output biases", "4", 4)]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Counting the numbers", t)
        T = b.end
        tot = 0
        for i, (name, shape, n) in enumerate(rows):
            a = fade(t, 1 + i * T * .18)
            y = 190 + i * 90
            text(d, (120, y), name, 30, FG, a)
            text(d, (700, y), shape, 30, MUTED, a, mono=True)
            text(d, (1050, y), f"{n:>4}", 34, AMBER, a, mono=True, bold=True, anchor="ra")
            if a > .5:
                tot += n
        a = fade(t, 1 + 4 * T * .18)
        d.line([(120, 560), (1060, 560)], fill=C.mix(BG, MUTED, a), width=2)
        text(d, (120, 590), "total", 34, FG, a, bold=True)
        text(d, (1050, 590), f"{tot:>4}", 44, AMBER, a, mono=True, bold=True, anchor="ra")
        return img
    return Scene("count", render, [14])


def draw_theta(d, box, theta, prev=None, a=1.0):
    x0, y0, x1, y1 = box
    w = (x1 - x0) / len(theta)
    for k, v in enumerate(theta):
        c = C.mix(BG, BLUE if v >= 0 else ORANGE, a * min(1, abs(v) / 1.2))
        d.rectangle([x0 + k * w, y0, x0 + (k + 1) * w, y1], fill=c)
        if prev is not None and abs(prev[k] - v) > 1e-9:
            d.rectangle([x0 + k * w, y1 + 6, x0 + (k + 1) * w, y1 + 6 + min(30, abs(prev[k] - v) * 60)], fill=C.mix(BG, AMBER, a))
    for lab, lo in [("W1 (768)", 0), ("b1", 768), ("W2 (128)", 800), ("b2", 928)]:
        d.line([(x0 + lo * w, y0 - 8), (x0 + lo * w, y1)], fill=C.mix(BG, FG, a), width=1)
        text(d, (x0 + lo * w + 4, y0 - 28), lab, 15, MUTED, a)


def l2_theta():
    theta = _theta()
    noise = np.random.default_rng(3).normal(0, 0.1, 932)
    _, _, obs = neat_trace()
    x = obs[400]
    import common

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "θ: the whole walker in one list", t)
        k, lt, _ = b.local(t)
        u = 0 if k == 0 else min(1, lt / 2.5)
        th = theta + u * noise
        draw_theta(d, (60, 220, 1220, 330), th, theta if k == 1 else None)
        text(d, (60, 380), "932 numbers  ·  saved as one 7 KB file  ·  this list IS the walker", 24, FG)
        if k == 1:
            y0 = common.MLP((24, 32, 4), theta).activate(x)
            y1 = common.MLP((24, 32, 4), th).activate(x)
            text(d, (60, 450), "nudge θ by σ = 0.1 × random noise:", 24, AMBER, bold=True)
            for i in range(4):
                text(d, (80 + i * 290, 510), C.OUTPUT_SHORT[i], 22, MUTED)
                text(d, (80 + i * 290, 545), f"{y0[i]:+.2f} → {y1[i]:+.2f}", 26, FG, mono=True)
            paragraph(d, (60, 620), "a slightly different walker: exactly how training makes new candidates", 70, 22, MUTED)
        return img
    return Scene("theta", render, [7, 8])


def landscape_img(f, R=5, size=(560, 560), lo=0, hi=60, dark=True):
    n = 200
    xs = np.linspace(-R, R, n)
    X, Y = np.meshgrid(xs, xs[::-1])
    Z = np.vectorize(lambda a, b2: f(np.array([a, b2])))(X, Y)
    v = np.clip((np.log1p(Z) - np.log1p(lo)) / (np.log1p(hi) - np.log1p(lo)), 0, 1)
    band = (np.floor(v * 14) % 2) * .07
    g = np.stack([20 + 30 * (1 - v), 40 + 110 * (1 - v), 60 + 60 * (1 - v)], -1) * (1 - band[..., None])
    return Image.fromarray(g.astype(np.uint8)).resize(size, Image.BILINEAR)


def l2_space():
    def f(x):
        return .3 * (x[0] ** 2 + x[1] ** 2) + 2.5 * (2 - np.cos(1.6 * x[0]) - np.cos(1.6 * x[1] + 1))
    bg = landscape_img(f, hi=30)
    pts = np.random.default_rng(5).uniform(-4.6, 4.6, (60, 2))

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "One walker = one point", t)
        k, lt, _ = b.local(t)
        if k == 0:
            img.paste(bg, (60, 130))
            d = ImageDraw.Draw(img)
        else:
            d.rectangle([60, 130, 620, 690], fill=PANEL)
        n = int(min(60, 3 + t * 6))
        for i, (px, py) in enumerate(pts[:n]):
            X, Y = 60 + (px + 5) / 10 * 560, 130 + (5 - py) / 10 * 560
            col = AMBER if k == 1 else FG
            d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=col)
            if k == 1 and i < 12:
                text(d, (X + 8, Y - 8), f"{-f((px, py)) * 10:.0f}", 13, FG)
        paragraph(d, (680, 170), "932 axes, one per number. Every possible walker is a point. Its height = average score on a few tracks.", 34, 24, FG)
        if k == 0:
            paragraph(d, (680, 330), "training = find a high point", 34, 28, GREEN)
        a1 = beat_alpha(b, t, 1)
        if a1 > 0:
            paragraph(d, (680, 330), "but we can only measure heights at points we try. No map, no slope: the physics can't be differentiated, falls are sudden, every track is different.", 34, 24, AMBER, a1)
            text(d, (680, 560), "→ lesson 3: climbing by sampling", 26, FG, a1, bold=True)
        return img
    return Scene("space", render, [8, 9])


def lesson2():
    return [title_card("intro", "LESSON 2", "A policy is just numbers", "932 numbers and a little arithmetic", 6),
            l2_neuron(), l2_network(), l2_hidden(), l2_count(), l2_theta(), l2_space()]


# =============================================================================
# Lesson 3: CMA-ES
# =============================================================================

def valley(x):
    a = .6
    c, s = np.cos(a), np.sin(a)
    u, v = c * x[0] + s * x[1], -s * x[0] + c * x[1]
    return u * u + 25 * v * v


def bumpy(x):
    return .5 * (x[0] ** 2 + x[1] ** 2) + 3 * (2 - np.cos(2.2 * x[0]) - np.cos(2.2 * x[1]))


@_cache
def run_cma(fname, lam, seed, gens=150, start=(-3.5, 3.0)):
    f = {"valley": valley, "bumpy": bumpy}[fname]
    es = cma.CMAEvolutionStrategy(list(start), 1.0, {"popsize": lam, "seed": seed, "verbose": -9,
                                                     "tolx": 0, "tolfun": 0, "tolfunhist": 0})
    hist = []
    for g in range(gens):
        m, s, Cm = es.mean.copy(), es.sigma, es.sm.C.copy()
        X = es.ask()
        fs = [f(x) for x in X]
        es.tell(X, fs)
        hist.append({"mean": m, "sigma": s, "C": Cm, "X": np.array(X), "f": np.array(fs),
                     "new_mean": es.mean.copy(), "err": f(m)})
    return hist


@_cache
def run_plain(fname, lam, seed, gens=150, start=(-3.5, 3.0), sigma=1.0):
    f = {"valley": valley, "bumpy": bumpy}[fname]
    rng = np.random.default_rng(seed)
    m = np.array(start, float)
    mu = lam // 2
    hist = []
    for g in range(gens):
        X = m + sigma * rng.normal(size=(lam, 2))
        fs = np.array([f(x) for x in X])
        nm = X[np.argsort(fs)[:mu]].mean(0)
        hist.append({"mean": m.copy(), "sigma": sigma, "C": np.eye(2), "X": X, "f": fs, "new_mean": nm, "err": f(m)})
        m = nm
    return hist


class Panel:
    def __init__(self, f, box, hi=60):
        self.box = box
        self.bg = landscape_img(f, size=(box[2], box[3]), hi=hi)

    def X(self, p):
        x, y, w, h = self.box
        return x + (p[0] + 5) / 10 * w, y + (5 - p[1]) / 10 * h

    def draw(self, img, g, trail, show_pop=True, ranked=True, show_new=False, title=None, opt=(0, 0)):
        img.paste(self.bg, self.box[:2])
        d = ImageDraw.Draw(img)
        ox, oy = self.X(opt)
        d.line([(ox - 9, oy), (ox + 9, oy)], fill=FG, width=2)
        d.line([(ox, oy - 9), (ox, oy + 9)], fill=FG, width=2)
        if len(trail) > 1:
            d.line([self.X(p) for p in trail], fill=AMBER, width=3)
        lam_, vec = np.linalg.eigh(g["C"])
        ang = np.linspace(0, 2 * np.pi, 80)
        for k, col in [(1, BLUE), (2, C.mix(BG, BLUE, .5))]:
            pts = [self.X(g["mean"] + k * g["sigma"] * (vec @ (np.sqrt(np.maximum(lam_, 0)) * [np.cos(a), np.sin(a)]))) for a in ang]
            d.line(pts + [pts[0]], fill=col, width=2)
        if show_pop:
            order = np.argsort(g["f"])
            good = set(order[:len(order) // 2])
            for i, x in enumerate(g["X"]):
                px, py = self.X(x)
                if px < self.box[0] or py < self.box[1] or px > self.box[0] + self.box[2] or py > self.box[1] + self.box[3]:
                    continue
                c = (GREEN if i in good else MUTED) if ranked else FG
                r = 6 if (ranked and i in good) else 4
                d.ellipse([px - r, py - r, px + r, py + r], fill=c)
        m = g["new_mean"] if show_new else g["mean"]
        px, py = self.X(m)
        d.ellipse([px - 9, py - 9, px + 9, py + 9], fill=AMBER, outline=FG, width=2)
        if title:
            label(d, (self.box[0] + 10, self.box[1] + 10), title, 18)


def l3_loop():
    hist = run_cma("valley", 8, 2)
    pan = Panel(valley, (60, 120, 560, 560))
    steps = ["1  sample λ candidates around the mean (cloud size = σ)",
             "2  evaluate and rank: keep the best half",
             "3  move the mean to their weighted average",
             "4  repeat: one round = one generation"]

    def render(t, b):
        img = blank()
        k, lt, _ = b.local(t)
        if k <= 3:
            g = hist[0]
            pan.draw(img, g, [g["mean"]] + ([g["new_mean"]] if k >= 3 else []), show_pop=k >= 1, ranked=k >= 2, show_new=k >= 3)
        else:
            n = min(len(hist) - 1, 1 + int(lt * 3))
            pan.draw(img, hist[n], [h["mean"] for h in hist[:n + 1]], title=f"generation {n}")
        d = ImageDraw.Draw(img)
        text(d, (60, 50), "An evolution strategy, one generation at a time", 32, FG, bold=True)
        for i, s in enumerate(steps):
            a = 1.0 if k >= 4 else (beat_alpha(b, t, i + 1) if i < 3 else beat_alpha(b, t, 4))
            col = AMBER if (k == i + 1 or (k == 4 and i == 3)) else FG
            paragraph(d, (680, 170 + i * 100), s, 34, 22, col, a)
        a = beat_alpha(b, t, 4, .8)
        paragraph(d, (680, 590), "only the ranking matters: one terrible score can't drag the search off course", 36, 20, GREEN, a)
        return img
    return Scene("loop", render, [6, 6, 6, 6, 8])


def l3_compare():
    hc, hp = run_cma("valley", 8, 2), run_plain("valley", 8, 2)
    A, B = Panel(valley, (40, 130, 580, 520)), Panel(valley, (660, 130, 580, 520))

    def render(t, b):
        img = blank()
        n = min(len(hc) - 1, int(t * 2.2))
        A.draw(img, hc[n], [h["mean"] for h in hc[:n + 1]], title="CMA-ES")
        B.draw(img, hp[n], [h["mean"] for h in hp[:n + 1]], title="plain ES: fixed σ, round cloud")
        d = ImageDraw.Draw(img)
        text(d, (40, 50), f"Same landscape, same λ = 8   ·   generation {n}", 30, FG, bold=True)
        text(d, (40, 670), f"error {hc[n]['err']:.1e}", 20, AMBER, mono=True)
        text(d, (660, 670), f"error {hp[n]['err']:.1e}", 20, MUTED, mono=True)
        return img
    return Scene("compare", render, [7, 9])


def l3_sigma():
    hc, hp = run_cma("valley", 8, 2), run_plain("valley", 8, 2)
    ec = np.log10(np.array([h["err"] for h in hc]) + 1e-30)
    ep = np.log10(np.array([h["err"] for h in hp]) + 1e-30)
    sc = np.log10(np.array([h["sigma"] for h in hc]))

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Step size σ adapts by itself", t)
        box = (140, 150, 1200, 560)
        lo, hi = -32, 4
        axes(d, box, lo, hi, [-30, -20, -10, 0], xlabel="generation → 150", fmt="1e{:.0f}")
        n = int(min(150, 10 + t * 14))
        series(d, box, sc[:n], BLUE, lo, hi, n=150)
        series(d, box, ep[:n], MUTED, lo, hi, n=150)
        series(d, box, np.maximum(ec[:n], lo), AMBER, lo, hi, n=150, width=4)
        text(d, (140, 600), "CMA-ES error", 20, AMBER, bold=True)
        text(d, (360, 600), "CMA-ES σ", 20, BLUE, bold=True)
        text(d, (540, 600), "plain ES error (fixed σ = 1)", 20, MUTED, bold=True)
        a = beat_alpha(b, t, 1)
        text(d, (140, 650), f"after 150 generations: CMA-ES {10 ** ec[-1]:.0e} vs plain ES {10 ** ep[-1]:.0e}", 22, FG, a)
        return img
    return Scene("sigma", render, [9, 7])


def l3_cov():
    hc = run_cma("valley", 8, 2)
    rng = np.random.default_rng(4)
    pts = rng.multivariate_normal([0, 0], [[1, .85], [.85, 1]], 40)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The covariance matrix C: the shape of the cloud", t)
        k, lt, _ = b.local(t)
        if k == 0:
            pan = Panel(valley, (60, 130, 520, 520))
            n = min(40, int(lt * 4))
            pan.draw(img, hc[n], [h["mean"] for h in hc[:n + 1]], show_pop=False, title=f"generation {n}")
            d = ImageDraw.Draw(img)
            paragraph(d, (640, 170), "2-D: C is an ellipse", 30, 28, FG)
            paragraph(d, (640, 250), "the walker: a 932 × 932 table", 30, 28, AMBER)
            paragraph(d, (640, 300), "= 868,624 numbers, one for every pair of weights", 30, 24, MUTED)
        else:
            x0, y0, s = 340, 410, 110
            d.line([(x0 - 260, y0), (x0 + 260, y0)], fill=PANEL, width=2)
            d.line([(x0, y0 - 260), (x0, y0 + 260)], fill=PANEL, width=2)
            text(d, (x0 + 250, y0 + 20), "hip weight", 18, MUTED, anchor="ra")
            text(d, (x0 + 10, y0 - 250), "knee weight", 18, MUTED)
            u = min(1, lt / 3)
            for p in pts[:int(5 + u * 35)]:
                X, Y = x0 + p[0] * s, y0 - p[1] * s
                d.ellipse([X - 5, Y - 5, X + 5, Y + 5], fill=GREEN)
            ang = np.linspace(0, 2 * np.pi, 80)
            lam_, vec = np.linalg.eigh(np.array([[1, .85 * u], [.85 * u, 1]]))
            e = [(x0 + s * 2 * (vec @ (np.sqrt(lam_) * [np.cos(a), np.sin(a)]))[0], y0 - s * 2 * (vec @ (np.sqrt(lam_) * [np.cos(a), np.sin(a)]))[1]) for a in ang]
            d.line(e + [e[0]], fill=BLUE, width=3)
            paragraph(d, (700, 200), "Successful steps raised both weights together.", 30, 26, FG)
            paragraph(d, (700, 300), "C learns that correlation, and new candidates change both at once.", 30, 26, AMBER)
        return img
    return Scene("cov", render, [9, 8])


def l3_pop():
    small, big = run_cma("bumpy", 6, 5, 80, (-3.8, 3.4)), run_cma("bumpy", 30, 1, 80, (-3.8, 3.4))
    A, B = Panel(bumpy, (40, 130, 560, 500), hi=25), Panel(bumpy, (680, 130, 560, 500), hi=25)

    def render(t, b):
        img = blank()
        n = min(79, int(t * 4))
        A.draw(img, small[n], [h["mean"] for h in small[:n + 1]], title="λ = 6")
        B.draw(img, big[n], [h["mean"] for h in big[:n + 1]], title="λ = 30")
        d = ImageDraw.Draw(img)
        text(d, (40, 50), "Population size on a landscape full of small traps", 30, FG, bold=True)
        a = fade(t, 12)
        text(d, (40, 655), "λ = 6: trapped in 19 of 40 runs", 22, RED, a, bold=True)
        text(d, (680, 655), "λ = 30: trapped in 1 of 40 runs", 22, GREEN, a, bold=True)
        return img
    return Scene("pop", render, [16])


CODE = """es = cma.CMAEvolutionStrategy(np.zeros(932), 0.5, {"popsize": 48})
while not done:
    thetas  = es.ask()                     # 48 candidate walkers
    tracks  = random_tracks(k=5)           # the same 5 for all 48
    fitness = [mean(walk(θ, tr) for tr in tracks) for θ in thetas]
    es.tell(thetas, [-f for f in fitness]) # cma minimises
    # every 10 generations: test the mean + best 3 on 20 held-out tracks"""


def l3_code():
    lines = CODE.split("\n")

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The walker's training loop (train.py)", t)
        d.rectangle([50, 150, 1230, 520], fill=(7, 11, 24))
        for i, ln in enumerate(lines):
            a = fade(t, 0.6 + i * b.end / 9)
            text(d, (75, 175 + i * 46), ln, 20, FG, a, mono=True, bg=(7, 11, 24))
        a = fade(t, b.end * .75)
        text(d, (60, 570), "48 walkers × 5 tracks = 240 episodes per generation, on 4 CPU cores", 24, AMBER, a)
        return img
    return Scene("code", render, [18])


def l3_why():
    rows = [("", "evolution strategy", "PPO / TD3 (gradients)"),
            ("learns from", "whole-episode totals", "individual steps"),
            ("moving parts", "few: λ, σ0", "many: critic, buffers, rates"),
            ("parallel", "trivially", "harder"),
            ("samples", "many", "fewer")]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Why not gradients?", t)
        for i, (a_, b_, c_) in enumerate(rows):
            al = fade(t, .5 + i * 1.2)
            y = 170 + i * 80
            text(d, (70, y), a_, 24, MUTED, al)
            text(d, (380, y), b_, 26, AMBER if i == 0 else FG, al, bold=i == 0)
            text(d, (820, y), c_, 26, BLUE if i == 0 else FG, al, bold=i == 0)
        text(d, (70, 620), "Next: lesson 4, when every measurement is noisy", 26, AMBER, fade(t, b.end - 5))
        return img
    return Scene("why", render, [16])


def lesson3():
    return [title_card("intro", "LESSON 3", "Evolution strategies and CMA-ES", "sample, rank, move, stretch", 6),
            l3_loop(), l3_compare(), l3_sigma(), l3_cov(), l3_pop(), l3_code(), l3_why()]


# =============================================================================
# Lesson 4: noisy fitness
# =============================================================================

def l4_spread():
    d100 = json.load(open(os.path.join(C.DATA, "neat_100.json")))
    r = np.array(d100["rewards"])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "One walker, 100 different tracks", t)
        n = int(min(100, t * 14))
        bins = np.linspace(-100, 300, 21)
        cnt = np.histogram(r[:n], bins)[0]
        x0, y0, x1, y1 = 120, 160, 1180, 560
        mx = max(12, cnt.max())
        for i, c in enumerate(cnt):
            bx = x0 + (x1 - x0) * i / 20
            d.rectangle([bx + 2, y1 - (y1 - y0) * c / mx, bx + (x1 - x0) / 20 - 2, y1], fill=BLUE)
        for v in range(-100, 301, 100):
            text(d, (x0 + (x1 - x0) * (v + 100) / 400, y1 + 20), str(v), 16, MUTED, anchor="mm")
        text(d, (x1, y1 + 50), "score on one track (NEAT champion, same weights every time)", 16, MUTED, anchor="ra")
        if n >= 100:
            text(d, (x0, 640), f"mean {r.mean():.0f}, spread ± {r.std():.0f}", 28, FG, bold=True)
        a = beat_alpha(b, t, 1)
        text(d, (680, 640), "average of K tracks: luck shrinks like 1/√K", 22, AMBER, a)
        return img
    return Scene("spread", render, [9, 8])


@_cache
def curse_sim():
    rng = np.random.default_rng(11)
    N, NOISE, SH = 48, 65, 50
    OWN = np.sqrt(NOISE ** 2 - SH ** 2)
    skill = np.linspace(150, 250, N)

    def measure(K, shared=True):
        tracks = SH * rng.normal(size=K)
        noise = (tracks[None, :] if shared else SH * rng.normal(size=(N, K))) + OWN * rng.normal(size=(N, K))
        return skill + noise.mean(1)
    one = measure(5)
    infl = {K: np.array([(lambda m: m[np.argmax(m)] - skill[np.argmax(m)])(measure(K)) for _ in range(2000)]) for K in (1, 5)}
    return skill, one, infl


def l4_curse():
    skill, m, infl = curse_sim()
    w = int(np.argmax(m))

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The winner's curse", t)
        k, lt, _ = b.local(t)
        if k < 2:
            X = lambda v: 120 + (v - 50) / 350 * 1060
            for v in range(50, 401, 50):
                text(d, (X(v), 690), str(v), 15, MUTED, anchor="mm")
            for i in range(48):
                y = 140 + i * 11
                d.line([(X(skill[i]), y), (X(m[i]), y)], fill=PANEL, width=1)
                d.ellipse([X(skill[i]) - 3, y - 3, X(skill[i]) + 3, y + 3], fill=MUTED)
                r = 9 if (k == 1 and i == w) else 4
                d.ellipse([X(m[i]) - r, y - r, X(m[i]) + r, y + r], fill=AMBER if (k == 1 and i == w) else BLUE)
            text(d, (120, 115), "grey = true skill   ·   blue = measured on 5 tracks", 18, MUTED)
            if k == 1:
                y = 140 + w * 11
                label(d, (min(X(m[w]) + 16, 900), y - 10), f"picked: measured {m[w]:.0f}, true {skill[w]:.0f}", 18)
        else:
            x0, y0, x1, y1 = 120, 170, 1180, 560
            for K, col, yy in [(1, RED, 0), (5, AMBER, 1)]:
                cnt = np.histogram(infl[K], np.linspace(-60, 240, 31))[0]
                mx = cnt.max()
                for i, c in enumerate(cnt):
                    bx = x0 + (x1 - x0) * i / 30
                    h = (y1 - y0) * c / mx * .9
                    d.rectangle([bx + 2 + yy * 6, y1 - h, bx + (x1 - x0) / 30 - 8 + yy * 6, y1], fill=C.mix(BG, col, .75))
            for v in range(-60, 241, 60):
                text(d, (x0 + (x1 - x0) * (v + 60) / 300, y1 + 20), f"{v:+d}", 16, MUTED, anchor="mm")
            text(d, (x0, y1 + 50), "how much better the winner looked than it really was (2,000 simulated generations)", 16, MUTED)
            text(d, (x0, 640), f"5 tracks: ≈ {infl[5].mean():.0f} points too high", 26, AMBER, bold=True)
            text(d, (x0 + 560, 640), f"1 track: ≈ {infl[1].mean():.0f} points too high", 26, RED, bold=True)
        return img
    return Scene("curse", render, [7, 7, 9])


def l4_crn():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Common random numbers", t)
        rng = np.random.default_rng(2)
        skill = np.array([200, 220, 180, 240])
        cols = [BLUE, GREEN, ORANGE, AMBER]
        for side, (x0, title, shared) in enumerate([(80, "different tracks per candidate", False), (680, "the same 5 tracks for everyone", True)]):
            text(d, (x0, 140), title, 24, FG, bold=True)
            hard = rng.normal(0, 50, 4) if not shared else np.full(4, -60.0)
            meas = skill + hard
            for i in range(4):
                y = 200 + i * 80
                text(d, (x0, y + 10), f"candidate {i + 1}", 18, MUTED)
                L = 260 * meas[i] / 260
                d.rectangle([x0 + 150, y, x0 + 150 + L * 1.6, y + 40], fill=cols[i])
                text(d, (x0 + 160 + L * 1.6, y + 20), f"{meas[i]:.0f}", 18, FG, anchor="lm")
            rank = ", ".join(f"#{j + 1}" for j in np.argsort(-meas))
            true = ", ".join(f"#{j + 1}" for j in np.argsort(-skill))
            text(d, (x0, 540), f"measured ranking: {rank}", 20, FG)
            text(d, (x0, 575), f"true ranking:     {true}", 20, MUTED)
        a = beat_alpha(b, t, 1)
        text(d, (80, 640), "a hard track lowers everyone together, so it cancels out of the ranking", 24, GREEN, a)
        return img
    return Scene("crn", render, [10, 8])


def l4_champ():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Choosing the champion", t)
        boxes = [("every 10 generations", "CMA mean + best 3 candidates", BLUE),
                 ("20 held-out tracks", "seeds 10,000,000+, never trained on; full episodes, raw reward", AMBER),
                 ("champion", "best held-out score so far", GREEN)]
        for i, (h, s, c) in enumerate(boxes):
            a = fade(t, .5 + i * 2.5)
            x = 60 + i * 410
            d.rounded_rectangle([x, 200, x + 360, 420], 14, outline=C.mix(BG, c, a), width=3, fill=C.mix(BG, PANEL, a))
            text(d, (x + 20, 225), h, 24, c, a, bold=True, bg=PANEL)
            paragraph(d, (x + 20, 275), s, 26, 20, FG, a)
            if i < 2:
                text(d, (x + 385, 310), "→", 36, FG, a, anchor="mm")
        a = beat_alpha(b, t, 1)
        paragraph(d, (60, 480), "The mean is the centre of the cloud: an average of many good candidates. Tricks that only worked on lucky tracks tend to cancel out.", 80, 24, FG, a)
        return img
    return Scene("champ", render, [10, 8])


def l4_claim():
    rows = [("training", "0 – 999,999"), ("hall of fame", "10,000,000+"), ("evaluation (final score)", "20,000,000+"),
            ("videos", "30,000,000+"), ("comparisons", "40,000,000+")]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Claiming a result", t)
        k, _, _ = b.local(t)
        if k == 0:
            for i, (a_, b_) in enumerate(rows):
                al = fade(t, .4 + i * .8)
                text(d, (100, 170 + i * 70), a_, 26, AMBER if i == 2 else FG, al, bold=i == 2)
                text(d, (700, 170 + i * 70), b_, 26, MUTED, al, mono=True)
            text(d, (100, 560), "seed ranges never overlap (a test checks it)", 22, GREEN, fade(t, 5))
        else:
            text(d, (W // 2 - 280, 300), "200", 110, AMBER, bold=True, anchor="mm")
            text(d, (W // 2 - 280, 390), "during selection", 24, MUTED, anchor="mm")
            text(d, (W // 2, 300), "→", 80, FG, anchor="mm")
            text(d, (W // 2 + 280, 300), "≈ 175", 110, BLUE, bold=True, anchor="mm")
            text(d, (W // 2 + 280, 390), "on 100 new tracks", 24, MUTED, anchor="mm")
            text(d, (W // 2, 540), "comparing two walkers: same 100 tracks, differences < ~20 are usually noise", 24, FG, fade(t, b.at(1) + 4), anchor="mm")
        return img
    return Scene("claim", render, [8, 10])


def l4_outro():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 300), "That's the background.", 44, FG, fade(t, .2), bold=True, anchor="mm")
        text(d, (W // 2, 380), "Next: lesson 5, the training run, including three false starts", 26, AMBER, fade(t, 1.2), anchor="mm")
        return img
    return Scene("outro", render, [6])


def lesson4():
    return [title_card("intro", "LESSON 4", "Noisy fitness, honest selection", "how not to fool yourself", 6),
            l4_spread(), l4_curse(), l4_crn(), l4_champ(), l4_claim(), l4_outro()]


LESSONS = {"01-the-task": ("lesson-01", lesson1), "02-a-policy-is-numbers": ("lesson-02", lesson2),
           "03-cma-es": ("lesson-03", lesson3), "04-noisy-fitness": ("lesson-04", lesson4)}
