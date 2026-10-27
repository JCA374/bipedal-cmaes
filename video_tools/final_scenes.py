"""Scenes for lessons 5-7 and the explainer (narration from video_tools/narration.py).

Beat counts come from the narration files, so they always match. The explainer reuses
lesson scenes through `remap`, which plays chosen beats of an existing scene.
"""
import csv
import json
import os
import re
import sys

import numpy as np
from PIL import ImageDraw

import lesson_scenes as LS
import vcommon as C
from timeline import Beats
from vcommon import (AMBER, BG, BLUE, FG, GREEN, MUTED, ORANGE, PANEL, RED, TEAL, W, H, Scene, axes, bars, blank,
                     cover_frame, fade, label, paragraph, paste_frame, series, text, title_card)

sys.path.insert(0, os.path.join(C.REPO_ROOT, "experiments"))
import common  # noqa: E402

DATA = os.path.join(C.REPO_ROOT, "docs", "data")
NARR = os.path.join(C.REPO_ROOT, "video_tools", "narration")
RUN = os.environ.get("RUN_DIR", os.path.join(C.REPO_ROOT, "runs", "v3-f20"))
J = lambda f: json.load(open(os.path.join(DATA, f)))
heading = C.heading


def nbeats(doc, scene):
    """Number of `## scene` / `## scene.n` sections in a narration file."""
    txt = open(os.path.join(NARR, f"{doc}.md")).read()
    return len(re.findall(rf"^## {re.escape(scene)}(?:\.\d+)?\s*$", txt, flags=re.M))


def remap(scene, name, modes):
    """A new scene named `name` whose beat k shows beat modes[k] of `scene`."""
    n = len(scene.default)

    def render(t, b):
        k, lt, dur = b.local(t)
        mode = modes[min(k, len(modes) - 1)]
        fake = Beats([dur if i == mode else 0.001 for i in range(n)])
        return scene.render(fake.starts[mode] + lt, fake)
    return Scene(name, render, [scene.default[m] for m in modes], [scene.min[m] for m in modes])


@LS._cache
def stats():
    return list(csv.DictReader(open(os.path.join(RUN, "stats.csv"))))


@LS._cache
def champion():
    return common.load_policy(RUN)


@LS._cache
def champ_player(seed=30_000_008):
    return C.Player(champion()[0], seed)


@LS._cache
def arche_player(key):
    sys.path.insert(0, os.path.join(C.REPO_ROOT, "experiments"))
    import archetypes
    path = {a[0]: a[2] for a in archetypes.ARCHETYPES}[key]
    pol, _, _ = archetypes.load(path, os.path.relpath(RUN, C.REPO_ROOT))
    return C.Player(pol, 500_003)


# =============================================================================
# Lesson 5
# =============================================================================

def l5_shaping(modes=(0, 1)):
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Reported score vs training fitness", t)
        d.rounded_rectangle([60, 150, 600, 560], 16, outline=BLUE, width=3)
        text(d, (90, 175), "reported score", 30, BLUE, bold=True)
        paragraph(d, (90, 240), "raw environment reward, full 1600-step episodes, never changed", 28, 24, FG)
        k, _, _ = b.local(t)
        a = beat_alpha(b, t, 0, .6)
        d.rounded_rectangle([680, 150, 1220, 560], 16, outline=C.mix(BG, AMBER, a), width=3)
        text(d, (710, 175), "training fitness", 30, AMBER, a, bold=True)
        paragraph(d, (710, 240), "what CMA-ES climbs, ours to design:", 28, 24, FG, a)
        a2 = beat_alpha(b, t, 1)
        for i, s in enumerate(["early stop when the walker stops moving", "the price of a fall", "the price of a stall"]):
            C.bullet(d, (710, 330 + i * 60), s, a2, 22)
        return img
    return Scene("shaping", render, [6, 8])


def beat_alpha(b, t, k, dur=.4):
    return fade(t, b.at(k), dur)


def l5_traps(name="traps", modes=(0, 1, 2, 3)):
    A = {a["key"]: a for a in J("archetypes.json")["archetypes"]}
    cases = [("v0", "fall 100 · stall 100", "dive", 100, 100), ("v1", "fall 0 · stall 100", "lunge", 0, 100),
             ("v2", "fall 0 · stall 0", "early", 0, 0), ("v3", "fall 20 · stall 0", "stand", 20, 0)]

    def render(t, b):
        k, lt, _ = b.local(t)
        run, setting, key, fc, sc = cases[modes[k]]
        img = blank()
        p = arche_player(key if key != "stand" else "walk")
        paste_frame(img, p.at_time(lt, 0.8, loop_pause=0.8, length=400 if key == "stand" else 300), (40, 130, 600, 400),
                    border=GREEN if key == "stand" else RED)
        d = ImageDraw.Draw(img)
        text(d, (40, 50), f"Attempt {run}:  {setting}", 34, FG, bold=True)
        label(d, (55, 145), {"dive": "converged to: dive and fall", "lunge": "converged to: lunge and fall",
                             "early": "converged to: fall early", "stand": "the run that walked"}[key], 18)
        rows = sorted(((a["name"].split(" (")[0], a["raw"] + (100 - fc) * a["fell"] - sc * a["stalled"], a["key"])
                       for a in A.values() if a["key"] in ("stand", "dive", "lunge", "early", "random")), key=lambda r: -r[1])
        text(d, (680, 140), "training fitness of each behaviour", 20, MUTED)
        for i, (nm, f, kk) in enumerate(rows):
            y = 180 + i * 62
            col = AMBER if i == 0 else FG
            text(d, (680, y), f"{i + 1}. {nm}", 22, col, bold=i == 0)
            text(d, (1220, y), f"{f:+.0f}", 22, col, mono=True, anchor="ra")
        paragraph(d, (40, 560), {"dive": "standing (−108) scores worse than falling (−98): falling forward wins, and every dive scores alike",
                                 "lunge": "lunge, catch yourself, stand: cut off by the stall detector and charged 100",
                                 "early": "a random net pays ≈0.11 per step in effort: ending the episode early is cheapest",
                                 "stand": "standing beats falling; from standing, every step forward adds reward"}[key], 60, 22, MUTED)
        return img
    return Scene(name, render, [8] * len(modes))


def l5_breakthrough(name="breakthrough", modes=(0, 1)):
    rows = stats()
    best = np.array([float(r["best_fitness"]) for r in rows])
    mean = np.array([float(r["mean_fitness"]) for r in rows])
    champ = [float(r["best_score"]) if r["best_score"] else np.nan for r in rows]
    n = min(len(rows), 140)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "The breakthrough", t)
        k, lt, _ = b.local(t)
        mode = modes[k]
        box = (120, 150, 1200, 600)
        axes(d, box, -60, 260, [0, 100, 200], xlabel=f"generation → {n}")
        m = n if mode == 1 else int(min(n, 55 + lt * 3))
        series(d, box, mean[:m], MUTED, -60, 260, n=n, width=2)
        series(d, box, best[:m], AMBER, -60, 260, n=n, width=3)
        if mode == 1:
            ch = np.array(champ[:n], float)
            ok = ~np.isnan(ch)
            if ok.any():
                first = int(np.argmax(ok))
                series(d, (box[0] + (box[2] - box[0]) * first / (n - 1), box[1], box[2], box[3]), ch[first:], BLUE, -60, 260, n=n - first, width=4)
            text(d, (120, 640), "orange: best candidate · grey: mean of 48 · blue: champion on 20 held-out tracks", 18, MUTED)
        return img
    return Scene(name, render, [7] * len(modes))


def l5_stages(name="stages", modes=None):
    import gzip
    import pickle
    st = pickle.load(gzip.open(os.path.join(RUN, "checkpoints", "latest.pkl.gz")))["state"]
    order = [k for k in ("first_positive", "100", "200", "300") if k in st["milestones"]]
    groups = []
    for k in order:
        g = st["milestones"][k]["generation"]
        if groups and groups[-1][1] == g:
            groups[-1][0].append(k)
        else:
            groups.append(([k], g))
    players = [C.Player(common.load_policy(os.path.join(RUN, "milestones", grp[-1]))[0], 30_000_008) for grp, _ in groups]
    gen0 = C.Player(common.load_policy(os.path.join(RUN, "milestones", "gen0"))[0], 30_000_008)
    modes = modes or list(range(len(groups)))

    def render(t, b):
        img = blank()
        k, lt, _ = b.local(t)
        i = modes[k]
        grp, g = groups[i]
        p = players[i]
        cover_frame(img, p.at_time(lt, 1.5, loop_pause=0.5))
        d = ImageDraw.Draw(img)
        sc = st["milestones"][grp[-1]]["mean"]
        label(d, (40, 40), f"generation {g}: champion {sc:.0f} on 20 held-out tracks", 24)
        label(d, (40, 90), f"step {p.step}   reward {p.total:.0f}", 18)
        return img
    return Scene(name, render, [6] * len(modes))


def l5_curve(name="curve", modes=(0, 1)):
    rows = stats()
    best = np.array([float(r["best_fitness"]) for r in rows])
    mean = np.array([float(r["mean_fitness"]) for r in rows])
    sig = np.array([float(r["sigma"]) for r in rows])
    ch, last = [], np.nan
    for r in rows:
        if r["best_score"]:
            last = float(r["best_score"])
        ch.append(last)
    ch = np.array(ch)
    first = int(np.argmax(~np.isnan(ch))) if (~np.isnan(ch)).any() else 0
    n = len(rows)

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, f"The whole run: {n} generations", t)
        box = (120, 140, 1200, 470)
        hi = max(320, best.max() + 10)
        axes(d, box, -60, hi, [0, 100, 200, 300])
        u = min(1, t / 5)
        m = max(2, int(n * u))
        series(d, box, mean[:m], MUTED, -60, hi, n=n, width=2)
        series(d, box, best[:m], AMBER, -60, hi, n=n, width=2)
        if m > first + 1:
            series(d, (box[0] + (box[2] - box[0]) * first / (n - 1), box[1], box[2], box[3]), ch[first:m], BLUE, -60, hi, n=n - first, width=4)
        sbox = (120, 520, 1200, 650)
        axes(d, sbox, 0, max(0.6, sig.max()), [0, 0.5], fmt="{:.1f}")
        a = beat_alpha(b, t, 1)
        series(d, sbox, sig[:m], C.mix(BG, TEAL, max(.35, a)), 0, max(0.6, sig.max()), n=n, width=3)
        text(d, (1200, 665), "step size σ", 18, TEAL, max(.35, a), anchor="ra")
        return img
    return Scene(name, render, [7] * len(modes))


def l5_outro():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 270), "flailing → standing → a step → walking", 40, FG, fade(t, .2), bold=True, anchor="mm")
        text(d, (W // 2, 350), "every link must score a little better than the one before", 26, MUTED, fade(t, 1), anchor="mm")
        text(d, (W // 2, 470), "Next: lesson 6, inside the champion", 26, AMBER, fade(t, 3), anchor="mm")
        return img
    return Scene("outro", render, [6])


def lesson5():
    return [title_card("intro", "LESSON 5", "The training run", "three false starts, and the fitness that worked", 6),
            l5_shaping(), l5_traps(), l5_breakthrough(), l5_stages(modes=list(range(nbeats("lesson-05", "stages")))),
            l5_curve(), l5_outro()]


# =============================================================================
# Lesson 6
# =============================================================================

class Probe:
    def __init__(self, mlp, hidden=None, value=0.0):
        self.mlp, self.hidden, self.value, self.h = mlp, hidden, value, np.zeros(mlp.sizes[1])

    def activate(self, x):
        (W1, b1), (W2, b2) = self.mlp.layers
        h = np.tanh(W1 @ x + b1)
        if self.hidden is not None:
            h[self.hidden] = self.value
        self.h = h
        return np.tanh(W2 @ h + b2)


def heat_strip(d, box, hist, order, contacts):
    x0, y0, x1, y1 = box
    T = len(hist)
    if T == 0:
        return
    cw = (x1 - x0) / 200
    rh = (y1 - y0 - 40) / 32
    for ti, h in enumerate(hist[-200:]):
        for r, j in enumerate(order):
            v = float(np.clip(h[j], -1, 1))
            d.rectangle([x0 + ti * cw, y0 + r * rh, x0 + (ti + 1) * cw, y0 + (r + 1) * rh], fill=C.mix(BG, BLUE if v >= 0 else ORANGE, abs(v)))
    for ti, (c1, c2) in enumerate(contacts[-200:]):
        if c1:
            d.rectangle([x0 + ti * cw, y1 - 34, x0 + (ti + 1) * cw, y1 - 22], fill=BLUE)
        if c2:
            d.rectangle([x0 + ti * cw, y1 - 16, x0 + (ti + 1) * cw, y1 - 4], fill=ORANGE)


def l6_heat(name="heat", modes=(0, 1)):
    I = J("inside.json")
    rh = np.array(I["gait"]["rhythmic"]) if I["gait"] else np.zeros(32)
    order = list(np.argsort(-rh))
    pol = Probe(champion()[0])
    p = C.Player(pol, I["trace"]["seed"])
    state = {"step": -1, "hist": [], "cont": []}

    def render(t, b):
        img = blank()
        fr = p.at_time(t, 0.5, loop_pause=0)
        if p.step < state["step"]:
            state["hist"], state["cont"] = [], []
        if p.step != state["step"]:
            state["hist"].append(pol.h.copy())
            state["cont"].append((p.x_obs[8] > .5, p.x_obs[13] > .5))
            state["step"] = p.step
        paste_frame(img, fr, (40, 40, 480, 320))
        d = ImageDraw.Draw(img)
        text(d, (560, 50), "32 hidden neurons over time", 28, FG, bold=True)
        text(d, (560, 95), "blue = positive · orange = negative · strips: foot 1 / foot 2 on the ground", 16, MUTED)
        heat_strip(d, (40, 380, 1240, 700), state["hist"], order, state["cont"])
        k, _, _ = b.local(t)
        if modes[k] == 1:
            label(d, (560, 150), f"{round(I['saturation'] * 100)}% of the time a neuron sits at ±1: a switch", 20)
        return img
    return Scene(name, render, [8] * len(modes))


def l6_gait(name="gait", modes=(0, 1)):
    g = J("inside.json")["gait"]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, f"One stride, averaged (≈ {g['stride_steps_median']:.0f} steps)", t)
        prof = np.array(g["profile"])
        order = np.argsort(-np.array(g["rhythmic"]))
        x0, y0, nb = 200, 130, prof.shape[1]
        cw, rh = 900 / nb, 380 / 32
        for r, j in enumerate(order):
            for k in range(nb):
                v = prof[j, k]
                d.rectangle([x0 + k * cw, y0 + r * rh, x0 + (k + 1) * cw - 1, y0 + (r + 1) * rh - 1], fill=C.mix(BG, BLUE if v >= 0 else ORANGE, min(1, abs(v))))
        ap = np.array(g["action_profile"])
        for i in range(4):
            text(d, (x0 - 12, 535 + i * 32), C.OUTPUT_SHORT[i], 18, FG, anchor="ra")
            for k in range(nb):
                v = ap[i, k]
                d.rectangle([x0 + k * cw, 522 + i * 32, x0 + (k + 1) * cw - 1, 548 + i * 32], fill=C.mix(BG, BLUE if v >= 0 else ORANGE, min(1, abs(v))))
        text(d, (x0, 670), "stride phase →   (0 = foot 1 touches down)", 16, MUTED)
        text(d, (x0 - 12, y0 + 190), "hidden", 16, MUTED, anchor="ra")
        a = beat_alpha(b, t, 1) if len(modes) > 1 else 0
        if a > 0:
            label(d, (700, 300), "no memory: the rhythm runs through the legs", 20)
        return img
    return Scene(name, render, [7] * len(modes))


def l6_ablation(name="ablation", modes=(0, 1, 2)):
    I = J("inside.json")
    abl = I["ablation"]["hidden"]
    top = max(abl, key=lambda x: x["drop"])
    mlp = champion()[0]
    hid_mean = np.array(I["trace"]["hid"]).mean(0)
    frozen = C.Player(Probe(mlp, top["index"], float(hid_mean[top["index"]])), I["trace"]["seed"])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        k, lt, _ = b.local(t)
        mode = modes[k]
        heading(d, "Ablation: freeze one neuron, measure the damage", t)
        if mode == 1:
            paste_frame(img, frozen.at_time(lt, 1.0, loop_pause=1.0, length=600), (40, 140, 600, 400), border=RED)
            label(d, (55, 155), f"neuron {top['index'] + 1} frozen: reward {frozen.total:.0f}", 18)
            box = (680, 160, 1240, 520)
        else:
            box = (60, 160, 1240, 560)
        x0, y0, x1, y1 = box
        mx = max(1, max(a["drop"] for a in abl))
        bw = (x1 - x0) / 32
        for a in abl:
            h = (y1 - y0) * max(0, a["drop"]) / mx
            col = AMBER if a["index"] == top["index"] or (mode == 2 and a["drop"] > 50) else BLUE
            d.rectangle([x0 + a["index"] * bw + 2, y1 - h, x0 + (a["index"] + 1) * bw - 2, y1], fill=col)
            if mode != 1 or a["index"] % 4 == 3:
                text(d, (x0 + (a["index"] + .5) * bw, y1 + 14), str(a["index"] + 1), 12, MUTED, anchor="mm")
        text(d, (x0, y0 - 24), f"score lost when frozen (base {I['ablation']['base']:.0f} on {I['ablation']['episodes']} tracks)", 16, MUTED)
        if mode == 2:
            big = sum(a["drop"] > 50 for a in abl)
            text(d, (60, 620), f"{big} of 32 neurons each cost more than 50 points; the rest is spare capacity or fine tuning", 22, FG)
        return img
    return Scene(name, render, [7] * len(modes))


def l6_inputs(name="inputs", modes=(0, 1)):
    I = J("inside.json")
    inp = I["ablation"]["inputs"]
    mx = max(1, max(a["drop"] for a in inp))

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Which senses does it use?", t)
        k, _, _ = b.local(t)
        hl_lidar = modes[k] == 1
        vals = [max(0, a["drop"]) / mx for a in inp]
        cols = [C.mix(BG, C.INPUT_COL[i], .3 if (hl_lidar and i < 14) else 1) for i in range(24)]
        x0, y0, rh = 240, 130, 22
        for i, v in enumerate(vals):
            y = y0 + i * rh
            text(d, (x0 - 10, y + rh / 2), C.INPUT_SHORT[i], 14, MUTED, anchor="rm")
            d.rectangle([x0, y + 3, x0 + max(2, v * 850), y + rh - 3], fill=cols[i])
            text(d, (x0 + max(2, v * 850) + 8, y + rh / 2), f"{v * mx:.0f}", 13, MUTED, anchor="lm")
        text(d, (x0, y0 - 20), "score lost when this input is frozen", 16, MUTED)
        if hl_lidar:
            lid = sum(vals[14:]) * mx
            body = sum(vals[:14]) * mx
            label(d, (700, 600), f"lidar total {lid:.0f} vs body + legs {body:.0f}", 20)
        return img
    return Scene(name, render, [7] * len(modes))


def l6_outro():
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 300), "a few switch-like neurons, driven by the legs' own feel", 34, FG, fade(t, .2), bold=True, anchor="mm")
        text(d, (W // 2, 420), "Next: lesson 7, the final test", 26, AMBER, fade(t, 1.5), anchor="mm")
        return img
    return Scene("outro", render, [6])


def lesson6():
    sc = [title_card("intro", "LESSON 6", "Inside the champion", "what the hidden neurons do while it walks", 6), l6_heat()]
    if nbeats("lesson-06", "gait"):
        sc.append(l6_gait())
    return sc + [l6_ablation(), l6_inputs(), l6_outro()]


# =============================================================================
# Lesson 7
# =============================================================================

def l7_final(name="final", modes=(0, 1)):
    E = J("eval_best.json")
    rw = np.array([e["reward"] for e in E["episodes_detail"]])

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "100 tracks it has never seen", t)
        for i, (v, s) in enumerate([(f"{E['mean']:.0f}", f"mean (± {E['std']:.0f})"), (str(E["falls"]), "falls"),
                                    (str(E["finished"]), "finished"), (str(E["n_ge_300"]), "runs ≥ 300")]):
            a = fade(t, .3 + i * .5)
            text(d, (110 + i * 290, 170), v, 56, AMBER if i == 0 else FG, a, bold=True)
            text(d, (110 + i * 290, 240), s, 20, MUTED, a)
        box = (110, 320, 1170, 640)
        lo, hi, nb = -150, 350, 25
        cnt = np.histogram(rw, np.linspace(lo, hi, nb + 1))[0]
        u = min(1, max(0, (t - 1.5) / 2))
        for i, c in enumerate(cnt):
            h = (box[3] - box[1]) * c / max(1, cnt.max()) * u
            bx = box[0] + (box[2] - box[0]) * i / nb
            d.rectangle([bx + 3, box[3] - h, bx + (box[2] - box[0]) / nb - 3, box[3]], fill=BLUE)
        x3 = box[0] + (box[2] - box[0]) * (300 - lo) / (hi - lo)
        d.line([(x3, box[1]), (x3, box[3])], fill=AMBER, width=2)
        text(d, (x3 + 8, box[1]), "solved", 18, AMBER)
        for v in range(-100, 301, 100):
            text(d, (box[0] + (box[2] - box[0]) * (v - lo) / (hi - lo), box[3] + 20), str(v), 16, MUTED, anchor="mm")
        return img
    return Scene(name, render, [7] * len(modes))


def l7_budget(name="budget", modes=(0, 1)):
    E = J("eval_best.json")
    eps = E["episodes_detail"]
    prog = np.mean([130 / 30 * (min(e["x"], 88.67) - 4.67) for e in eps])
    falls = 100 * E["falls"] / len(eps)
    eff = prog - falls - E["mean"]
    cp = champ_player()

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Where the points go", t)
        sc = 1000 / 380
        u = ease_in(t / 3)
        d.rectangle([120, 180, 120 + prog * sc * u, 250], fill=GREEN)
        text(d, (130, 215), f"earned by moving  +{prog:.0f}", 24, BG, anchor="lm", bold=True, bg=GREEN)
        a = fade(t, 2.5)
        if a > 0:
            d.rectangle([120 + (prog - eff - falls) * sc, 280, 120 + (prog - falls) * sc, 350], fill=C.mix(BG, ORANGE, a))
            text(d, (130 + (prog - eff - falls) * sc, 315), f"effort  −{eff:.0f}", 24, BG, a, anchor="lm", bold=True, bg=ORANGE)
            text(d, (120, 385), f"score ≈ {E['mean']:.0f}", 30, AMBER, a, bold=True)
        k, lt, _ = b.local(t)
        if modes[k] == 1:
            paste_frame(img, cp.at_time(lt, 1.0, start=300, loop_pause=0, length=800), (120, 440, 360, 240))
            bars(d, (540, 450, 1100, 670), cp.act, C.OUTPUT_SHORT, [BLUE, BLUE, ORANGE, ORANGE], size=18, label_w=100)
            text(d, (540, 690), "motor commands: mostly near ±1", 18, MUTED)
        return img
    return Scene(name, render, [7] * len(modes))


def ease_in(x):
    return C.ease(x)


def l7_neat(name="neat", modes=(0, 1)):
    P = J("compare_neat.json")

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Against NEAT, on the same 100 tracks", t)
        x0, y0, S = 120, 140, 500
        X = lambda v: x0 + (v + 150) / 500 * S
        Y = lambda v: y0 + S - (v + 150) / 500 * S
        d.rectangle([x0, y0, x0 + S, y0 + S], fill=PANEL)
        d.line([(X(-150), Y(-150)), (X(350), Y(350))], fill=MUTED, width=1)
        n = int(min(len(P["pairs"]), 5 + t * 30))
        for s, a, bb in P["pairs"][:n]:
            d.ellipse([X(a) - 5, Y(bb) - 5, X(a) + 5, Y(bb) + 5], fill=AMBER if a >= bb else BLUE)
        text(d, (x0 + S, y0 + S + 25), "CMA-ES →", 18, MUTED, anchor="ra")
        text(d, (x0 - 10, y0), "NEAT ↑", 18, MUTED, anchor="ra")
        k, _, _ = b.local(t)
        fl = lambda n: f"{n} fall" + ("" if n == 1 else "s")
        rows = [("CMA-ES", f"{P['mean_a']:.0f}", fl(P['falls_a'])), ("NEAT", f"{P['mean_b']:.0f}", fl(P['falls_b']))]
        for i, (nm, m, f) in enumerate(rows):
            text(d, (720, 180 + i * 90), nm, 30, AMBER if i == 0 else BLUE, bold=True)
            text(d, (920, 180 + i * 90), m, 40, FG, bold=True)
            text(d, (1040, 190 + i * 90), f, 24, MUTED)
        text(d, (720, 380), f"difference {P['diff_mean']:+.0f}  (95% CI {P['diff_ci95'][0]:+.0f} to {P['diff_ci95'][1]:+.0f})", 22, FG)
        text(d, (720, 420), f"CMA-ES higher on {P['a_better_on']} of {P['episodes']} tracks", 22, FG)
        return img
    return Scene(name, render, [7] * len(modes))


def l7_table(name="table", modes=(0, 1)):
    G = J("progress.json")
    E = J("eval_best.json")
    rows = [("NEAT", "173–179", "30 conn.", "414 M", "~12 h"), ("PPO", "272", "–", "8 M", "~8–10 h"),
            ("TD3", "308–319", "131,504 + critics", "~2 M", "~2 h"),
            ("CMA-ES", f"{E['mean']:.0f}", "932", f"{G['env_steps'] / 1e6:.0f} M", f"~{G['hours']:.1f} h")]

    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        heading(d, "Where CMA-ES sits", t)
        hdr = ["method", "score", "policy size", "env steps", "laptop time"]
        xs = [80, 300, 520, 860, 1060]
        for x, h in zip(xs, hdr):
            text(d, (x, 160), h, 20, MUTED, bold=True)
        for i, r in enumerate(rows):
            a = fade(t, .4 + i * .6)
            for x, v in zip(xs, r):
                text(d, (x, 220 + i * 80), v, 28, AMBER if i == 3 else FG, a, bold=i == 3)
        a = beat_alpha(b, t, 1) if len(modes) > 1 else 0
        paragraph(d, (80, 560), "gradients use far fewer samples; evolution needs no critic, no replay buffer, and finds a policy ~140× smaller", 80, 22, GREEN, a)
        return img
    return Scene(name, render, [7] * len(modes))


def l7_outro(name="outro"):
    cp = champ_player()

    def render(t, b):
        img = blank()
        cover_frame(img, cp.at_time(t, 1.0, start=200, loop_pause=0), dim=0.55)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 300), "932 numbers · rankings of whole episodes · a fitness that leads", 34, FG, fade(t, .3), bold=True, anchor="mm")
        text(d, (W // 2, 380), "jca374.github.io/bipedal-cmaes", 26, AMBER, fade(t, 2), anchor="mm")
        return img
    return Scene(name, render, [8])


def lesson7():
    return [title_card("intro", "LESSON 7", "Results and comparison", "the honest test", 6),
            l7_final(), l7_budget(), l7_neat(), l7_table(), l7_outro()]


# =============================================================================
# Explainer
# =============================================================================

def ex_title():
    cp = champ_player()

    def render(t, b):
        img = blank()
        cover_frame(img, cp.at_time(t, 1.0, start=100, loop_pause=0), dim=0.35)
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 270), "Teaching a network to walk", 54, FG, fade(t, .3), bold=True, anchor="mm")
        text(d, (W // 2, 340), "without gradients", 54, AMBER, fade(t, .8), bold=True, anchor="mm")
        text(d, (W // 2, 420), "CMA-ES · 932 weights · a 4-core laptop", 24, FG, fade(t, 1.5), anchor="mm")
        return img
    return Scene("title", render, [8])


def explainer():
    n_st = nbeats("explainer", "stages")
    return [ex_title(),
            remap(LS.l1_reward(), "task", [0, 3]),
            remap(LS.l2_network(), "network", [1]),
            remap(LS.l3_compare(), "loop", [0, 1]),
            remap(LS.l4_crn(), "noise", [1]),
            l5_traps("traps", (0, 3)),
            l5_breakthrough("breakthrough", (1,)),
            l5_stages("stages", list(range(n_st))),
            l6_heat("heat", (1,)),
            l6_ablation("ablation", (1,)),
            l6_inputs("inputs", (1,)),
            l7_final("final", (0, 1)),
            l7_budget("budget", (1,)),
            l7_neat("neat", (0,)),
            l7_outro("outro")]


VIDEOS = {"05-the-training-run": ("lesson-05", lesson5), "06-inside-the-champion": ("lesson-06", lesson6),
          "07-results": ("lesson-07", lesson7), "explainer": ("explainer", explainer)}
