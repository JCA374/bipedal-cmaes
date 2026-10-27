"""Shared pieces for the lesson clips and the explainer: Pillow drawing helpers (same look
as ../bipedal-neat/video_tools), and an episode Player that re-simulates instead of storing
frames (1600 frames would be >1 GB per episode on this 7.6 GB laptop).
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
import common  # noqa: E402
from timeline import Scene  # noqa: E402,F401

W, H, FPS = 1280, 720, 30
BG = (11, 16, 32)
PANEL = (22, 30, 54)
FG = (232, 236, 245)
MUTED = (140, 152, 180)
AMBER = (255, 184, 64)
BLUE = (96, 168, 255)
GREEN = (84, 212, 140)
RED = (240, 96, 96)
ORANGE = (235, 104, 52)
TEAL = (40, 190, 140)
DATA = os.path.join(REPO_ROOT, "video_tools", "data")

INPUT_SHORT = ["hull angle", "ang. vel", "vel x", "vel y",
               "hip1 ang", "hip1 spd", "knee1 ang", "knee1 spd", "foot1",
               "hip2 ang", "hip2 spd", "knee2 ang", "knee2 spd", "foot2"] + [f"lidar {i}" for i in range(10)]
OUTPUT_SHORT = ["hip 1", "knee 1", "hip 2", "knee 2"]
INPUT_COL = [AMBER] * 4 + [BLUE] * 5 + [ORANGE] * 5 + [TEAL] * 10

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
_fonts = {}


def font(size, bold=False, mono=False):
    key = (size, bold, mono)
    if key not in _fonts:
        name = "DejaVuSansMono" if mono else "DejaVuSans"
        if bold:
            name += "-Bold"
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)
    return _fonts[key]


def mix(c1, c2, a):
    a = max(0.0, min(1.0, a))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * a) for i in range(3))


def ease(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fade(t, start, dur=0.4):
    return ease((t - start) / dur)


def blank():
    return Image.new("RGB", (W, H), BG)


def text(d, xy, s, size=24, fill=FG, a=1.0, bold=False, mono=False, anchor="la", bg=BG):
    if a <= 0:
        return
    d.text(xy, s, font=font(size, bold, mono), fill=mix(bg, fill, a), anchor=anchor)


def heading(d, title, t, sub=None):
    a = fade(t, 0)
    text(d, (60, 42), title, 38, FG, a, bold=True)
    d.rectangle([60, 92, 60 + int(120 * a), 96], fill=AMBER)
    if sub:
        text(d, (200, 84), sub, 18, MUTED, a)


def wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + [cur] if cur else lines


def paragraph(d, xy, s, width=40, size=22, fill=FG, a=1.0, gap=1.45):
    x, y = xy
    for i, line in enumerate(wrap(s, width)):
        text(d, (x, y + i * size * gap), line, size, fill, a)


def bullet(d, xy, s, a, size=24, fill=FG, width=44):
    x, y = xy
    if a <= 0:
        return
    d.ellipse([x, y + size * .35, x + 9, y + size * .35 + 9], fill=mix(BG, AMBER, a))
    paragraph(d, (x + 24, y), s, width, size, fill, a)


def paste_frame(img, frame, box, border=None, dim=0.0):
    x, y, w, h = box
    tile = Image.fromarray(frame).resize((w, h), Image.BILINEAR)
    if dim > 0:
        tile = Image.blend(tile, Image.new("RGB", (w, h), BG), dim)
    img.paste(tile, (x, y))
    if border:
        ImageDraw.Draw(img).rectangle([x - 3, y - 3, x + w + 2, y + h + 2], outline=border, width=3)


def cover_frame(img, frame, dim=0.0):
    """Fill the whole 16:9 canvas with a 3:2 game frame (scaled up, cropped top/bottom)."""
    tile = Image.fromarray(frame).resize((W, int(W * frame.shape[0] / frame.shape[1])), Image.BILINEAR)
    top = (tile.height - H) // 2 + 40
    tile = tile.crop((0, top, W, top + H))
    if dim > 0:
        tile = Image.blend(tile, Image.new("RGB", (W, H), BG), dim)
    img.paste(tile, (0, 0))


def label(d, xy, s, size=17, anchor="la"):
    """Readable label on top of a game frame: light text on a dark box."""
    f = font(size, True)
    bb = d.textbbox(xy, s, font=f, anchor=anchor)
    d.rectangle([bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4], fill=(20, 24, 40))
    d.text(xy, s, font=f, fill=FG, anchor=anchor)


def title_card(name, kicker, title, sub, dur=5.0):
    def render(t, b):
        img = blank()
        d = ImageDraw.Draw(img)
        text(d, (W // 2, 250), kicker, 26, AMBER, fade(t, 0.1), bold=True, anchor="mm")
        text(d, (W // 2, 330), title, 56, FG, fade(t, 0.3, 0.6), bold=True, anchor="mm")
        u = fade(t, 0.5)
        d.rectangle([W // 2 - int(80 * u), 380, W // 2 + int(80 * u), 384], fill=AMBER)
        text(d, (W // 2, 430), sub, 24, MUTED, fade(t, 0.7, 0.6), anchor="mm")
        return img
    return Scene(name, render, [dur])


def bars(d, box, values, labels, colors, a=1.0, size=13, label_w=96):
    """Vertical list of centred bars for values in [-1, 1]."""
    x0, y0, x1, y1 = box
    n = len(values)
    rh = (y1 - y0) / n
    mid = x0 + label_w + (x1 - x0 - label_w) / 2
    half = (x1 - x0 - label_w) / 2 - 4
    d.line([(mid, y0), (mid, y1)], fill=mix(BG, MUTED, .5 * a), width=1)
    for i, v in enumerate(values):
        v = float(np.clip(v, -1, 1))
        y = y0 + i * rh
        c = colors[i] if isinstance(colors, (list, tuple)) and isinstance(colors[0], tuple) else colors
        d.rectangle([min(mid, mid + v * half), y + 2, max(mid, mid + v * half), y + rh - 2], fill=mix(BG, c, a))
        if labels:
            text(d, (x0 + label_w - 8, y + rh / 2), labels[i], size, MUTED, a, anchor="rm")


def series(d, box, ys, color, lo, hi, a=1.0, width=3, n=None):
    x0, y0, x1, y1 = box
    n = n or len(ys)
    if len(ys) < 2:
        return
    pts = [(x0 + (x1 - x0) * i / max(1, n - 1), y1 - (y1 - y0) * (float(v) - lo) / (hi - lo)) for i, v in enumerate(ys)]
    d.line(pts, fill=mix(BG, color, a), width=width, joint="curve")


def axes(d, box, lo, hi, ticks, a=1.0, xlabel=None, fmt="{:.0f}"):
    x0, y0, x1, y1 = box
    for v in ticks:
        y = y1 - (y1 - y0) * (v - lo) / (hi - lo)
        d.line([(x0, y), (x1, y)], fill=mix(BG, PANEL, a), width=1)
        text(d, (x0 - 8, y), fmt.format(v), 14, MUTED, a, anchor="rm")
    if xlabel:
        text(d, (x1, y1 + 18), xlabel, 14, MUTED, a, anchor="rm")


# ----------------------------------------------------------------------------
# Episodes
# ----------------------------------------------------------------------------

class ActionPolicy:
    """Replays recorded actions (exact, because common.fresh_reset makes episodes deterministic)."""

    def __init__(self, actions):
        self.actions, self.i = np.asarray(actions), 0

    def reset(self):
        self.i = 0

    def activate(self, x):
        a = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return a


def load_actions(name):
    return ActionPolicy(np.load(os.path.join(DATA, f"{name}_actions.npy")))


class Player:
    """Frames of one deterministic episode, re-simulated on demand. Also keeps the current
    scaled observation, action, running reward and hull x for overlays."""

    def __init__(self, policy, seed, max_steps=1600):
        self.policy, self.seed, self.max_steps = policy, seed, max_steps
        self.env = common.make_env(render_mode="rgb_array")
        self._restart()

    def _restart(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.obs, _ = common.fresh_reset(self.env, self.seed)
        self.step, self.done, self.total = 0, False, 0.0
        self.x_obs = common.scale_obs(self.obs)
        self.act = np.zeros(4)
        self.rewards = [0.0]
        self.cur = self.env.render()

    def frame(self, step):
        step = int(step)
        if step < self.step:
            self._restart()
        while self.step < step and not self.done and self.step < self.max_steps:
            self.x_obs = common.scale_obs(self.obs)
            self.act = np.asarray(self.policy.activate(self.x_obs))
            self.obs, r, term, trunc, _ = self.env.step(self.act.astype(np.float32))
            self.total += float(r)
            self.rewards.append(self.total)
            self.step += 1
            self.done = term or trunc
            if self.step >= step or self.done:
                self.cur = self.env.render()
        return self.cur

    @property
    def x(self):
        return float(self.env.unwrapped.hull.position.x)

    def at_time(self, t, speed=1.0, start=0, loop_pause=1.5, length=None):
        """Frame for scene time t at `speed` x real time (50 steps/s), looping."""
        n = length or self.max_steps
        k = int(t * 50 * speed) % (n + int(loop_pause * 50 * speed))
        return self.frame(min(start + k, start + n))
