#!/usr/bin/env python3
"""Why the walker doesn't reach 300: a score analysis in the failure-analysis-report style.

This walker rarely falls, so the "failure" studied here is a low score: time-outs and effort.
Uses record_episodes.py output (raw obs, actions, hull state per step).

  python experiments/analyze_score.py results/episodes/final.jsonl.gz

Writes results/score_analysis/<model>/report.html + PNGs + summary.json. Questions:
  1. Where do the points go?   progress vs effort vs tilt vs falls, per episode, sorted by score
  2. Who finishes in time?     hull x over time, finished vs timed out
  3. What predicts a low score? per-episode signals: AUC for "times out" + correlation with score
  4. Where is effort wasted?   torque on joints that aren't moving (pushing against itself / a limit)
"""
import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analyze_falls import GRID, INK, INK_2, SERIES, SURFACE, auc, save  # noqa: E402  (also sets rcParams)

FINISH = (200 - 10) * 14 / 30          # hull x at the finish line
START = 14 / 30 * 20 / 2
JOINTS = ["hip 1", "knee 1", "hip 2", "knee 2"]
SPEED_OBS = [5, 7, 10, 12]             # raw obs index of each joint's speed (already / max speed)
STILL = 0.05                            # |joint speed| below this (5% of max) counts as "not moving"


def load(path):
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())["header"]
        eps = [json.loads(line) for line in f]
    return header, eps


def per_episode(ep, cols):
    T = np.array(ep["trace"], dtype=float)
    c = {n: T[:, i] for i, n in enumerate(cols)}
    a = np.clip(np.abs(T[:, 24:28]), 0, 1)
    x = c["x"]
    progress = 130 / 30 * (min(x[-1] if not ep["fell"] else ep["final_x"], FINISH) - START)
    effort = 0.00035 * 80 * a.sum()
    spd = np.abs(T[:, SPEED_OBS])
    still = spd < STILL
    wasted = 0.00035 * 80 * (a * still).sum()
    slope = c["slope_under"]
    contact = T[:, [8, 13]] > 0.5
    return {
        "seed": ep["seed"], "reward": ep["reward"], "fell": ep["fell"], "steps": ep["steps"],
        "finished": (not ep["fell"]) and ep["steps"] < 1600, "timeout": (not ep["fell"]) and ep["steps"] >= 1600,
        "final_x": ep["final_x"], "progress": progress, "effort": effort, "wasted": wasted,
        "tilt_etc": progress - effort - (100 if ep["fell"] else 0) - ep["reward"],
        "x_curve": x[::20].tolist(),
        "signals": {
            "mean speed (units/s)": (x[-1] - x[0]) / len(x) * 50,
            "mean |action|": float(a.mean()),
            "share of effort wasted": wasted / max(effort, 1e-9),
            "terrain roughness (std slope)": float(np.std(slope)),
            "uphill share (slope > 0.05)": float(np.mean(slope > 0.05)),
            "hull angle variability": float(np.std(c["angle"])),
            "airborne share (no foot down)": float(np.mean(~contact.any(1))),
            "double support share": float(np.mean(contact.all(1))),
        },
        "joint_effort": (0.00035 * 80 * a.sum(0)).tolist(),
        "joint_wasted": (0.00035 * 80 * (a * still).sum(0)).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    header, raw = load(args.episodes)
    model = header["model"]
    out_dir = args.out or os.path.join("results", "score_analysis", model)
    os.makedirs(out_dir, exist_ok=True)
    E = [per_episode(e, header["columns"]) for e in raw]
    E.sort(key=lambda e: e["reward"])
    rw = np.array([e["reward"] for e in E])
    n_fin, n_to, n_fell = (sum(e[k] for e in E) for k in ("finished", "timeout", "fell"))
    sections = []

    # 1. where do the points go
    fig, ax = plt.subplots(figsize=(11, 3.6))
    idx = np.arange(len(E))
    pr, ef, ti = (np.array([e[k] for e in E]) for k in ("progress", "effort", "tilt_etc"))
    fl = np.array([100.0 if e["fell"] else 0.0 for e in E])
    ax.bar(idx, pr, color=SERIES[2], width=1.0, label="progress earned")
    ax.bar(idx, -ef, color=SERIES[1], width=1.0, label="motor effort")
    ax.bar(idx, -fl, bottom=-ef, color=SERIES[7], width=1.0, label="fall")
    ax.plot(idx, rw, color=INK, lw=1.5, label="score")
    ax.axhline(300, color=SERIES[3], ls="--", lw=1)
    ax.set_xlabel("episodes, sorted by score")
    ax.set_ylabel("points")
    ax.legend(ncol=4, loc="lower right", frameon=False)
    ax.set_title("Where do the points go?")
    save(fig, out_dir, "budget.png")
    sections.append(("1. Where do the points go?", "budget.png",
                     f"Each column is one episode: green = points earned by moving forward, orange = spent on motor effort, red = fall penalty; the black line is the score. "
                     f"On average it earns {pr.mean():.0f} and spends {ef.mean():.0f} on effort. The dashed line is 300."))

    # 2. who finishes in time
    fig, ax = plt.subplots(figsize=(11, 3.6))
    for e in E:
        col = SERIES[0] if e["finished"] else (SERIES[7] if e["fell"] else SERIES[1])
        ax.plot(np.arange(len(e["x_curve"])) * 20, e["x_curve"], color=col, lw=0.8, alpha=0.5)
    ax.axhline(FINISH, color=SERIES[3], ls="--", lw=1)
    ax.plot([0, 1600], [START, FINISH], color=INK_2, lw=1, ls=":")
    ax.set_xlabel("step")
    ax.set_ylabel("hull x")
    ax.set_title(f"Who finishes in time? blue = finished ({n_fin}), orange = timed out ({n_to}), red = fell ({n_fell})")
    save(fig, out_dir, "progress.png")
    to_x = [e["final_x"] for e in E if e["timeout"]]
    sections.append(("2. Who finishes in time?", "progress.png",
                     f"Hull position over each episode. The dotted line is the pace needed to finish exactly at step 1600; the dashed line is the finish. "
                     + (f"Time-outs end at x ≈ {np.mean(to_x):.1f} on average, {FINISH - np.mean(to_x):.1f} short of the line." if to_x else "No time-outs.")))

    # 3. what predicts a low score
    names = list(E[0]["signals"])
    S = np.array([[e["signals"][k] for k in names] for e in E])
    tout = np.array([e["timeout"] for e in E])
    aucs = [float(auc(S[:, j], tout)) if 0 < tout.sum() < len(tout) else np.nan for j in range(len(names))]
    cors = [float(np.corrcoef(S[:, j], rw)[0, 1]) if np.std(S[:, j]) > 0 else 0.0 for j in range(len(names))]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    order = np.argsort([-(abs(a - .5) if not np.isnan(a) else 0) for a in aucs])
    axs[0].barh([names[j] for j in order][::-1], [0.5 if np.isnan(aucs[j]) else aucs[j] for j in order][::-1], color=SERIES[0])
    axs[0].axvline(0.5, color=INK_2, lw=1, ls="--")
    axs[0].set_xlim(0, 1)
    axs[0].set_title("AUC for 'times out' (0.5 = no signal)")
    axs[1].barh([names[j] for j in order][::-1], [cors[j] for j in order][::-1], color=SERIES[1])
    axs[1].axvline(0, color=INK_2, lw=1)
    axs[1].set_xlim(-1, 1)
    axs[1].set_yticklabels([])
    axs[1].set_title("correlation with the score")
    save(fig, out_dir, "predict.png")
    sections.append(("3. What predicts a low score?", "predict.png",
                     "Per-episode signals. Left: how well each one separates time-outs from finishes (AUC; below 0.5 means low values go with time-outs). "
                     "Right: its correlation with the score across all episodes."))

    # 4. where is effort wasted
    je = np.array([e["joint_effort"] for e in E]).mean(0)
    jw = np.array([e["joint_wasted"] for e in E]).mean(0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(JOINTS, je, color=SERIES[1], label="effort spent")
    ax.bar(JOINTS, jw, color=SERIES[7], label=f"spent while the joint isn't moving (<{STILL:.0%} of max speed)")
    for i in range(4):
        ax.text(i, je[i] + 1, f"{je[i]:.0f} ({jw[i] / je[i]:.0%} wasted)", ha="center", fontsize=8, color=INK_2)
    ax.set_ylabel("points per episode")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Where is effort wasted?")
    save(fig, out_dir, "effort.png")
    sections.append(("4. Where is effort wasted?", "effort.png",
                     f"Average effort cost per joint, and the part spent while that joint is barely moving: torque that pushes against a joint limit or the ground "
                     f"without producing motion. In total {jw.sum():.0f} of {je.sum():.0f} points ({jw.sum() / je.sum():.0%}) go to joints that aren't moving."))

    summary = {"model": model, "episodes": len(E), "mean_reward": float(rw.mean()), "finished": int(n_fin),
               "timeouts": int(n_to), "falls": int(n_fell), "progress_mean": float(pr.mean()),
               "effort_mean": float(ef.mean()), "wasted_mean": float(jw.sum()), "wasted_share": float(jw.sum() / je.sum()),
               "joint_effort": dict(zip(JOINTS, je.round(2).tolist())), "joint_wasted": dict(zip(JOINTS, jw.round(2).tolist())),
               "timeout_final_x": float(np.mean(to_x)) if to_x else None,
               "signals": {k: {"auc_timeout": None if np.isnan(aucs[j]) else round(aucs[j], 3),
                               "corr_score": round(cors[j], 3),
                               "median": round(float(np.median(S[:, j])), 4)} for j, k in enumerate(names)}}
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    rows = "".join(f"<tr><td>{k}</td><td>{v['median']}</td><td>{v['auc_timeout']}</td><td>{v['corr_score']}</td></tr>"
                   for k, v in summary["signals"].items())
    secs = "".join(f'<section><h2>{t}</h2><p>{d}</p><img src="{img}" alt="{t}"></section>' for t, img, d in sections)
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Score analysis: {model}</title>
<style>body{{font-family:system-ui,sans-serif;background:{SURFACE};color:{INK};max-width:1200px;margin:24px auto;padding:0 16px;line-height:1.5}}
.sub{{color:{INK_2}}} img{{max-width:100%;border:1px solid {GRID};border-radius:6px}} table{{border-collapse:collapse;font-size:14px}}
td,th{{padding:4px 10px;border-bottom:1px solid {GRID};text-align:right}} td:first-child,th:first-child{{text-align:left}} section{{margin:28px 0}} a{{color:{SERIES[0]}}}</style></head><body>
<p class="sub"><a href="../../index.html">← Bipedal CMA-ES</a> · <a href="../falls/report.html">fall analysis</a></p>
<h1>Score analysis: {model}</h1>
<p class="sub">Why the walker stays below 300. {len(E)} analysis tracks (seeds {header['seed0']}+, never used in training or selection), full episodes, raw reward.
Mean {rw.mean():.1f}; finished {n_fin}, timed out {n_to}, fell {n_fell}. Effort cost = 0.028 × |action| per joint per step.</p>
<table><tr><th>signal (per episode)</th><th>median</th><th>AUC for time-out</th><th>corr. with score</th></tr>{rows}</table>
{secs}</body></html>"""
    open(os.path.join(out_dir, "report.html"), "w").write(html)
    print(f"{len(E)} episodes: finished {n_fin}, timed out {n_to}, fell {n_fell}; effort {ef.mean():.0f}, wasted {jw.sum():.0f} ({jw.sum() / je.sum():.0%})")
    print(f"Wrote {out_dir}/report.html")


if __name__ == "__main__":
    main()
