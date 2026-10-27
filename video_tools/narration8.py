"""Narration for lesson 8 (two ways to walk) from docs/data/gait.json, adapting to the outcome.

  python video_tools/narration8.py      # -> video_tools/narration/lesson-08.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from narration import DATA, OUT, sec, words  # noqa: E402


def main():
    G = json.load(open(os.path.join(DATA, "gait.json")))
    W = {w["run"]: w for w in G["walkers"]}
    v3, v4, C = W["v3-f20"], W.get("v4-walk"), G.get("compare")
    s = ["# Lesson 9: Two ways to walk\n",
         sec("intro", "Lesson nine. The champion scores well, but it never puts one foot in front of the other. Let's look at why, and test a network built to take turns."),
         sec("skip", "Watch the feet. The line below shows which foot is in front. Above the line, foot one leads. Below it, foot two leads. A normal walk crosses the line on every step."),
         sec("skip.2", f"This walker never crosses it. On a hundred tracks, foot two leads {words(round(v3['front2_share'] * 100))} percent of the time. Its touchdowns still alternate, so it isn't hopping. It's skipping, with one leg always reaching forward."),
         sec("why", "Why did evolution keep this gait? The reward pays for distance, not for style, and the skip earns well. To switch, the walker would have to pass through clumsy in between gaits that score worse. CMA-ES searches nearby, so it stays where it is."),
         sec("fix", "So instead of asking nicely, change the network. One controller now drives both legs. Each leg sees the body, its own leg, the other leg and the lidar. Leg two uses the same weights, with the legs swapped. Neither leg can become a permanent prop."),
         sec("fix.2", "The controller also gets a clock, a rhythm whose speed it learns. Leg two's clock runs half a cycle behind. With no shift, both legs would move together, which is a hop. With half a cycle, they take turns."),
         sec("fix.3", "On top, a small training bonus for every step where the front foot changes and the body moves forward. Reported scores stay the raw reward."),
         ]
    if v4:
        walks = v4["crossings"] > 10 * max(1, v3["crossings"]) and 0.2 < v4["front2_share"] < 0.8
        s.append(sec("results", (f"The result. The new walker changes its leading foot about {words(v4['lead_changes'])} times per run, and foot two leads {words(round(v4['front2_share'] * 100))} percent of the time. It really walks."
                                 if walks else f"The result. The new walker changes its leading foot only about {words(v4['lead_changes'])} times per run. In this run, it didn't clearly learn to walk.")))
        if C:
            d = C["diff_mean"]
            sig = "significantly " if (C["diff_ci95"][0] > 0 or C["diff_ci95"][1] < 0) else ""
            verdict = ("higher" if d > 0 else "lower") if sig else "about the same"
            s.append(sec("results.2", f"On the same hundred tracks, it scores {words(v4['mean'])}, against {words(v3['mean'])} for the skipping walker. That's {sig}{verdict}"
                                      + ("." if not sig else f", by {words(abs(d))} points.") + f" It falls on {words(v4['falls'])} tracks, against {words(v3['falls'])}."))
        s.append(sec("outro", "The lesson: evolution finds what you reward, the nearest way. If you care how it's done, build it into the structure."))
    else:
        s.append(sec("outro", "The new walker is still training. Its results appear on this page when it's done."))
    open(os.path.join(OUT, "lesson-08.md"), "w").write("\n".join(s))
    print(open(os.path.join(OUT, "lesson-08.md")).read())


if __name__ == "__main__":
    main()
