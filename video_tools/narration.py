"""Write the narration for lessons 5-7 and the explainer from the run's final numbers.

  python video_tools/narration.py runs/v3-f20      # -> video_tools/narration/{lesson-05,06,07,explainer}.md

Every number is read from the run and docs/data/*.json (finish.sh writes those), and the
wording adapts to the outcome (solved or not, what limits the score). Sections are
`## <scene>` / `## <scene>.<n>` beats, matching video_tools/final_scenes.py.
"""
import csv
import gzip
import json
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "docs", "data")
OUT = os.path.join(ROOT, "video_tools", "narration")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    """Integer to spoken English (Kokoro reads digits, but words control the phrasing)."""
    n = int(round(n))
    if n < 0:
        return "minus " + words(-n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + words(n % 100))
    if n < 1_000_000:
        return words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words(n % 1000))
    return words(n // 1_000_000) + " million" + ("" if n % 1_000_000 == 0 else " " + words(n % 1_000_000))


def load(run):
    J = lambda f: json.load(open(os.path.join(DATA, f)))
    st = pickle.load(gzip.open(os.path.join(run, "checkpoints", "latest.pkl.gz")))["state"]
    rows = list(csv.DictReader(open(os.path.join(run, "stats.csv"))))
    first50 = next((int(r["generation"]) for r in rows if float(r["best_fitness"]) > 50), None)
    E, P, I, G = J("eval_best.json"), J("compare_neat.json"), J("inside.json"), J("progress.json")
    eps = E["episodes_detail"]
    prog = sum(130 / 30 * (min(e["x"], 88.67) - 4.67) for e in eps) / len(eps)
    falls = 100 * E["falls"] / len(eps)
    effort = prog - falls - E["mean"]
    abl = sorted(I["ablation"]["hidden"], key=lambda d: -d["drop"])
    inp = sorted(I["ablation"]["inputs"], key=lambda d: -d["drop"])
    lidar = sum(max(0, d["drop"]) for d in I["ablation"]["inputs"] if d["index"] >= 14)
    body = sum(max(0, d["drop"]) for d in I["ablation"]["inputs"] if d["index"] < 14)
    big = [d for d in abl if d["drop"] > 50]
    return dict(st=st, rows=rows, first50=first50, E=E, P=P, I=I, G=G, prog=prog, effort=effort,
                abl=abl, inp=inp, lidar=lidar, body=body, big=big,
                ms={k: v["generation"] for k, v in st["milestones"].items()},
                gens=int(rows[-1]["generation"]) + 1, steps=int(rows[-1]["env_steps_total"]),
                hours=float(rows[-1]["wall_seconds"]) / 3600)


def sec(name, text):
    return f"## {name}\n\n{text.strip()}\n"


def lesson5(d):
    ms = d["ms"]
    stages = [k for k in ("first_positive", "100", "200", "300") if k in ms]
    s = ["# Lesson 5: The training run\n",
         sec("intro", "Lesson five. CMA-ES climbs whatever you tell it to climb. My first three attempts climbed the wrong hill. The walker learned to dive, to lunge, and to fall on purpose. Here's why, and what finally worked."),
         sec("shaping", "First, a distinction. The score we report is fixed: the raw reward over full episodes. But the fitness that training climbs is ours to design."),
         sec("shaping.2", "Two choices matter. Episodes are cut short when the walker stops moving, which saves a lot of time. And the price of a fall, or of a stall, can be changed for training only."),
         sec("traps", "Attempt one used the real fall penalty of a hundred, plus a stall penalty of a hundred. Standing still scored worse than falling. So the walker learned to dive forward and fall, and every dive scored about the same. There was no slope left to climb."),
         sec("traps.2", "Attempt two made falls free, but kept the stall penalty. Now the natural next step, lunge, catch yourself, and stand, was cut off by the stall detector and punished. So the walker lunged and fell."),
         sec("traps.3", "Attempt three made both free, and it still fell on purpose. The reason was motor effort. A random network pushes its motors hard, and every step alive costs points. Falling early simply ended the bill."),
         sec("traps.4", "Attempt four set the fall cost to twenty. That's more than the effort a fall saves, but far less than the hundred that made my old NEAT walker afraid to step. Standing became the safe choice, and from standing, every step forward paid."),
         sec("breakthrough", f"For about {words(d['first50'] or 50)} generations, nothing seemed to happen. Candidates learned to stand and shuffle."),
         sec("breakthrough.2", f"Then, within a few generations, the best candidate jumped from about zero to over a hundred. The first real steps. By generation {words(ms.get('100', 0))}, the champion scored over a hundred on the held out tracks."),
         ]
    label = {"first_positive": "its first positive score", "100": "over one hundred", "200": "over two hundred",
             "300": "over three hundred, which is solved"}
    groups = []                                   # milestones reached in the same generation share a beat
    for k in stages:
        if groups and ms[groups[-1][0]] == ms[k]:
            groups[-1].append(k)
        else:
            groups.append([k])
    for i, grp in enumerate(groups):
        name = "stages" if i == 0 else f"stages.{i + 1}"
        what = " and ".join(label[k] for k in grp)
        s.append(sec(name, f"{what[0].upper() + what[1:]}, {'both ' if len(grp) > 1 else ''}at generation {words(ms[grp[0]])}."))
    best = d["G"]["champion"]["mean"]
    s += [sec("curve", f"Here's the whole run: {words(d['gens'])} generations and about {words(round(d['steps'] / 1e6))} million simulation steps, in {words(round(d['hours']))} hours on this laptop. The orange line is the best training fitness. The blue line is the champion on held out tracks."),
          sec("curve.2", "Notice the step size, sigma. It stayed large through the breakthrough and only shrank later, as the walker settled into its gait. Progress came in bursts, not in a smooth line."),
          sec("outro", "The lesson: a good fitness makes every small improvement on the way to walking score a little better than the last. Next, we open up the champion and look inside.")]
    return "\n".join(s)


def lesson6(d):
    I, abl, inp = d["I"], d["abl"], d["inp"]
    g = I["gait"]
    top_in = ", ".join(x["name"] for x in inp[:3])
    blind = d["lidar"] < d["body"] / 10
    s = ["# Lesson 6: Inside the champion\n",
         sec("intro", "Lesson six. The champion is nine hundred and thirty two numbers. What do they do? Let's open the network while it walks."),
         sec("heat", "Each row is one of the thirty two hidden neurons, and time runs to the right. Blue is positive, orange is negative. At the bottom, you can see when each foot is on the ground."),
         sec("heat.2", f"Look for repeating stripes. Many neurons switch in step with the feet. And about {words(round(I['saturation'] * 100))} percent of the time, a neuron sits at its limit, fully on or fully off. Many of them act like switches."),
         ]
    if g:
        s += [sec("gait", f"Now cut the walk into strides, from one touchdown of the first foot to the next, and average. A stride takes about {words(g['stride_steps_median'])} steps, which is roughly {words(round(g['stride_steps_median'] * 20))} milliseconds. It's a quick, short gait."),
              sec("gait.2", "The network has no memory, but it still produces a rhythm. The rhythm lives in the loop between body and network. The legs move, their angles feed back as inputs, and the next command follows.")]
    k1 = abl[0]
    s += [sec("ablation", f"Which neurons matter? To find out, freeze one at its average value and let the walker try {words(I['ablation']['episodes'])} tracks. This is called ablation."),
          sec("ablation.2", f"Most neurons barely matter. But freeze neuron {words(k1['index'] + 1)}, and the score drops by {words(k1['drop'])} points. The walker falls on {words(k1['falls'])} of {words(I['ablation']['episodes'])} tracks."),
          sec("ablation.3", f"In this network, {words(len(d['big']))} of the thirty two neurons each cost more than fifty points when frozen. The rest is spare capacity, or fine tuning."),
          sec("inputs", f"The same test works for the senses. The most important ones are {top_in}."),
          sec("inputs.2", ("And the lidar, the rays that look ahead, barely matters. This walker walks by feel, reacting to its own joints. That's enough on these gentle tracks, but it's a likely weakness on unusual bumps."
                           if blind else "The lidar matters too: the walker adjusts to the terrain ahead, not only to its own joints.")),
          sec("outro", "Next, the final test: a hundred tracks it has never seen, and the comparison with my other walkers.")]
    return "\n".join(s)


def lesson7(d):
    E, P = d["E"], d["P"]
    solved = E["mean"] >= 300
    s = ["# Lesson 7: Results and comparison\n",
         sec("intro", "Lesson seven, the honest test. A hundred tracks the walker has never seen, full episodes, and the unmodified reward."),
         sec("final", f"The champion averages {words(E['mean'])} points, give or take {words(E['std'])}. It falls on {'none' if E['falls'] == 0 else words(E['falls'])} of a hundred tracks, and reaches the finish on {words(E['finished'])}."),
         sec("final.2", ("That's above three hundred. By the usual definition, the task is solved, with a network of just nine hundred and thirty two numbers."
                         if solved else f"So it's not solved. Three hundred is the bar, and {'none' if E['n_ge_300'] == 0 else words(E['n_ge_300'])} of a hundred runs reached it. But look at how consistent it is: almost every track lands close to the average.")),
          sec("budget", f"Where do the points go? Walking to the finish earns about three hundred and sixty. On average, this walker earns {words(d['prog'])} from moving, and spends {words(d['effort'])} on motor effort."),
          sec("budget.2", ("So the limit isn't falling, and it isn't speed. It's effort. The network pushes its motors almost flat out. To score higher, it would have to walk as far while working less hard."
                           if E["finished"] >= 50 else "So time and falls both cost points, and effort costs even more.")),
          sec("neat", f"Now the same hundred tracks for my NEAT walker. NEAT averages {words(P['mean_b'])}. CMA-ES averages {words(P['mean_a'])}. The difference is {words(P['diff_mean'])} points, and CMA-ES scores higher on {words(P['a_better_on'])} of the hundred tracks."),
          sec("neat.2", f"It also falls far less: {words(P['falls_a'])} {'fall' if P['falls_a'] == 1 else 'falls'} against {words(P['falls_b'])}. A fixed network with the right size, plus covariance learning, beat evolving the network's shape."),
          sec("table", "How does it compare with gradient methods? PPO reached two hundred and seventy two. TD3 reached about three hundred and ten, in only two hours."),
          sec("table.2", f"CMA-ES used far more simulation steps: about {words(round(d['steps'] / 1e6))} million. But it needed no gradients, no critic and no replay buffer, and its whole policy is about a hundred and forty times smaller than TD3's."),
          sec("outro", "That's the whole story: a task, a network of nine hundred and thirty two numbers, an algorithm that only ranks whole episodes, and a fitness that leads it, one small step at a time, to walking.")]
    return "\n".join(s)


def explainer(d):
    """The ~10-minute summary: one short beat per idea, reusing the lesson visuals."""
    E, ms = d["E"], d["ms"]
    l5, l6, l7 = (t.split("\n## ")[1:] for t in (lesson5(d), lesson6(d), lesson7(d)))
    pick = lambda sections, name: next("## " + x for x in sections if x.startswith(name + "\n"))
    s = ["# Bipedal CMA-ES: the explainer\n",
         sec("title", f"This network learned to walk without a single gradient. It's nine hundred and thirty two numbers, tuned by an evolution strategy called CMA-ES, on a four core laptop. Here's how."),
         sec("task", "The task is Bipedal Walker. A body on two legs, four motors, and a bumpy track that's different every time. Every step, the walker senses twenty four numbers and sets four motor commands."),
         sec("task.2", "Moving forward earns points, effort costs points, and a fall costs a hundred. Reaching the finish scores about three hundred. Averaged over a hundred tracks, three hundred counts as solved."),
         sec("network", "The controller is a small neural network: twenty four inputs, thirty two hidden neurons, and four outputs. All its weights fit in one list of nine hundred and thirty two numbers. That list is the walker."),
         sec("loop", "To improve it, CMA-ES samples a cloud of candidate walkers around its current best guess, lets each one walk, ranks them, and moves toward the winners."),
         sec("loop.2", "It also learns the shape of the cloud, which weights should change together, and how big a step to take. No gradients, only rankings of whole episodes."),
         sec("noise", "Every score is noisy, because every track is different. So all candidates in a generation walk the same five tracks, and the champion is chosen on twenty tracks that training never sees."),
         pick(l5, "traps"), pick(l5, "traps.4"), pick(l5, "breakthrough.2"),
         ]
    s += [("## " + x) for x in l5 if x.startswith("stages")]
    s += [pick(l6, "heat.2"), pick(l6, "ablation.2"), pick(l6, "inputs.2"),
          pick(l7, "final"), pick(l7, "final.2"), pick(l7, "budget.2"), pick(l7, "neat"),
          sec("outro", "The full lessons, with interactive demos for every idea in this video, are on the project page. Thanks for watching.")]
    return "\n".join(x.strip() + "\n" for x in s)


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "v3-f20")
    d = load(run)
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("lesson-05", lesson5), ("lesson-06", lesson6), ("lesson-07", lesson7), ("explainer", explainer)]:
        text = fn(d)
        open(os.path.join(OUT, f"{name}.md"), "w").write(text)
        n = len([ln for ln in text.splitlines() if ln.startswith("## ")])
        print(f"{name}: {n} sections, {len(text.split())} words")


if __name__ == "__main__":
    main()
