"""The curriculum's training log (docs/data/curriculum_log.json): one append-only list of events,
oldest to newest, for the site's Curriculum page.

Events: journal notes, stage starts/ends and pass checks (from each attempt's curriculum log),
goal checks, and "new champion" snapshots of the active stages (at most one per stage per hour).
Events that show a result get a replay snapshot taken when the event is first seen, so later
entries show the walker as it was then; events found after the fact (backfill) say so.

  python make_curriculum_log.py        # also called by make_curriculum_page.py
"""
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "docs" / "data"
LOGF = DATA / "curriculum_log.json"
REPLAYS = DATA / "episodes" / "log"
CHAMP_EVERY_MIN = 60
NOW = dt.datetime.now()


def eid(*parts):
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def run_info(run):
    d = ROOT / "runs" / run
    meta = json.loads((d / "best" / "meta.json").read_text()) if (d / "best" / "meta.json").exists() else {}
    cfg = json.loads((d / "run_config.json").read_text()).get("effective", {}) if (d / "run_config.json").exists() else {}
    return meta, cfg


def snapshot(event_id, run):
    """Replay of the run's current champion (with the run's training-wheels support, if any)."""
    meta, cfg = run_info(run)
    if not meta:
        return None, None
    sup = float(cfg.get("support", 0) or 0)
    REPLAYS.mkdir(parents=True, exist_ok=True)
    out = REPLAYS / f"{event_id}.json"
    r = subprocess.run([sys.executable, str(ROOT / "experiments" / "export_episode.py"), "--run", str(ROOT / "runs" / run),
                        "--seed", "30000008", "--every", "4", "--support", str(sup),
                        "--label", f"{run} champion (gen {meta.get('generation')})", "--out", str(out)], capture_output=True)
    if r.returncode != 0:
        return None, None
    ep = json.loads(out.read_text())
    return f"episodes/log/{out.name}", {"run": run, "gen": meta.get("generation"), "support": sup, "reward": ep["reward"],
                                        "steps": ep["steps"], "fell": ep["fell"]}


def parse_attempt(a):
    """Events from one attempt's log; the log has times only, so dates roll over when time goes back."""
    path = ROOT / a["log"]
    if not path.exists():
        return []
    day = dt.date.fromisoformat(a.get("start_date", NOW.date().isoformat()))
    last, out = None, []
    for line in path.read_text().splitlines():
        m = re.match(r"\[(\d\d):(\d\d):(\d\d)\]", line)
        if not m:
            continue
        t = dt.time(*map(int, m.groups()))
        if last and t < last:
            day += dt.timedelta(days=1)
        last = t
        iso = dt.datetime.combine(day, t).isoformat(timespec="seconds")
        body = line[m.end():].strip()
        mm = re.match(r"=== stage (\S+) \((.*)\)", body)
        if mm:
            out.append({"type": "stage_start", "iso": iso, "run": mm.group(1), "title": f"Stage started: {mm.group(1)}",
                        "text": mm.group(2)})
            continue
        mm = re.match(r"=== (\S+) done: (.+)", body)
        if mm:
            out.append({"type": "stage_end", "iso": iso, "run": mm.group(1), "outcome": mm.group(2),
                        "title": f"Stage ended: {mm.group(1)} ({mm.group(2)})"})
            continue
        mm = re.match(r"check: (\{.*\})", body)
        if mm:
            try:
                c = json.loads(mm.group(1))
            except json.JSONDecodeError:
                continue
            run = Path(c["run"]).name
            out.append({"type": "check", "iso": iso, "run": run, "passed": c.get("passed"), "metrics": c,
                        "title": f"Check {'passed' if c.get('passed') else 'not passed'}: {run}"})
    for e in out:
        e["attempt"] = a["prefix"]
    return out


def main():
    store = json.loads(LOGF.read_text()) if LOGF.exists() else {"events": []}
    have = {e["id"]: e for e in store["events"]}
    attempts = json.loads((DATA / "curriculum_attempts.json").read_text())
    cand = []
    for a in attempts:
        cand += parse_attempt(a)
    for j in json.loads((DATA / "curriculum_journal.json").read_text()):
        cand.append({"type": "note", "iso": j["iso"] + (":00" if len(j["iso"]) == 16 else ""), "title": j["title"], "text": j["text"]})
    for g in sorted(DATA.glob("goal_*.json")):
        r = json.loads(g.read_text())
        cand.append({"type": "goal", "iso": dt.datetime.fromtimestamp(g.stat().st_mtime).isoformat(timespec="seconds"),
                     "run": Path(r["run"]).name, "gen": r.get("champion_gen"), "metrics": r,
                     "title": f"Goal check {'MET' if r.get('met') else 'not met'}: {Path(r['run']).name}"})
    for e in cand:
        e["id"] = eid(e["type"], e.get("run"), e["iso"] if e["type"] != "goal" else e.get("gen"), e.get("title"))

    # active stages: started, not ended -> champion snapshots, at most one per stage per hour
    starts = {e["run"]: e for e in cand if e["type"] == "stage_start"}
    ended = {e["run"] for e in cand if e["type"] == "stage_end"}
    for run, st in starts.items():
        if run in ended:
            continue
        meta, cfg = run_info(run)
        if not meta:
            continue
        prev = [e for e in store["events"] if e["type"] == "champion" and e.get("run") == run]
        if prev and (prev[-1].get("gen") == meta.get("generation") or
                     NOW - dt.datetime.fromisoformat(prev[-1]["iso"]) < dt.timedelta(minutes=CHAMP_EVERY_MIN)):
            continue
        cand.append({"type": "champion", "iso": NOW.isoformat(timespec="seconds"), "run": run, "attempt": st.get("attempt"),
                     "gen": meta.get("generation"), "metrics": {k: meta.get(k) for k in ("mean", "selection", "falls", "front2_share",
                                                                                         "lead_changes", "abs_lead_mean", "mean_x")},
                     "title": f"New champion in {run}: generation {meta.get('generation')}"})
        cand[-1]["id"] = eid("champion", run, meta.get("generation"))

    def stage_at(iso):
        """The stage running at a given time (for notes)."""
        best = None
        for e in sorted((x for x in cand if x["type"] == "stage_start"), key=lambda x: x["iso"]):
            if e["iso"] <= iso:
                best = e["run"]
        return best

    new = 0
    for e in sorted(cand, key=lambda x: x["iso"]):
        if e["id"] in have:
            continue
        backfill = NOW - dt.datetime.fromisoformat(e["iso"]) > dt.timedelta(minutes=45)
        run = e.get("run") or (stage_at(e["iso"]) if e["type"] == "note" else None)
        if e["type"] in ("stage_end", "goal", "champion", "note") and run:
            e["replay"], e["replay_info"] = snapshot(e["id"], run)
            e["replay_when"] = "later" if backfill else "now"
        e["seen"] = NOW.isoformat(timespec="seconds")
        have[e["id"]] = e
        new += 1
    store["events"] = sorted(have.values(), key=lambda x: x["iso"])
    store["updated"] = NOW.strftime("%a %d %b %H:%M")
    LOGF.write_text(json.dumps(store, separators=(",", ":")))
    print(f"curriculum log: {len(store['events'])} events ({new} new)")


if __name__ == "__main__":
    main()
