"""docs/data/podcast.json from podcast/episode-NN.md (+ optional episode-NN.notes.json) and the published
docs/podcast/episode-NN.mp3 files.      python podcast/make_feed.py
"""
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
eps = []
for md in sorted((ROOT / "podcast").glob("episode-*.md")):
    n = re.search(r"episode-(\d+)", md.name).group(1)
    mp3 = ROOT / "docs" / "podcast" / f"episode-{n}.mp3"
    if not mp3.exists():
        continue
    text = md.read_text().strip().splitlines()
    title = text[0].lstrip("# ").strip()
    paras = [p.strip() for p in "\n".join(text[1:]).split("\n\n") if p.strip()]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp3)],
                               capture_output=True, text=True).stdout or 0)
    notes = json.loads((md.with_suffix(".notes.json")).read_text()) if md.with_suffix(".notes.json").exists() else []
    eps.append({"n": int(n), "title": title, "date": time.strftime("%a %d %b %Y, %H:%M", time.localtime(mp3.stat().st_mtime)),
                "minutes": round(dur / 60, 1), "mp3": f"../podcast/episode-{n}.mp3", "notes": notes, "transcript": paras})
(ROOT / "docs" / "data" / "podcast.json").write_text(json.dumps({"episodes": eps[::-1]}, indent=1))
print(f"{len(eps)} episode(s)")
