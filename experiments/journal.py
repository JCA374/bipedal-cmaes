"""Append an entry to the curriculum journal shown on the site.
  python experiments/journal.py "Title" "Text (plain, can be several sentences)"
"""
import json
import sys
import time
from pathlib import Path

P = Path(__file__).resolve().parent.parent / "docs" / "data" / "curriculum_journal.json"
J = json.loads(P.read_text()) if P.exists() else []
J.append({"time": time.strftime("%a %d %b %H:%M"), "iso": time.strftime("%Y-%m-%dT%H:%M"),
          "title": sys.argv[1], "text": sys.argv[2]})
P.write_text(json.dumps(J, indent=1))
print(f"journal: {len(J)} entries")
