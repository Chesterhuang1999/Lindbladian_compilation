#!/usr/bin/env python3
"""Check that the paper pinned by the theorem checklist is the supplied PDF."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
record = json.loads((root / "paper-source.json").read_text())
source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(record["path"])
actual = hashlib.file_digest(source.open("rb"), "sha256").hexdigest()
if actual != record["sha256"]:
    raise SystemExit("Paper changed: update the theorem correspondence before reusing these results.")
print("Paper SHA-256 matches the formalization source.")
