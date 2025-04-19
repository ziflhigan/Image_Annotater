"""CLI: Re‑generate schemas WITHOUT touching human fields (dangerous)."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.schema_utils import FixedSchema

SCHEMA_ROOT = Path("annotated_dataset")

for p in SCHEMA_ROOT.rglob("*.json"):
    data = json.loads(p.read_text("utf‑8"))
    fixed = FixedSchema(**data)  # will auto‑fix timestamp etc.
    fixed.to_json(p)
    print(f"Re‑written → {p}")

print("Done.")
