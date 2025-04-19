"""CLI: Re-load and save schemas to apply model defaults/validators (Pydantic V2)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Adjust path if scripts are run from root or src/scripts
try:
    from utils.schema_utils import FixedSchema
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))  # Add src to path
    from utils.schema_utils import FixedSchema

ANNOT_ROOT = Path("annotated_dataset")
count = 0
errors = 0

print("Re-validating and re-saving schema files...")
for p in ANNOT_ROOT.rglob("schema_*/**/*.json"):
    if p.is_file():
        try:
            # Load using Pydantic V2 (validates and applies defaults)
            schema_obj = FixedSchema.model_validate_json(p.read_text("utf‑8"))
            # Save using Pydantic V2's method
            schema_obj.to_json(p)  # Overwrite existing file
            print(f"[RE-SAVED] {p.relative_to(ANNOT_ROOT)}")
            count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {p.relative_to(ANNOT_ROOT)}: {e}")
            errors += 1

print("-" * 20)
if errors:
    print(f"Finished processing {count + errors} files with {errors} error(s).")
else:
    print(f"Successfully re-validated and re-saved {count} schema files.")

if errors > 0:
    sys.exit(1)
