"""CLI: Validate every JSON under annotated_dataset/schema_* ."""

from pathlib import Path
import sys

from src.utils.schema_utils import FixedSchema

SCHEMA_ROOT = Path("annotated_dataset")

errors = 0
for p in SCHEMA_ROOT.rglob("*.json"):
    try:
        FixedSchema.load(p)
    except Exception as e:
        print(f"[ERROR] {p}: {e}")
        errors += 1

if errors:
    print(f"Finished with {errors} invalid schema files.")
    sys.exit(1)
print("All schema files valid")