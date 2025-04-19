"""CLI: Validate every JSON under annotated_dataset/schema_* using Pydantic V2."""

from pathlib import Path
import sys
import json

# Adjust path if scripts are run from root or src/scripts
try:
    from utils.schema_utils import FixedSchema
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent))  # Add src to path
    from utils.schema_utils import FixedSchema

ANNOT_ROOT = Path("annotated_dataset")

errors = 0
count = 0
for p in ANNOT_ROOT.rglob("schema_*/**/*.json"):
    if p.is_file():
        count += 1
        try:
            # Use model_validate_json in Pydantic V2
            FixedSchema.model_validate_json(p.read_text("utf-8"))
            print(f"[OK]    {p.relative_to(ANNOT_ROOT)}")
        except Exception as e:  # Catch Pydantic's ValidationError and others
            print(f"[ERROR] {p.relative_to(ANNOT_ROOT)}: {e}")
            errors += 1

print("-" * 20)
if errors:
    print(f"Finished validating {count} files with {errors} invalid schema file(s).")
    sys.exit(1)
print(f"All {count} schema files validated successfully.")
