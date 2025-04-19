"""Filesystem helpers: walk dataset, derive category, save annotated artefacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

from .schema_utils import FixedSchema, BBox

DATASET_ROOT = Path("dataset")
ANNOT_ROOT = Path("annotated_dataset")


def list_images() -> List[str]:
    """Return all common image format paths under dataset/ (relative str)."""
    # Ensure you include all extensions present in your dataset
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic"}  # Example set
    imgs: List[str] = []
    if not DATASET_ROOT.exists():
        st.warning(f"Dataset directory '{DATASET_ROOT}' not found! Create it and add images.")
        return imgs
    if not any(DATASET_ROOT.iterdir()):
        st.warning(f"Dataset directory '{DATASET_ROOT}' is empty.")
        # return imgs # Keep commented to allow app load even if empty

    # Use rglob to find files recursively
    for p in DATASET_ROOT.rglob("*"):
        # Check if it's a file and has a recognized extension
        if p.is_file() and p.suffix.lower() in exts:
            try:
                # Store path relative to CWD, works well with Streamlit
                rel_path_str = str(p.relative_to(Path.cwd()))
                imgs.append(rel_path_str)
            except ValueError:
                # Fallback if not relative to CWD (e.g., different drive)
                imgs.append(str(p.resolve()))  # Store absolute path as fallback

    if not imgs and DATASET_ROOT.exists() and any(DATASET_ROOT.iterdir()):
        st.warning(f"No image files found with extensions: {exts}. Check your dataset structure and file types.")

    return sorted(imgs)


def derive_category(img_path: Path | str) -> str:
    """Derive category based on the parent directory relative to DATASET_ROOT."""
    p = Path(img_path)
    try:
        # Ensure DATASET_ROOT is absolute for reliable comparison
        abs_dataset_root = DATASET_ROOT.resolve()
        abs_p = p.resolve()
        # Get path relative to the absolute DATASET_ROOT
        rel_path = abs_p.relative_to(abs_dataset_root)
        # Use the parent directory path as the category string
        # Use as_posix() for cross-platform path separator consistency ('/')
        category_path = rel_path.parent.as_posix()
        # If the image is directly in DATASET_ROOT, parent is '.', use a specific name
        return category_path if category_path != "." else "(root)"
    except ValueError:
        # Fallback if path is not inside DATASET_ROOT
        parent_name = p.parent.name
        # Try to provide more context if possible
        return f"(external) {parent_name}" if parent_name and parent_name != "." else "(unknown)"
    except Exception:  # Catch other potential errors like file not found during resolve
        return "(error determining category)"


def save_annotated_image(original_path: str | Path, rects: List[BBox]) -> Path:
    """Draw rectangles on a copy of *original_path* and save to annotated_<cat>/.
    
    Args:
        original_path: Path to the original image
        rects: List of bounding boxes to draw
        
    Returns:
        Path to the saved annotated image
        
    Raises:
        Exception: If image can't be opened or saved
    """
    orig = Path(original_path)
    cat = derive_category(orig)
    out_dir = ANNOT_ROOT / f"annotated_{cat}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / orig.name

    try:
        # Copy first to preserve exif etc.
        shutil.copy2(orig, out_path)

        if rects:
            # Open copied image and draw rectangles
            try:
                img = Image.open(out_path).convert("RGB")
            except UnidentifiedImageError:
                st.error(f"Could not open image: {out_path}. Format may be unsupported.")
                return out_path

            draw = ImageDraw.Draw(img)

            # Draw each rectangle with different colors for multiple boxes
            colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]
            for i, bbox in enumerate(rects):
                color = colors[i % len(colors)]  # Cycle through colors
                # Draw polygon and add a box number
                draw.polygon(bbox, outline=color, width=3)
                # Add a small label with box number
                draw.text((bbox[0][0] + 5, bbox[0][1] + 5), f"#{i + 1}", fill=color)

            # Save with rectangles
            img.save(out_path)

            # Show success message
            st.success(f"Saved annotated image with {len(rects)} bounding boxes")

        return out_path

    except Exception as e:
        st.error(f"Error saving annotated image: {str(e)}")
        # Return original path if saving fails
        return Path(original_path)


def save_schema(schema: FixedSchema, category: str) -> Path:
    """Save schema to JSON file.
    
    Args:
        schema: The schema object to save
        category: Category name for folder organization
        
    Returns:
        Path to the saved schema file
    """
    # Create output directory
    out_dir = ANNOT_ROOT / f"schema_{category}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define output path
    out_path = out_dir / f"{schema.image_id}.json"

    # Check if file exists
    if out_path.exists():
        # We're overwriting an existing schema
        st.info(f"Updating existing schema: {out_path.name}")

    # Save schema
    try:
        schema.to_json(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error saving schema: {str(e)}")
        raise


def update_schema(path: Path, **updates) -> None:
    """Update existing schema with new values.
    
    Args:
        path: Path to the schema JSON file
        updates: Key-value pairs to update in the schema
    """
    try:
        # Read existing schema
        data = json.loads(path.read_text("utf-8"))

        # Update with new values
        data.update(updates)

        # Write back to file
        path.write_text(json.dumps(data, indent=2), "utf-8")
    except Exception as e:
        st.error(f"Error updating schema: {str(e)}")
        raise


def check_existing_annotation(image_path: str | Path) -> Optional[Dict[str, Any]]:
    """Check if annotation already exists for this image.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Existing schema as dict if found, None otherwise
    """
    path = Path(image_path)
    category = derive_category(path)
    schema_dir = ANNOT_ROOT / f"schema_{category}"

    if not schema_dir.exists():
        return None

    # Look for JSON with same name as image
    json_path = schema_dir / f"{path.stem}.json"
    if not json_path.exists():
        return None

    # Read and return schema
    try:
        return json.loads(json_path.read_text("utf-8"))
    except Exception:
        return None


def list_annotated_images() -> Dict[str, Path]:
    """List all annotated images with their schema paths.
    
    Returns:
        Dict mapping image IDs to schema paths
    """
    result = {}

    if not ANNOT_ROOT.exists():
        return result

    # Check all schema directories
    for schema_dir in ANNOT_ROOT.glob("schema_*"):
        if not schema_dir.is_dir():
            continue

        # Get all JSON files
        for json_path in schema_dir.glob("*.json"):
            try:
                # Use filename without extension as image ID
                image_id = json_path.stem
                result[image_id] = json_path
            except Exception:
                continue

    return result


def get_schema_stats() -> Dict[str, int]:
    """Get statistics about annotated schemas.
    
    Returns:
        Dict with statistics about annotations
    """
    stats = {
        "total": 0,
        "captioning": 0,
        "vqa": 0,
        "instruction": 0,
        "with_boxes": 0,
        "categories": set()
    }

    if not ANNOT_ROOT.exists():
        return stats

    # Iterate through all schema directories
    for schema_dir in ANNOT_ROOT.glob("schema_*"):
        if not schema_dir.is_dir():
            continue

        # Extract category from directory name
        category = schema_dir.name.replace("schema_", "")
        stats["categories"].add(category)

        # Process each JSON file
        for json_path in schema_dir.glob("*.json"):
            try:
                data = json.loads(json_path.read_text("utf-8"))
                stats["total"] += 1

                # Count by task type
                task_type = data.get("task_type", "unknown")
                if task_type in stats:
                    stats[task_type] += 1

                # Count schemas with bounding boxes
                if data.get("bounding_box") and len(data["bounding_box"]) > 0:
                    stats["with_boxes"] += 1

            except Exception:
                continue

    # Convert categories set to count
    stats["category_count"] = len(stats["categories"])
    stats["categories"] = sorted(list(stats["categories"]))

    return stats
