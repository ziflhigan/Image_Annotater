"""Filesystem helpers: walk dataset, derive category, save annotated artefacts, rename dataset."""

from __future__ import annotations

import json
import os  # Keep for renaming function
import uuid  # Keep for renaming function
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError

# Assuming schema_utils is in the same directory or accessible via python path
try:
    from .schema_utils import VLMSFTData, BBox
except ImportError:
    # Fallback if running script directly might need path adjustment
    from schema_utils import VLMSFTData, BBox

DATASET_ROOT = Path("dataset").resolve()  # Resolve to absolute path
ANNOT_ROOT = Path("annotated_dataset").resolve()

# Register HEIF opener with Pillow
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    print("HEIF support enabled.")
except ImportError:
    print("Warning: pillow-heif not installed. HEIC support disabled.")


def list_images() -> List[str]:
    """Return all common image format paths under dataset/ (relative str to CWD)."""
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".heic", ".heif"}
    imgs: List[str] = []
    if not DATASET_ROOT.exists():
        st.warning(f"Dataset directory '{DATASET_ROOT}' not found!")
        return imgs
    if not any(DATASET_ROOT.iterdir()):
        st.warning(f"Dataset directory '{DATASET_ROOT}' is empty.")
        return imgs  # Return empty list if dir is empty

    print(f"--- list_images --- Searching in: {DATASET_ROOT}")  # DEBUG
    for p in DATASET_ROOT.rglob("*"):
        if p.is_file() and p.suffix.lower() in extensions:
            try:
                # Store path relative to CWD, works well with Streamlit widgets
                rel_path_str = str(p.relative_to(Path.cwd()))
                # print(f"    Found Image (relative to CWD): {rel_path_str}") # DEBUG
                imgs.append(rel_path_str)
            except ValueError:
                try:
                    # Fallback relative to DATASET_ROOT
                    rel_path_str = str(p.relative_to(DATASET_ROOT))
                    full_rel_path = str(Path("dataset") / rel_path_str)
                    # print(f"    Found Image (relative to DATASET_ROOT): {full_rel_path}") # DEBUG
                    imgs.append(full_rel_path)
                except ValueError:
                    # Absolute path as last resort
                    abs_path_str = str(p.resolve())
                    # print(f"    Found Image (Absolute Path): {abs_path_str}") # DEBUG
                    imgs.append(abs_path_str)

    print(f"--- list_images --- Found {len(imgs)} images.")  # DEBUG
    if not imgs and any(DATASET_ROOT.iterdir()):
        st.warning(f"No image files found with extensions: {extensions}. Check file types.")

    return sorted(imgs)


def derive_full_relative_path(img_path: Path | str) -> str:
    """
    Derive the relative path structure *within* DATASET_ROOT.
    Example: 'Food/Chinese', 'Transportation/Local', or '' if in root.
    """
    # print(f"--- derive_full_relative_path --- Input: {img_path}") # DEBUG
    try:
        p = Path(img_path).resolve()
        # print(f"    Resolved Path (p): {p}") # DEBUG
        # print(f"    DATASET_ROOT: {DATASET_ROOT}") # DEBUG
        rel_path = p.relative_to(DATASET_ROOT)
        # print(f"    Relative Path to DATASET_ROOT: {rel_path}") # DEBUG
        parent_path = rel_path.parent.as_posix()
        # print(f"    Parent Path (Posix): {parent_path}") # DEBUG
        result = parent_path if parent_path != "." else ""
        # print(f"    Derived Result: '{result}'") # DEBUG
        return result
    except ValueError:
        # print(f"    ERROR deriving path: Path {p} likely not inside {DATASET_ROOT}") # DEBUG
        return "(external)"
    except Exception as e:
        # print(f"    UNEXPECTED ERROR deriving path for {img_path}: {e}") # DEBUG
        st.error(f"Error deriving relative path for {img_path}: {e}")
        return "(error_deriving_path)"


def _get_output_subdir(base_prefix: str, relative_structure: str) -> Path:
    """Helper to construct the nested output subdirectory path."""
    if relative_structure and relative_structure not in ["(external)", "(error_deriving_path)"]:
        # Split the relative path (e.g., "Food/Chinese") into parts
        parts = Path(relative_structure).parts
        # Create the base output dir (e.g., "annotated_Food" or "schema_Food")
        output_base = ANNOT_ROOT / f"{base_prefix}_{parts[0]}"
        # Join the remaining parts (e.g., "Chinese")
        output_dir = output_base.joinpath(*parts[1:])
    elif relative_structure == "":  # Image was in dataset root
        output_dir = ANNOT_ROOT / f"{base_prefix}_(root)"
    else:  # Handle external or error cases
        output_dir = ANNOT_ROOT / f"{base_prefix}_{relative_structure}"

    return output_dir


def save_annotated_image(original_path_str: str, image_id: str, rects: List[BBox],
                         rect_colors: List[str] = None, rotated_image=None,
                         rotation_angle: int = 0) -> Path:
    """
    Loads original image, converts to JPG, draws rectangles,
    and saves to annotated_<category>/.../<image_id>.jpg using nested structure.

    Args:
        original_path_str: Relative path string to the original image.
        image_id: The ID (original stem or UUID) used for the output filename.
        rects: List of bounding boxes (scaled to original dimensions).
        rect_colors: List of colors for each rectangle (hex color strings).
        rotated_image: Optional pre-rotated PIL image to use instead of loading the original.
        rotation_angle: Angle of rotation to apply if rotated_image not provided.

    Returns: Path to the saved annotated JPG image.
    Raises: FileNotFoundError, UnidentifiedImageError, Exception.
    """
    original_path = Path(original_path_str)
    if not original_path.exists() and rotated_image is None:
        raise FileNotFoundError(f"Original image not found: {original_path_str}")

    relative_structure = derive_full_relative_path(original_path)
    out_dir = _get_output_subdir("annotated", relative_structure)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_id}.jpg"  # Use image_id for filename

    print(f"--- save_annotated_image ---")  # DEBUG
    print(f"    Original Path: {original_path_str}")  # DEBUG
    print(f"    Image ID: {image_id}")  # DEBUG
    print(f"    Relative Structure: {relative_structure}")  # DEBUG
    print(f"    Output Dir: {out_dir}")  # DEBUG
    print(f"    Output Path: {out_path}")  # DEBUG
    print(f"    Rotation Angle: {rotation_angle}")  # DEBUG
    print(f"    Using Provided Rotated Image: {rotated_image is not None}")  # DEBUG
    print(f"    Rect Colors Provided: {rect_colors is not None}")  # DEBUG

    try:
        # Use provided rotated image if available, otherwise load and rotate
        if rotated_image is not None:
            img = rotated_image
            print(f"    Using provided rotated image: {img.size}")  # DEBUG
        else:
            img = Image.open(original_path).convert("RGB")
            if rotation_angle != 0:
                img = img.rotate(-rotation_angle, expand=True, resample=Image.Resampling.BILINEAR)
                print(f"    Applied rotation of {rotation_angle}° to image: {img.size}")  # DEBUG
            else:
                print(f"    Loaded original image without rotation: {img.size}")  # DEBUG

        if rects:
            draw = ImageDraw.Draw(img)
            height = img.height
            # Only use this as fallback if no colors are provided
            default_colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF"]

            for i, bbox in enumerate(rects):
                if len(bbox) == 4:
                    # Convert from bottom-left to top-left coordinates
                    converted_bbox = [
                        (pt[0], height - pt[1]) for pt in bbox
                    ]

                    # Use the provided color if available, otherwise fall back to default colors
                    if rect_colors and i < len(rect_colors):
                        color = rect_colors[i]
                        print(f"    Using provided color for box #{i + 1}: {color}")  # DEBUG
                    else:
                        color = default_colors[i % len(default_colors)]
                        print(f"    Using default color for box #{i + 1}: {color}")  # DEBUG

                    draw.polygon(converted_bbox, outline=color, width=3)
                else:
                    st.warning(f"Skipping invalid bbox for drawing: {bbox}")
        img.save(out_path, "JPEG", quality=95)
        return out_path
    except UnidentifiedImageError:
        st.error(f"Could not identify image format: {original_path_str}")
        raise
    except Exception as e:
        st.error(f"Error processing/saving annotated image {image_id}.jpg: {str(e)}")
        raise


def save_schema(schema: VLMSFTData) -> Path:
    """
    Save schema to JSON file named <image_id>.json in schema_<category>/.../
    using nested structure derived from the original image path.

    Args:
        schema: The schema object to save (contains image_id and original image_path).

    Returns: Path to the saved schema file.
    """
    relative_structure = derive_full_relative_path(schema.image_path)
    out_dir = _get_output_subdir("schema", relative_structure)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{schema.image_id}.json"  # Use image_id from schema

    print(f"--- save_schema ---")  # DEBUG
    print(f"    Schema Image ID: {schema.image_id}")  # DEBUG
    print(f"    Original Path: {schema.image_path}")  # DEBUG
    print(f"    Relative Structure: {relative_structure}")  # DEBUG
    print(f"    Output Dir: {out_dir}")  # DEBUG
    print(f"    Output Path: {out_path}")  # DEBUG

    if out_path.exists():
        st.info(f"Updating existing schema: {out_path.relative_to(Path.cwd())}")

    try:
        schema.to_json(out_path)
        return out_path
    except Exception as e:
        st.error(f"Error saving schema {schema.image_id}.json: {str(e)}")
        raise


def check_existing_annotation(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Check if an annotation JSON file exists for this image using its filename stem.
    Looks in the corresponding schema_<category>/.../ directory.

    Args:
        image_path: Path to the original image.

    Returns: Existing schema as dict if found, None otherwise.
    """
    img_path_obj = Path(image_path)
    image_stem = img_path_obj.stem  # Get ID from filename stem
    relative_structure = derive_full_relative_path(img_path_obj)
    schema_dir = _get_output_subdir("schema", relative_structure)

    # print(f"--- check_existing_annotation ---") # DEBUG
    # print(f"    Image Path: {image_path}") # DEBUG
    # print(f"    Image Stem: {image_stem}") # DEBUG
    # print(f"    Schema Dir: {schema_dir}") # DEBUG

    if not schema_dir.exists():
        # print("    Schema dir does not exist.") # DEBUG
        return None

    json_path = schema_dir / f"{image_stem}.json"  # Look for <stem>.json
    # print(f"    Checking for: {json_path}") # DEBUG
    if not json_path.exists():
        # print("    Schema JSON file does not exist.") # DEBUG
        return None

    try:
        # print("    Schema JSON found, loading...") # DEBUG
        return json.loads(json_path.read_text("utf-8"))
    except Exception as e:
        st.error(f"Error reading existing schema {json_path.name}: {e}")
        return None


def get_annotated_image_stems() -> Set[str]:
    """Returns set of image stems (filenames without ext) that have schema files."""
    annotated_stems = set()
    if not ANNOT_ROOT.exists():
        return annotated_stems

    print("--- get_annotated_image_stems --- Searching schemas...")  # DEBUG
    # Check every potential schema file recursively
    for json_file in ANNOT_ROOT.rglob("schema_*/**/*.json"):
        if json_file.is_file():
            annotated_stems.add(json_file.stem)  # Store the stem

    print(f"    Found {len(annotated_stems)} annotated stems.")  # DEBUG
    return annotated_stems


def load_and_convert_image(image_path: str | Path) -> Optional[Image.Image]:
    """Loads an image from path, converts to RGB, returns PIL Image."""
    # print(f"--- load_and_convert_image --- Loading: {image_path}") # DEBUG
    try:
        resolved_path = Path(image_path).resolve()
        if not resolved_path.exists():
            st.error(f"Error: Image file not found at {resolved_path}")
            return None
        img = Image.open(resolved_path).convert("RGB")
        return img
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
        return None


def get_annotated_image_path(original_path: str, image_id: str) -> Optional[Path]:
    """Get the path to an annotated image if it exists.

    Args:
        original_path: Path to the original image
        image_id: Image ID (stem)

    Returns:
        Path to annotated image if exists, None otherwise
    """
    try:
        relative_structure = derive_full_relative_path(original_path)
        out_dir = _get_output_subdir("annotated", relative_structure)
        out_path = out_dir / f"{image_id}.jpg"

        if out_path.exists():
            return out_path
        return None
    except Exception as e:
        print(f"Error finding annotated image: {e}")
        return None


# --- Renaming Functionality ---

def rename_dataset_files_to_uuid(progress_bar=None) -> tuple[int, int, int]:
    """
    Renames image files in the 'dataset' folder to <uuid><ext>.
    Attempts to find corresponding schema files and rename them + update internal paths.

    !! WARNING: This modifies your original dataset and annotations. BACK UP FIRST. !!

    Args:
        progress_bar: Streamlit progress bar object (optional).

    Returns:
        Tuple: (success_count, annotation_updated_count, error_count)
    """
    print("--- RENAME Function Started ---")  # DEBUG
    st.warning(
        "**WARNING:** This operation will rename files in your original `dataset` folder and "
        "attempt to update corresponding annotations in `annotated_dataset`. "
        "**BACK UP BOTH FOLDERS BEFORE PROCEEDING.** This cannot be easily undone.")

    # Double confirmation (optional but recommended)
    # if not st.button("I understand the risk, proceed with renaming"):
    #      st.info("Renaming cancelled.")
    #      return 0, 0, 0

    all_images = list_images()  # Get paths relative to CWD
    total_files = len(all_images)
    success_count = 0
    annotation_updated_count = 0
    error_count = 0
    processed_stems = set()  # To avoid processing the same file if listed twice (e.g., abs/rel path issue)

    status_placeholder = st.empty()  # For dynamic status updates

    for i, img_path_str in enumerate(all_images):
        status_placeholder.text(f"Processing file {i + 1}/{total_files}...")
        if progress_bar:
            progress_bar.progress((i + 1) / total_files)

        try:
            original_path = Path(img_path_str).resolve()  # Use absolute path for safety
            original_stem = original_path.stem
            original_suffix = original_path.suffix.lower()  # Keep original extension

            # Skip if already processed (e.g. if list_images returned duplicates)
            if str(original_path) in processed_stems:
                print(f"    Skipping already processed path: {original_path}")
                continue
            processed_stems.add(str(original_path))

            # Skip if filename *already looks like* a UUID (basic check)
            try:
                uuid.UUID(original_stem)
                print(f"    Skipping potential UUID filename: {original_path.name}")
                continue  # Assume already renamed
            except ValueError:
                pass  # Not a UUID, proceed

            print(f"\n    Processing: {original_path}")  # DEBUG

            # --- Generate New UUID Name ---
            new_uuid = str(uuid.uuid4())
            new_filename = f"{new_uuid}{original_suffix}"
            new_path = original_path.with_name(new_filename)
            # Need new relative path for updating schema
            try:
                new_relative_path_str = str(new_path.relative_to(Path.cwd()))
            except ValueError:
                new_relative_path_str = str(Path("dataset") / new_path.relative_to(DATASET_ROOT))

            print(f"        Old Stem: {original_stem}")  # DEBUG
            print(f"        New UUID Stem: {new_uuid}")  # DEBUG
            print(f"        New Filename: {new_filename}")  # DEBUG
            print(f"        New Full Path: {new_path}")  # DEBUG
            print(f"        New Relative Path: {new_relative_path_str}")  # DEBUG

            # --- Check for and Update Annotation ---
            annotation_updated = False
            relative_structure = derive_full_relative_path(original_path)
            schema_dir = _get_output_subdir("schema", relative_structure)
            old_schema_path = schema_dir / f"{original_stem}.json"
            new_schema_path = schema_dir / f"{new_uuid}.json"

            print(f"        Checking for schema: {old_schema_path}")  # DEBUG
            if old_schema_path.exists():
                print(f"        FOUND existing schema: {old_schema_path}")  # DEBUG
                try:
                    # Load schema data
                    schema_data = json.loads(old_schema_path.read_text("utf-8"))
                    print(
                        f"            Loaded schema. Old image_id: {schema_data.get('image_id')}, "
                        f"Old image_path: {schema_data.get('image_path')}")  # DEBUG
                    # Update fields
                    schema_data["image_id"] = new_uuid
                    schema_data["image_path"] = new_relative_path_str  # Store new relative path
                    print(
                        f"            Updated schema. New image_id: {schema_data.get('image_id')}, "
                        f"New image_path: {schema_data.get('image_path')}")  # DEBUG

                    # Write updated data to *new* schema path (temporary step before renaming)
                    # Actually, better to rename schema file first, then update content
                    print(
                        f"            Attempting to rename schema "
                        f"{old_schema_path.name} -> {new_schema_path.name}")  # DEBUG
                    os.rename(old_schema_path, new_schema_path)  # Rename the JSON file
                    print(f"            Schema file renamed successfully.")  # DEBUG
                    print(f"            Attempting to write updated content to {new_schema_path.name}")  # DEBUG
                    new_schema_path.write_text(json.dumps(schema_data, indent=2),
                                               encoding="utf-8")  # Save updated content
                    print(f"            Schema content updated successfully.")  # DEBUG

                    annotation_updated = True
                    annotation_updated_count += 1
                except Exception as e_schema:
                    print(f"        ERROR updating schema for {original_stem}: {e_schema}")  # DEBUG
                    st.error(f"Failed to update schema for {original_path.name}: {e_schema}. Image was NOT renamed.")
                    # Attempt to rename schema back if rename succeeded but content update failed
                    if new_schema_path.exists() and not old_schema_path.exists():
                        try:
                            os.rename(new_schema_path, old_schema_path)
                            print(f"            Rolled back schema rename for {new_schema_path.name}")  # DEBUG
                        except Exception as e_rollback:
                            print(f"            ERROR rolling back schema rename: {e_rollback}")  # DEBUG
                    error_count += 1
                    continue  # Skip image rename if schema update failed

            # --- Rename Original Image File ---
            print(f"        Attempting to rename image file {original_path.name} -> {new_path.name}")  # DEBUG
            os.rename(original_path, new_path)
            print(f"        Image file renamed successfully.")  # DEBUG
            success_count += 1

        except Exception as e_main:
            print(f"    ERROR processing file {img_path_str}: {e_main}")  # DEBUG
            st.error(f"Failed to process {Path(img_path_str).name}: {e_main}")
            error_count += 1
            continue  # Move to next file

    status_placeholder.text(
        f"Renaming finished: {success_count} succeeded, {annotation_updated_count} annotations updated, {error_count} errors.")
    print(
        f"--- RENAME Function Finished --- Success: {success_count}, Annotations Updated: {annotation_updated_count}, Errors: {error_count}")  # DEBUG
    return success_count, annotation_updated_count, error_count


def get_schema_stats() -> Dict[str, Any]:
    """Get statistics about annotated schemas by scanning schema_* dirs recursively."""
    stats = {
        "total": 0,
        "captioning": 0,
        "vqa": 0,
        "instruction": 0,
        "with_boxes": 0,
        "categories": set()  # Store full category paths like 'Food/Chinese'
    }
    if not ANNOT_ROOT.exists():
        print("--- get_schema_stats --- Annotation root directory not found.")  # DEBUG
        return stats

    print(f"--- get_schema_stats --- Searching schemas in: {ANNOT_ROOT}")  # DEBUG
    # Iterate through all potential schema directories recursively looking for JSON files
    schema_files_found = list(ANNOT_ROOT.rglob("schema_*/**/*.json"))
    print(f"    Found {len(schema_files_found)} potential schema JSON files.")  # DEBUG

    for json_path in schema_files_found:
        if json_path.is_file():
            # print(f"    Processing schema file: {json_path}") # DEBUG
            try:
                # Derive category structure from the json path relative to ANNOT_ROOT
                rel_path = json_path.relative_to(ANNOT_ROOT)
                # Remove the 'schema_' prefix from the first part and the filename stem
                category_parts = list(rel_path.parent.parts)
                if category_parts:
                    # Ensure the first part actually starts with schema_ before replacing
                    if category_parts[0].startswith("schema_"):
                        category_parts[0] = category_parts[0].replace("schema_", "", 1)
                        category_str = "/".join(category_parts)  # Rebuild path string
                        stats["categories"].add(category_str)
                        # print(f"        Derived Category: {category_str}") # DEBUG
                    # else:
                    # print(f"        WARNING: Directory doesn't start with 'schema_': {rel_path.parent}") # DEBUG

                # Load data and update stats
                data = json.loads(json_path.read_text("utf-8"))
                stats["total"] += 1
                task_type = data.get("task_type", "unknown")
                if task_type in stats:
                    stats[task_type] += 1
                # Count schemas with non-empty bounding boxes
                if data.get("bounding_box") and isinstance(data["bounding_box"], list) and len(
                        data["bounding_box"]) > 0:
                    stats["with_boxes"] += 1
            except json.JSONDecodeError as json_err:
                print(f"        WARNING: Skipping invalid JSON file: {json_path} ({json_err})")  # DEBUG
                st.warning(f"Skipping invalid schema file during stats calculation: {json_path.name}")
                continue
            except Exception as e:
                print(f"        WARNING: Error processing schema file {json_path}: {e}")  # DEBUG
                st.warning(f"Skipping schema file due to error: {json_path.name} ({e})")
                continue

    stats["category_count"] = len(stats["categories"])
    stats["categories"] = sorted(list(stats["categories"]))
    print(f"--- get_schema_stats --- Finished. Stats: {stats}")  # DEBUG
    return stats
