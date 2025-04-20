"""
ai_utils.py  - Google Gen AI helper   (SDK ≥ 0.4.x)

* Uploads one image, asks Gemini‑2‑Flash to return a bilingual Q/A JSON array.
* Strips unsupported 'default' fields from the schema.
* Prints internal state when DEBUG_GEMINI=1 is set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any, List

from constants.prompts import SYSTEM_PROMPT, gemini_response_schema
from google import genai
from google.genai import types as gt  # typed config helpers
from google.genai.types import File  # upload handle
from pydantic import BaseModel
from utils.env_utils import getenv


# ── Pydantic model (NO non‑None defaults) ─────────────────────────────────────
class GeminiQA(BaseModel):
    task_type: Literal["captioning", "vqa", "instruction"]
    text_en: str  # Now required for all types including captioning
    text_ms: str  # Now required for all types including captioning
    answer_en: str
    answer_ms: str
    difficulty: Literal["easy", "medium", "hard"]
    language_quality_score: float
    tags: Optional[list[str]] = None


# ── Helpers / singletons ──────────────────────────────────────────────────────
_CLIENT: genai.Client | None = None
_FILE_CACHE: Dict[str, File] = {}
_DEBUG = bool(int(os.getenv("DEBUG_GEMINI", "0")))


def _client() -> genai.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=getenv("GEMINI_API_KEY", required=True))
    return _CLIENT


def _upload(path: Path) -> File:
    abs_path = str(path.resolve())
    if abs_path not in _FILE_CACHE:
        _FILE_CACHE[abs_path] = _client().files.upload(file=abs_path)
        if _DEBUG:
            print(f"[DEBUG] Uploaded → File ID {_FILE_CACHE[abs_path].name}")
    return _FILE_CACHE[abs_path]


def _strip_defaults(schema: dict) -> dict[Any, dict] | list[dict] | dict:
    """Recursively drop all 'default' keys from a JSON‑schema dict."""
    if isinstance(schema, dict):
        return {
            k: _strip_defaults(v)
            for k, v in schema.items()
            if k != "default"
        }
    if isinstance(schema, list):
        return [_strip_defaults(item) for item in schema]
    return schema


# ── Public API ────────────────────────────────────────────────────────────────
def generate_qa(
        image_path: str | Path,
        *,
        existing_schema: Optional[dict] = None,
        use_annotated_image: bool = False,
        model_name: str = "gemini-2.5-flash-preview-04-17",
) -> List[GeminiQA]:
    """
    Generate QA pairs for an image using Gemini.

    Args:
        image_path: Path to the image file
        existing_schema: Optional existing schema with text fields to consider
        use_annotated_image: Whether to use annotated image (with bounding boxes) instead of original
        model_name: Gemini model name to use

    Returns:
        List of GeminiQA objects containing question-answer pairs
    """
    # Determine the actual image path to use
    img_path = Path(image_path)

    if use_annotated_image:
        # Try to find an annotated image
        stem = img_path.stem
        rel_path = None
        try:
            from utils.file_utils import derive_full_relative_path, _get_output_subdir
            rel_structure = derive_full_relative_path(img_path)
            annot_dir = _get_output_subdir("annotated", rel_structure)
            annot_path = annot_dir / f"{stem}.jpg"
            if annot_path.exists():
                img_path = annot_path  # Use annotated image if exists
            else:
                print(f"[WARN] Annotated image not found at {annot_path}, using original")
        except Exception as e:
            print(f"[ERROR] Could not find annotated image: {e}")

    file_part = _upload(img_path)

    # Get response schema and clean it
    raw_schema = gemini_response_schema()
    clean_schema = _strip_defaults(raw_schema)

    if _DEBUG:
        print("[DEBUG] Clean response schema (truncated to 800 chars):")
        print(json.dumps(clean_schema, indent=2)[:800])

    # Create system instruction with existing text fields if provided
    system_instruction = SYSTEM_PROMPT
    if existing_schema:
        # Add existing text fields to help Gemini understand context
        context_info = ""
        if existing_schema.get("text_en"):
            context_info += f"\nExisting text_en: \"{existing_schema.get('text_en')}\""
        if existing_schema.get("text_ms"):
            context_info += f"\nExisting text_ms: \"{existing_schema.get('text_ms')}\""
        if context_info:
            system_instruction += context_info

    cfg = gt.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=clean_schema,
    )

    response = _client().models.generate_content(
        model=model_name,
        contents=[file_part],  # prompt is in system_instruction
        config=cfg,
    )

    if _DEBUG:
        print("[DEBUG] Raw Gemini text:")
        print(response.text[:500])

    try:
        # Parse the response as a list of QA pairs
        qa_pairs_data = json.loads(response.text)
        if not isinstance(qa_pairs_data, list):
            # If not a list, try to wrap it
            if isinstance(qa_pairs_data, dict):
                qa_pairs_data = [qa_pairs_data]
            else:
                raise ValueError(f"Expected list of QA pairs, got {type(qa_pairs_data)}")

        # Validate each QA pair
        qa_pairs = []
        for qa_data in qa_pairs_data:
            # Make sure all captioning tasks have questions
            if qa_data.get("task_type") == "captioning":
                # If question fields are missing or empty, add default questions
                if not qa_data.get("text_en"):
                    qa_data["text_en"] = "What can you see in this image?"
                if not qa_data.get("text_ms"):
                    qa_data["text_ms"] = "Apa yang anda dapat lihat dalam gambar ini?"

            # Try to parse each QA pair
            try:
                qa_pair = GeminiQA.model_validate(qa_data)
                qa_pairs.append(qa_pair)
            except Exception as e:
                print(f"[WARN] Invalid QA pair in response: {e}\n{qa_data}")

        if not qa_pairs:
            raise ValueError("No valid QA pairs in response")

        # Ensure we have at least one of each type if possible
        task_types = set(qa.task_type for qa in qa_pairs)
        if len(qa_pairs) >= 3:
            if "captioning" not in task_types:
                # Find a QA pair we can convert to captioning
                for qa in qa_pairs:
                    if qa.task_type != "captioning":
                        print("[INFO] Converting one QA pair to captioning type")
                        qa.task_type = "captioning"
                        # Don't reset the questions anymore
                        break

        return qa_pairs
    except Exception as err:
        raise RuntimeError(
            f"Error processing Gemini response: {err}\n--- RAW ---\n{response.text}"
        ) from err
