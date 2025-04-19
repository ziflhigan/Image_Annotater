"""
ai_utils.py  - Google Gen AI helper   (SDK ≥ 0.4.x)

* Uploads one image, asks Gemini‑2‑Flash to return a bilingual Q/A JSON.
* Strips unsupported 'default' fields from the schema.
* Prints internal state when DEBUG_GEMINI=1 is set.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Any, List

from google import genai
from google.genai import types as gt          # typed config helpers
from google.genai.types import File           # upload handle
from pydantic import BaseModel, ValidationError

from utils.env_utils import getenv
from constants.prompts import SYSTEM_PROMPT


# ── Pydantic model (NO non‑None defaults) ─────────────────────────────────────
class GeminiQA(BaseModel):
    task_type: Literal["captioning", "vqa", "instruction"]
    question_en: Optional[str] = None
    question_ms: Optional[str] = None
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
            print(f"[DEBUG] Uploaded → File ID {_FILE_CACHE[abs_path].name}")
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
    model_name: str = "gemini-2.0-flash",
) -> GeminiQA:
    file_part = _upload(Path(image_path))

    raw_schema = GeminiQA.model_json_schema()
    clean_schema = _strip_defaults(raw_schema)  # ✨ the critical line

    if _DEBUG:
        print("[DEBUG] Clean response schema (truncated to 800 chars):")
        print(json.dumps(clean_schema, indent=2)[:800])

    cfg = gt.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=clean_schema,
    )

    response = _client().models.generate_content(
        model=model_name,
        contents=[file_part],        # prompt is in system_instruction
        config=cfg,
    )

    if _DEBUG:
        print("[DEBUG] Raw Gemini text:")
        print(response.text[:500])

    try:
        return GeminiQA.parse_raw(response.text)
    except ValidationError as err:
        raise RuntimeError(
            f"Gemini response failed validation: {err}\n--- RAW ---\n{response.text}"
        ) from err
