"""
Gemini interaction (new Google‑GenAI SDK, image upload + structured JSON output).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal

from google import genai
from pydantic import BaseModel, ValidationError

from utils.env_utils import getenv
from constants.prompts import SYSTEM_PROMPT, gemini_response_schema


# ── Pydantic model that mirrors the JSON we expect from Gemini ────────────────
class GeminiQA(BaseModel):
    task_type: Literal["captioning", "vqa", "instruction"]
    question_en: str | None = ""
    question_ms: str | None = ""
    answer_en: str
    answer_ms: str
    difficulty: Literal["easy", "medium", "hard"]
    language_quality_score: float
    tags: list[str] | None = []


# ── Internal helpers / cache ──────────────────────────────────────────────────
_CLIENT: genai.Client | None = None
_FILE_CACHE: Dict[Path, str] = {}  # maps local path → Gemini File ID


def _get_client() -> genai.Client:
    """Return a singleton Client initialised with the API key from .env."""
    global _CLIENT
    if _CLIENT is None:
        api_key = getenv("GEMINI_API_KEY", required=True)
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _upload_image(path: Path) -> str:
    """
    Upload *path* (if not already cached) and return the Gemini File ID.

    The Files API keeps media for 48 hours, so re‑using IDs avoids quota waste.
    """
    client = _get_client()
    if path in _FILE_CACHE:
        return _FILE_CACHE[path]

    handle = client.files.upload(file=str(path))  # returns File object
    _FILE_CACHE[path] = handle.name  # 'name' is the File ID
    return handle.name


# ── Public function used by Streamlit UI ──────────────────────────────────────
def generate_qa(
        image_path: str | Path,
        *,
        model_name: str = "gemini-2.0-flash",  # fast & cheap; change if desired
) -> GeminiQA:
    """
    Send *image_path* + system prompt to Gemini and parse a structured JSON
    answer that matches `GeminiQA`.
    """
    path = Path(image_path)
    file_id = _upload_image(path)

    client = _get_client()
    response = client.models.generate_content(
        model=model_name,
        # Prompt is `[<file>, SYSTEM_PROMPT]`
        contents=[{"file_id": file_id}, SYSTEM_PROMPT],
        config={
            "response_mime_type": "application/json",
            "response_schema": gemini_response_schema(),
        },
    )

    try:
        return GeminiQA.parse_raw(response.text)
    except ValidationError as err:
        # Surface the raw text so annotators can see what went wrong
        raise RuntimeError(
            f"Gemini response failed validation: {err}\n--- RAW ---\n{response.text}"
        ) from err
