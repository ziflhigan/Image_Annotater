"""Pydantic models that represent the *fixed* dataset schema + helpers."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, validator


# ── Sub‑models ───────────────────────────────────────────────────────────

class LanguageInfo(BaseModel):
    source: List[Literal["ms", "en"]] = Field(..., description="Languages present in text_ms / text_en")
    target: Optional[List[Literal["ms", "en"]]] = None


class Metadata(BaseModel):
    license: Optional[str] = "CC-BY"
    annotator_id: Optional[str] = None
    language_quality_score: Optional[float] = Field(None, ge=0.0, le=5.0)
    timestamp: Optional[datetime] = None


# ── Top‑level schema ────────────────────────────────────────────────────

BBox = List[Tuple[int, int]]  # 4 corner points (x,y)


class FixedSchema(BaseModel):
    image_id: str
    image_path: str
    task_type: Literal["captioning", "vqa", "instruction"]

    text_ms: str = ""
    answer_ms: str = ""
    text_en: str = ""
    answer_en: str = ""

    language: LanguageInfo = Field(default_factory=lambda: LanguageInfo(source=["ms", "en"]))
    source: str = "Image_Annotater"
    split: Literal["train", "val", "test"] = "train"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: List[str] = Field(default_factory=list)

    bounding_box: List[BBox] = Field(default_factory=list)
    metadata: Optional[Metadata] = Field(default_factory=Metadata)

    # Auto‑populate
    @validator("image_id", pre=True, always=True)
    def _auto_image_id(cls, v, values):  # noqa: N805
        if v:
            return v
        path = Path(values["image_path"])
        return path.stem

    @validator("metadata", pre=True, always=True)
    def _auto_timestamp(cls, v):  # noqa: N805
        if v and v.timestamp:
            return v
        m = v or Metadata()
        m.timestamp = datetime.now()
        return m

    # Convenience
    def to_json(self, out_path: Path | str, *, pretty: bool = True) -> None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        txt = self.json(indent=2) if pretty else self.json()
        path.write_text(txt, encoding="utf‑8")

    @classmethod
    def load(cls, path: Path | str) -> "FixedSchema":
        return cls.parse_file(path)
