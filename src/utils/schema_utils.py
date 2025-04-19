"""Pydantic models (V2) for the annotation schema, using filename stem as default ID."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Any

from pydantic import BaseModel, Field, model_validator, ConfigDict


# Register HEIF opener with Pillow - place this early
try:
    from PIL import Image
    import pillow_heif

    pillow_heif.register_heif_opener()
    print("HEIF support enabled.")
except ImportError:
    print("Warning: pillow-heif not installed. HEIC support disabled.")


# ── Sub‑models ───────────────────────────────────────────────────────────

class LanguageInfo(BaseModel):
    source: List[Literal["ms", "en"]] = Field(default=["ms", "en"],
                                              description="Languages present in text_ms / text_en")
    target: Optional[List[Literal["ms", "en"]]] = None


class Metadata(BaseModel):
    license: Optional[str] = "CC-BY"
    annotator_id: Optional[str] = None
    language_quality_score: Optional[float] = Field(None, ge=0.0, le=5.0)
    timestamp: datetime = Field(default_factory=datetime.now)


# ── Top‑level schema ────────────────────────────────────────────────────

BBox = List[Tuple[int, int]]  # 4 corner points (x,y)


class FixedSchema(BaseModel):
    # Allow Pydantic extra fields if needed, though generally avoided
    model_config = ConfigDict(extra='ignore')

    # image_id defaults to stem of image_path if not provided
    image_id: str = Field("", description="Unique identifier (defaults to filename stem).")
    image_path: str = Field(..., description="Original relative path of the image.")  # Make path required
    task_type: Literal["captioning", "vqa", "instruction"] = "vqa"  # Default

    text_ms: str = ""
    answer_ms: str = ""
    text_en: str = ""
    answer_en: str = ""

    language: LanguageInfo = Field(default_factory=LanguageInfo)
    source: str = "Image_Annotater"
    split: Literal["train", "val", "test"] = "train"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: List[str] = Field(default_factory=list)

    bounding_box: List[BBox] = Field(default_factory=list)
    metadata: Metadata = Field(default_factory=Metadata)

    # Pydantic V2 Validator: Set image_id from image_path stem if image_id is empty
    @model_validator(mode='before')
    @classmethod
    def set_image_id_from_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            image_id = data.get('image_id')
            image_path = data.get('image_path')
            # Set image_id only if it's missing or empty, and image_path is present
            if not image_id and image_path and isinstance(image_path, str):
                data['image_id'] = Path(image_path).stem
                # print(f"    Validator: Set image_id to '{data['image_id']}' from path '{image_path}'") # DEBUG
        return data

    # Pydantic V2 Validator: Ensure metadata timestamp is set (usually handled by default_factory)
    # Could add more complex multi-field validation here if needed.

    # Convenience Methods using Pydantic V2
    def to_json(self, out_path: Path | str, *, pretty: bool = True) -> None:
        """Saves the schema model to a JSON file."""
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Use model_dump_json for V2, ensuring datetime is handled
        json_str = self.model_dump_json(indent=2 if pretty else None)
        path.write_text(json_str, encoding="utf‑8")

    @classmethod
    def load(cls, path: Path | str) -> "FixedSchema":
        """Loads a schema model from a JSON file."""
        # Use model_validate_json for V2
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def from_dict(cls, data: dict) -> "FixedSchema":
        """Loads a schema model from a dictionary."""
        # Use model_validate for V2 (replaces parse_obj)
        return cls.model_validate(data)
