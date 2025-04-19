"""Environment & config helpers (load_dotenv wrapper)."""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # Searches .env in CWD or parents


def getenv(name: str, default: Optional[str] = None, *, required: bool = False) -> str | None:
    """Read environment variable with optional *required* enforcement."""
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value
