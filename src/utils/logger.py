"""Logger configuration with file and console handlers."""

import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
console_formatter = logging.Formatter(
    '%(levelname)-8s | %(message)s'
)

# Create file handler with current date in filename
current_date = datetime.now().strftime("%Y-%m-%d")
log_file = logs_dir / f"image_annotater_{current_date}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(file_formatter)

# Create console handler with a less verbose format and higher level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Only INFO and above go to console
console_handler.setFormatter(console_formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)


def get_logger(name):
    """Get a logger with the specified name."""
    logger = logging.getLogger(name)

    # Conditionally set DEBUG level if environment variable is set
    if os.environ.get("DEBUG_ANNOTATER", "0") == "1":
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger


# Specialized loggers for different components
def get_app_logger():
    """Get the main application logger."""
    return get_logger("app")


def get_canvas_logger():
    """Get logger for canvas operations."""
    return get_logger("canvas")


def get_gemini_logger():
    """Get logger for Gemini AI operations."""
    return get_logger("gemini")


def get_file_logger():
    """Get logger for file operations."""
    return get_logger("file")


def log_schema(logger, schema, prefix="Schema"):
    """Log a schema object in a readable format."""
    import json
    if hasattr(schema, "model_dump"):
        # Pydantic v2 model
        schema_dict = schema.model_dump()
    else:
        # Regular dict or other object
        schema_dict = schema

    logger.debug(f"{prefix}: {json.dumps(schema_dict, indent=2, default=str)}")