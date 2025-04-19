"""Wrapper around streamlit‑drawable‑canvas that returns bbox coords."""

from __future__ import annotations

import math
from typing import List, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

# Import the image loader utility
from utils.file_utils import load_and_convert_image

BBox = List[Tuple[int, int]]

MAX_CANVAS_WIDTH = 1200


def draw(image_path: str, rotation_angle: int = 0) -> Tuple[List[BBox], float]:
    """Enhanced canvas using load_and_convert_image, with resizing, rotation, and coordinates.

    Args:
        image_path: Path to the image file
        rotation_angle: Angle (0, 90, 180, 270) to rotate the image for display

    Returns:
        Tuple containing:
            - List of bounding boxes relative to the *displayed* canvas.
            - Scale factor applied (original_width / displayed_width).
    """
    boxes: List[BBox] = []
    scale_factor = 1.0

    # Load image using the utility function (handles conversion to RGB)
    img_original = load_and_convert_image(image_path)

    if img_original is None:
        # Error already shown by load_and_convert_image
        return [], 1.0  # Return empty list and default scale on load error

    try:
        # --- Rotation ---
        if rotation_angle != 0:
            img_rotated = img_original.rotate(-rotation_angle, expand=True, resample=Image.Resampling.BILINEAR)
        else:
            img_rotated = img_original

        w_orig, h_orig = img_rotated.size

        # --- Resizing ---
        display_w = w_orig
        display_h = h_orig
        if w_orig > MAX_CANVAS_WIDTH:
            scale_factor = w_orig / MAX_CANVAS_WIDTH
            display_w = MAX_CANVAS_WIDTH
            display_h = math.ceil(h_orig / scale_factor)
            try:
                img_display = img_rotated.resize((display_w, display_h), Image.Resampling.LANCZOS)
            except AttributeError:  # Older Pillow
                img_display = img_rotated.resize((display_w, display_h), Image.LANCZOS)
        else:
            img_display = img_rotated
            scale_factor = 1.0

        st.caption(f"Original dimensions: {img_original.width}×{img_original.height} | "
                   f"Displayed as: {display_w}×{display_h} (Rotation: {rotation_angle}°, "
                   f"Scale: {1 / scale_factor:.2f}x)")

        # --- Canvas ---
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=img_display,  # Use the loaded, rotated, resized image
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode="rect",
            key=f"canvas_{image_path}_{rotation_angle}",  # Key includes path and angle
            display_toolbar=True,
        )

        # --- Extract Rectangles ---
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            st.subheader(
                f"📏 Bounding Boxes ({len(canvas_result.json_data['objects'])}) - Coords relative to displayed image")
            valid_objects = [obj for obj in canvas_result.json_data["objects"] if obj["type"] == "rect"]
            for i, obj in enumerate(valid_objects):
                # Coordinates from canvas (relative to display)
                # Add checks for existence and type before int conversion
                left = int(obj.get("left", 0))
                top = int(obj.get("top", 0))
                width = int(obj.get("width", 0))
                height = int(obj.get("height", 0))

                # Check for valid dimensions
                if width <= 0 or height <= 0:
                    st.warning(f"Skipping invalid rectangle (zero/negative size) from canvas data: {obj}")
                    continue

                # Store bbox relative to the *displayed* canvas
                bbox_display: BBox = [
                    (left, top),
                    (left + width, top),
                    (left + width, top + height),
                    (left, top + height),
                ]
                boxes.append(bbox_display)

                # Display info (Optional)
                with st.expander(f"Box #{i + 1} (Displayed Coords)", expanded=False):
                    cols = st.columns(2)
                    coords_display = [f"Top-Left: ({bbox_display[0][0]}, {bbox_display[0][1]})",
                                      f"Top-Right: ({bbox_display[1][0]}, {bbox_display[1][1]})",
                                      f"Bottom-Right: ({bbox_display[2][0]}, {bbox_display[2][1]})",
                                      f"Bottom-Left: ({bbox_display[3][0]}, {bbox_display[3][1]})"]
                    with cols[0]:
                        for coord in coords_display: st.text(coord)
                    with cols[1]:
                        st.text(f"Width: {width}px")
                        st.text(f"Height: {height}px")
                        # Show estimated original coordinates
                        orig_tl_x = int(round(bbox_display[0][0] * scale_factor))
                        orig_tl_y = int(round(bbox_display[0][1] * scale_factor))
                        orig_br_x = int(round(bbox_display[2][0] * scale_factor))
                        orig_br_y = int(round(bbox_display[2][1] * scale_factor))
                        st.text(f"Approx Orig TL: ({orig_tl_x}, {orig_tl_y})")
                        st.text(f"Approx Orig BR: ({orig_br_x}, {orig_br_y})")

    except Exception as e:
        st.error(f"Error during canvas processing for {image_path}: {str(e)}")
        return [], 1.0  # Return empty list and default scale on error

    return boxes, scale_factor
