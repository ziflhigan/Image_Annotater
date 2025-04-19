"""Enhanced wrapper around streamlit-drawable-canvas with keyboard shortcuts."""

from __future__ import annotations

import math
from typing import List, Tuple

import streamlit as st
from PIL import Image, UnidentifiedImageError
from streamlit_drawable_canvas import st_canvas

BBox = List[Tuple[int, int]]

# Define a maximum width for display
MAX_CANVAS_WIDTH = 1200  # Adjust as needed


def draw(image_path: str, rotation_angle: int = 0) -> Tuple[List[BBox], float]:
    """Enhanced canvas with resizing, rotation, keyboard shortcuts, and coordinates display.

    Args:
        image_path: Path to the image file
        rotation_angle: Angle (0, 90, 180, 270) to rotate the image for display

    Returns:
        Tuple containing:
            - List of bounding boxes relative to the *displayed* canvas.
            - Scale factor applied to the image (original_width / displayed_width).
    """
    boxes: List[BBox] = []
    scale_factor = 1.0  # Default scale factor

    try:
        img_original = Image.open(image_path)

        # --- Rotation ---
        # Only apply rotation if angle is not 0
        if rotation_angle != 0:
            # Use expand=True to prevent cropping after rotation
            img_rotated = img_original.rotate(-rotation_angle, expand=True)
        else:
            img_rotated = img_original

        w_orig, h_orig = img_rotated.size  # Use dimensions of rotated image for scaling calc

        # --- Resizing ---
        display_w = w_orig
        display_h = h_orig

        if w_orig > MAX_CANVAS_WIDTH:
            scale_factor = w_orig / MAX_CANVAS_WIDTH
            display_w = MAX_CANVAS_WIDTH
            # Calculate new height maintaining aspect ratio, use ceiling to avoid minor crops
            display_h = math.ceil(h_orig / scale_factor)

            # Resize using a high-quality down sampling filter like LANCZOS or BOX
            try:
                img_display = img_rotated.resize((display_w, display_h), Image.Resampling.LANCZOS)
            except AttributeError:  # Handle older Pillow versions
                img_display = img_rotated.resize((display_w, display_h), Image.LANCZOS)
        else:
            img_display = img_rotated  # No resizing needed
            scale_factor = 1.0  # Ensure scale factor is 1 if not resized

        st.caption(f"Original dimensions: {img_original.width}×{img_original.height} | "
                   f"Displayed as: {display_w}×{display_h} (Rotation: {rotation_angle}°, "
                   f"Scale: {1 / scale_factor:.2f}x)")

        # --- Canvas ---
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",
            stroke_width=2,  # Slightly thinner for potentially large images
            stroke_color="#FF0000",
            background_image=img_display,  # Use the resized/rotated image
            update_streamlit=True,
            height=display_h,  # Use calculated display height
            width=display_w,  # Use calculated display width
            drawing_mode="rect",
            key=f"canvas_{image_path}_{rotation_angle}",  # Key includes path and angle to reset canvas on change
            display_toolbar=True,
        )

        # (Optional: Keep the JavaScript for keyboard shortcuts if desired)
        # st.markdown(...)

        # --- Extract Rectangles ---
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            st.subheader(
                f"📏 Bounding Boxes ({len(canvas_result.json_data['objects'])}) - Coords relative to displayed image")
            for i, obj in enumerate(canvas_result.json_data["objects"]):
                if obj["type"] == "rect":
                    # Extract coords from canvas (relative to display)
                    left, top = int(obj["left"]), int(obj["top"])
                    width, height = int(obj["width"]), int(obj["height"])

                    # Store bbox relative to the *displayed* canvas
                    # Scaling back happens *before saving* in main.py
                    bbox_display: BBox = [
                        (left, top),
                        (left + width, top),
                        (left + width, top + height),
                        (left, top + height),
                    ]
                    boxes.append(bbox_display)

                    # Display info about the box on the canvas
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
                            # Show estimated original coordinates too
                            orig_tl_x = int(bbox_display[0][0] * scale_factor)
                            orig_tl_y = int(bbox_display[0][1] * scale_factor)
                            orig_br_x = int(bbox_display[2][0] * scale_factor)
                            orig_br_y = int(bbox_display[2][1] * scale_factor)
                            st.text(f"Approx Orig TL: ({orig_tl_x}, {orig_tl_y})")
                            st.text(f"Approx Orig BR: ({orig_br_x}, {orig_br_y})")

    except UnidentifiedImageError:
        st.error(f"Error: Could not identify image file format for {image_path}")
        return [], 1.0  # Return empty list and default scale on error
    except FileNotFoundError:
        st.error(f"Error: Image file not found at {image_path}")
        return [], 1.0
    except Exception as e:
        st.error(f"Error loading or processing image for canvas: {str(e)}")
        return [], 1.0

    # Return boxes relative to displayed canvas and the scale factor
    return boxes, scale_factor
