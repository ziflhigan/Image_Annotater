"""Wrapper around streamlit‑drawable‑canvas that returns bbox coords."""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
# Import the image loader utility
from utils.file_utils import load_and_convert_image
from utils.logger import get_canvas_logger

logger = get_canvas_logger()

BBox = List[Tuple[int, int]]

MAX_CANVAS_WIDTH = 1600

# Default color choices
DEFAULT_COLORS = {
    "Red": "#FF0000",
    "Green": "#00FF00",
    "Blue": "#0000FF",
    "Yellow": "#FFFF00",
    "Purple": "#8A2BE2",
    "Orange": "#FFA500",
    "Cyan": "#00FFFF"
}


def draw(image_path: str, rotation_angle: int = 0) -> Tuple[List[BBox], float, Optional[Image.Image], List[str]]:
    """Enhanced canvas using load_and_convert_image, with resizing, rotation, and coordinates.

    Args:
        image_path: Path to the image file
        rotation_angle: Angle (0, 90, 180, 270) to rotate the image for display

    Returns:
        Tuple containing:
            - List of bounding boxes relative to the *displayed* canvas.
            - Scale factor applied (original_width / displayed_width).
            - The rotated image (PIL.Image) that was displayed or None if error.
            - List of colors for each bounding box.
    """
    boxes: List[BBox] = []
    box_colors: List[str] = []  # Store colors for boxes
    scale_factor = 1.0
    displayed_image = None

    # Initialize drawing mode in session state if not present
    mode_key = f"drawing_mode_{image_path}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "rect"  # Default to rectangle drawing mode

    # Drawing mode selection (Add this at the top of the function)
    mode_cols = st.columns([1, 3])
    with mode_cols[0]:
        st.write("Mode:")

    with mode_cols[1]:
        mode_options = {
            "Draw Rectangles": "rect",
            "Select & Edit": "transform"
        }
        selected_mode_name = st.radio(
            "Canvas Mode",
            options=list(mode_options.keys()),
            index=0 if st.session_state[mode_key] == "rect" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key=f"mode_select_{image_path}"
        )
        # Update session state with the actual mode value
        st.session_state[mode_key] = mode_options[selected_mode_name]

    logger.debug(f"Canvas mode set to: {st.session_state[mode_key]}")

    # Color selection
    if "box_color" not in st.session_state:
        st.session_state.box_color = "#FF0000"  # Default red

    color_cols = st.columns([1, 3, 1])
    with color_cols[0]:
        st.write("Box Color:")

    with color_cols[1]:
        selected_color_name = st.selectbox(
            "Choose color",
            options=list(DEFAULT_COLORS.keys()),
            index=list(DEFAULT_COLORS.values()).index(st.session_state.box_color)
            if st.session_state.box_color in DEFAULT_COLORS.values()
            else 0,
            label_visibility="collapsed",
            key=f"color_select_{image_path}"
        )
        st.session_state.box_color = DEFAULT_COLORS[selected_color_name]

    with color_cols[2]:
        st.color_picker(
            "Custom:",
            value=st.session_state.box_color,
            key=f"color_picker_{image_path}",
            on_change=lambda: setattr(st.session_state, "box_color", st.session_state[f"color_picker_{image_path}"])
        )

    logger.debug(f"Color for boxes set to: {st.session_state.box_color}")

    # Load image using the utility function (handles conversion to RGB)
    logger.info(f"Loading image: {image_path}")
    img_original = load_and_convert_image(image_path)

    if img_original is None:
        # Error already shown by load_and_convert_image
        logger.error(f"Failed to load image: {image_path}")
        return [], 1.0, None, []  # Return empty lists and default scale on load error

    try:
        # --- Rotation ---
        if rotation_angle != 0:
            img_rotated = img_original.rotate(-rotation_angle, expand=True, resample=Image.Resampling.BILINEAR)
            logger.debug(f"Rotated image by {rotation_angle}°")
        else:
            img_rotated = img_original

        w_orig, h_orig = img_rotated.size
        logger.debug(f"Rotated image dimensions: {w_orig}×{h_orig}")

        # --- user-controlled zoom ---
        # Default zoom so the image fits inside MAX_CANVAS_WIDTH
        autofit_zoom = min(100, int(100 * MAX_CANVAS_WIDTH / w_orig))

        # Initialize zoom in session state if not present
        zoom_key = f"zoom_slider_{image_path}_{rotation_angle}"
        if zoom_key not in st.session_state:
            st.session_state[zoom_key] = autofit_zoom

        # Create two columns for zoom controls
        zoom_cols = st.columns([3, 1])

        with zoom_cols[0]:
            zoom_pct = st.slider(
                "Zoom (%)",
                min_value=10,
                max_value=200,
                value=st.session_state[zoom_key],
                step=5,
                key=zoom_key
            )

        # --- Resizing based on zoom percentage ---
        display_w = int(w_orig * zoom_pct / 100)
        display_h = int(h_orig * zoom_pct / 100)
        scale_factor = w_orig / display_w  # This maintains the same logic as before

        if zoom_pct != 100:
            try:
                img_display = img_rotated.resize((display_w, display_h), Image.Resampling.LANCZOS)
            except AttributeError:  # Older Pillow
                img_display = img_rotated.resize((display_w, display_h), Image.LANCZOS)
            logger.debug(f"Resized for display: {display_w}×{display_h}, scale factor: {scale_factor}")
        else:
            img_display = img_rotated
            scale_factor = 1.0
            logger.debug("No resizing needed (100% zoom)")

        # Keep track of the displayed image for saving
        displayed_image = img_rotated  # This is the full-size rotated image

        st.caption(f"Original dimensions: {img_original.width}×{img_original.height} | "
                   f"Displayed as: {display_w}×{display_h} (Rotation: {rotation_angle}°, "
                   f"Scale: {1 / scale_factor:.2f}x)")

        # --- Canvas ---
        # Use the drawing mode from session state
        current_drawing_mode = st.session_state[mode_key]

        # DON'T include drawing mode in canvas key to preserve objects when switching modes
        canvas_key = f"canvas_{image_path}_{rotation_angle}_{zoom_pct}"

        # Show a note about the coordinate system
        st.info("**Note:** Coordinates are shown using bottom-left as origin (0,0)")

        canvas_result = st_canvas(
            fill_color=f"rgba{tuple(int(st.session_state.box_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)) + (0.1,)}",
            stroke_width=2,
            stroke_color=st.session_state.box_color,
            background_image=img_display,  # Use the loaded, rotated, resized image
            update_streamlit=True,
            height=display_h,
            width=display_w,
            drawing_mode=current_drawing_mode,  # Use the mode from session state
            key=canvas_key,  # Key includes path, angle, zoom and drawing mode
            display_toolbar=True,
        )

        # --- Extract Rectangles ---
        if canvas_result.json_data and canvas_result.json_data.get("objects"):
            logger.debug(f"Canvas data received with {len(canvas_result.json_data['objects'])} objects")
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
                logger.info("Left: {}, Top: {}, Width: {}, Height: {}".format(left, top, width, height))

                # Get stroke color
                stroke_color = obj.get("stroke", st.session_state.box_color)
                box_colors.append(stroke_color)  # Store the box color

                # Check for valid dimensions
                if width <= 0 or height <= 0:
                    logger.warning(f"Skipping invalid rectangle with zero/negative dimensions: {obj}")
                    st.warning(f"Skipping invalid rectangle (zero/negative size) from canvas data: {obj}")
                    continue

                # Store bbox relative to the *displayed* canvas
                bbox_display: BBox = [
                    (left, display_h - top),
                    (left + width, display_h - top),
                    (left + width, display_h - top - height),
                    (left, display_h - top - height),
                ]
                boxes.append(bbox_display)
                logger.debug(
                    f"Added box #{i + 1}: top-left=({left},{top}), width={width}, height={height}, color={stroke_color}")

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
                        st.text(f"Color: {stroke_color}")  # Display the color
                        # Show estimated original coordinates
                        orig_tl_x = int(round(bbox_display[0][0] * scale_factor))
                        orig_tl_y = int(round(bbox_display[0][1] * scale_factor))
                        orig_br_x = int(round(bbox_display[2][0] * scale_factor))
                        orig_br_y = int(round(bbox_display[2][1] * scale_factor))
                        st.text(f"Approx Orig TL: ({orig_tl_x}, {orig_tl_y})")
                        st.text(f"Approx Orig BR: ({orig_br_x}, {orig_br_y})")

    except Exception as e:
        logger.error(f"Error during canvas processing: {str(e)}", exc_info=True)
        st.error(f"Error during canvas processing for {image_path}: {str(e)}")
        return [], 1.0, None, []  # Return empty lists and default scale on error

    return boxes, scale_factor, displayed_image, box_colors
