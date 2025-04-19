"""Streamlit UI entry-point with improved UX and error handling."""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import List, Optional  # Added Tuple

import streamlit as st

# Import components
from components.canvas_box import draw as draw_canvas
from components.json_viewer import show_json, interactive_json_editor
from components.sidebar import image_selector
# Import utils
from utils.ai_utils import generate_qa
from utils.file_utils import (
    derive_category,
    save_annotated_image,
    save_schema,
    check_existing_annotation,
    get_schema_stats,
)
from utils.schema_utils import FixedSchema, BBox  # Import BBox if needed, though defined in canvas_box

# from streamlit.delta_generator import DeltaGenerator # Not explicitly used

# --- Page Config ---
st.set_page_config(
    page_title="Image_Annotater",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Check and initialize keys to prevent errors on first run
default_keys = {
    "rects": [],  # Stores bounding boxes from canvas (display coordinates)
    "schema": None,  # Stores the current FixedSchema object
    "selected_image_path": None,  # Path selected via sidebar button click
    "current_image": None,  # The image path currently being processed/displayed
    "rotation_angle": 0,  # Current display rotation (0, 90, 180, 270)
    "image_scale_factor": 1.0,  # Scale factor applied to image for display
    "last_action_time": time.time()  # Tracks last user action time (optional)
}
for key, default_value in default_keys.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# --- Helper Functions ---

def render_header():
    """Render the app header with stats."""
    st.title("📑 Image Annotater – VLM Malaysia")
    try:
        stats = get_schema_stats()
        if stats["total"] > 0:
            category_list = ", ".join(stats.get('categories', []))
            st.caption(
                f"📊 Stats: {stats['total']} annotations ({stats['with_boxes']} with boxes) "
                f"across {stats.get('category_count', 0)} categories ({category_list})"
            )
    except Exception:
        pass  # Skip stats on error
    st.markdown("---")


def check_and_load_annotation(img_path: str) -> None:
    """Check if annotation exists and offer to load it."""
    existing = check_existing_annotation(img_path)
    if existing:
        # Use a unique key based on the image path for the checkbox
        checkbox_key = f"load_existing_{img_path}"
        # Default to True only if schema isn't already loaded for this image
        default_load = st.session_state.schema is None
        if st.sidebar.checkbox("Load existing annotation?", value=default_load, key=checkbox_key):
            if st.session_state.schema is None:  # Load only if not already loaded
                try:
                    schema = FixedSchema.parse_obj(existing)
                    st.session_state.schema = schema
                    # Note: We don't load rects from schema here, user redraws or confirms.
                    # If you want to load boxes onto canvas, you'd need canvas support for initial state.
                    st.session_state.rects = []  # Start with empty rects on canvas
                    st.sidebar.success("Loaded existing annotation data")
                except Exception as e:
                    st.sidebar.error(f"Error loading annotation: {str(e)}")
        elif st.session_state.schema is None:
            st.sidebar.info("Existing annotation found but not loaded.")


def handle_confirm_annotation(img_path: str, scaled_back_boxes: List[BBox]) -> Optional[FixedSchema]:
    """Handle the confirm annotation button click. Accepts boxes scaled back to original image coordinates."""
    st.session_state.last_action_time = time.time()
    with st.spinner("Processing annotation..."):
        saved_schema = None # Initialize schema variable

        # 1. Save Annotated Image (Independent Step)
        try:
            # Pass rotation=0 because we save the original orientation image
            saved_img_path = save_annotated_image(img_path, scaled_back_boxes)
            # Success message is now inside save_annotated_image
        except Exception as img_e:
            st.error(f"Failed to save annotated image: {img_e}")
            # Decide if you want to proceed without saving the image
            # return None # Option: Stop if image saving fails

        # 2. Prepare Data for Schema
        try:
            category = derive_category(img_path)
            image_id = Path(img_path).stem

            # Start with core data for the new/updated annotation
            core_data = {
                "image_id": image_id,
                "image_path": str(Path(img_path)), # Ensure string path
                "task_type": "vqa" if scaled_back_boxes else "captioning",
                "bounding_box": scaled_back_boxes,
            }

            # If there's an existing schema loaded FOR THE CURRENT IMAGE, merge its relevant data
            existing_schema = st.session_state.schema
            if existing_schema and existing_schema.image_id == image_id:
                # Preserve fields that Confirm shouldn't reset
                core_data["difficulty"] = existing_schema.difficulty
                core_data["tags"] = existing_schema.tags
                core_data["text_ms"] = existing_schema.text_ms
                core_data["answer_ms"] = existing_schema.answer_ms
                core_data["text_en"] = existing_schema.text_en
                core_data["answer_en"] = existing_schema.answer_en
                # Handle metadata carefully - copy existing if present
                if existing_schema.metadata:
                    # Use deepcopy to avoid modifying the state object directly
                    core_data["metadata"] = deepcopy(existing_schema.metadata)
                    # Pydantic validator should handle timestamp update if needed/missing
                else:
                    core_data["metadata"] = None # Let validator create default
            else:
                # No existing schema or different image, let defaults apply
                core_data["metadata"] = None # Let validator create default Metadata

            # 3. Create/Validate the Schema Object
            # Use **core_data to unpack the dictionary as keyword arguments
            schema_obj = FixedSchema(**core_data)

            # 4. Save the Schema Object
            schema_file_path = save_schema(schema_obj, category)
            st.success(f"✅ Schema saved/updated to {schema_file_path.name}")
            saved_schema = schema_obj # Assign the successfully saved schema

        except Exception as e:
            # Catch potential schema processing/saving errors
            st.error(f"Error processing or saving schema: {str(e)}")
            # Optionally re-raise or log e for more details

        return saved_schema # Return the schema object or None if saving failed


def handle_gemini_qa(img_path: str, schema: FixedSchema) -> None:
    """Handle Gemini Q/A generation button click."""
    st.session_state.last_action_time = time.time()
    try:
        with st.spinner("Calling Gemini..."):
            qa = generate_qa(img_path)  # Removed custom_prompt logic as it wasn't defined

        # Update schema with Gemini response
        schema.task_type = qa.task_type  # Update task type from Gemini if needed
        schema.text_en = qa.question_en or ""
        schema.text_ms = qa.question_ms or ""
        schema.answer_en = qa.answer_en
        schema.answer_ms = qa.answer_ms
        schema.difficulty = qa.difficulty
        if schema.metadata:  # Ensure metadata exists
            schema.metadata.language_quality_score = qa.language_quality_score
        else:
            # If no metadata, create it - though confirm should create it first
            from utils.schema_utils import Metadata
            schema.metadata = Metadata(language_quality_score=qa.language_quality_score)
        schema.tags = qa.tags or []

        # Save updated schema
        category = derive_category(img_path)
        save_schema(schema, category)  # Overwrite existing schema file
        st.success("✅ Gemini Q/A added → schema updated")
        st.session_state.schema = schema  # Update schema in session state

    except Exception as e:
        st.error(f"Error generating Q/A: {str(e)}")


# --- Main App Layout ---
def main():
    """Main application function."""
    render_header()

    # Call the sidebar function - it updates st.session_state.selected_image_path
    image_selector()

    # Get the potentially updated image path from session state
    img_path_selected = st.session_state.selected_image_path

    # --- State transition logic ---
    # Check if the selected image has changed from the currently processed one
    if img_path_selected != st.session_state.current_image:
        st.session_state.current_image = img_path_selected
        # Reset state associated with the image
        st.session_state.schema = None
        st.session_state.rects = []
        st.session_state.rotation_angle = 0  # Reset rotation
        st.session_state.image_scale_factor = 1.0  # Reset scale
        # Force a rerun to cleanly load the new image state
        st.rerun()  # Use st.rerun() instead of st.experimental_rerun()
    # --- End State transition logic ---

    # Proceed only if an image is effectively selected AND current
    current_img_path = st.session_state.current_image
    if current_img_path:

        # Check for existing annotation (pass the current image path)
        # This needs to run after the potential state reset above
        check_and_load_annotation(current_img_path)

        # --- Define Layout Containers ---
        st.header("📜 Schema & Actions")
        schema_placeholder = st.container()  # Container for schema section

        st.markdown("---")  # Separator

        st.header("🖼️ Canvas")
        rotation_controls_placeholder = st.container()  # Container for rotation controls
        canvas_placeholder = st.container()  # Container for canvas
        # --- End Layout ---

        # --- Rotation Controls ---
        with rotation_controls_placeholder:
            col_rot_1, col_rot_2 = st.columns([1, 4])
            with col_rot_1:
                if st.button("🔄 Rotate 90° CW", key=f"rotate_{current_img_path}"):
                    st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
                    st.session_state.rects = []  # Clear boxes on rotation
                    st.warning("Bounding boxes cleared due to rotation.")
                    st.rerun()  # Use st.rerun()
            with col_rot_2:
                st.caption(f"Current display rotation: {st.session_state.rotation_angle}° clockwise")
        # --- End Rotation Controls ---

        # --- Canvas Section ---
        with canvas_placeholder:
            # Draw canvas and get boxes relative to displayed canvas, plus scale factor
            try:
                # Pass rotation angle to draw_canvas
                boxes_display, scale_factor = draw_canvas(
                    current_img_path,
                    st.session_state.rotation_angle
                )
                # Store the scale factor from the current rendering
                st.session_state.image_scale_factor = scale_factor
                # Store the currently drawn boxes (relative to displayed canvas)
                st.session_state.rects = boxes_display
            except Exception as e:
                st.error(f"Failed to render canvas: {e}")
                # Ensure defaults if canvas fails
                st.session_state.rects = []
                st.session_state.image_scale_factor = 1.0

        # --- Schema and Actions Section ---
        with schema_placeholder:
            schema_changed = False
            # Show current schema if available
            if st.session_state.schema:
                # Allow schema editing (only if schema exists)
                updated_schema_dict = interactive_json_editor(
                    st.session_state.schema.dict(), key=f"editor_{current_img_path}"
                )
                if updated_schema_dict:
                    try:
                        # Re-validate the updated dictionary
                        schema = FixedSchema.parse_obj(updated_schema_dict)
                        st.session_state.schema = schema  # Update state
                        category = derive_category(current_img_path)
                        save_schema(schema, category)  # Save changes immediately
                        st.success("Schema updated with your edits.")
                        schema_changed = True
                    except Exception as e:
                        st.error(f"Error updating schema: {str(e)}")

                # Display the current (possibly updated) schema
                show_json(st.session_state.schema.dict(), label="Schema Preview")

            # Action buttons
            col1, col2 = st.columns(2)
            with col1:
                # Confirm annotation button
                if st.button("✅ Confirm", use_container_width=True, type="primary"):
                    # Get the *latest* boxes drawn on the canvas from state
                    current_drawn_boxes = st.session_state.rects

                    # --- Scale boxes back to ORIGINAL image coordinates ---
                    scaled_back_boxes = current_drawn_boxes  # Default if no scaling
                    if abs(st.session_state.image_scale_factor - 1.0) > 1e-6:  # Check if scaling was applied
                        scaled_back_boxes = [
                            [
                                (int(round(x * st.session_state.image_scale_factor)),
                                 int(round(y * st.session_state.image_scale_factor)))
                                for x, y in box
                            ]
                            for box in current_drawn_boxes
                        ]

                    # Call handler with scaled-back boxes
                    new_or_updated_schema = handle_confirm_annotation(current_img_path, scaled_back_boxes)
                    if new_or_updated_schema:
                        st.session_state.schema = new_or_updated_schema  # Update state
                        schema_changed = True  # Indicate schema was updated

            with col2:
                # Generate Q/A button - disable if no schema is confirmed yet
                qa_button_disabled = st.session_state.schema is None
                if qa_button_disabled:
                    st.caption("Confirm annotation first to enable Q/A generation.")

                qa_button = st.button(
                    "🤖 Generate Q/A",
                    type="secondary",
                    use_container_width=True,
                    disabled=qa_button_disabled,
                    key=f"qa_btn_{current_img_path}"
                )

                if qa_button and not qa_button_disabled:
                    handle_gemini_qa(current_img_path, st.session_state.schema)
                    schema_changed = True  # Gemini updates the schema

            # Rerun if schema changed to ensure UI consistency
            if schema_changed:
                st.rerun()  # Use st.rerun()

            # Instructions based on current state
            if not st.session_state.schema:
                st.info("👆 Draw boxes on the image (if needed) and click Confirm.")
            elif not getattr(st.session_state.schema, 'answer_en', ''):  # Check if QA likely missing
                st.info(
                    "👆 Schema saved. Click Generate Q/A to add questions and answers using Gemini, or edit fields "
                    "manually.")

    else:
        # No image selected - Show initial instructions
        st.info("👈 Select an image from the sidebar to start annotating")
        with st.expander("📘 How to use this app", expanded=True):
            st.markdown("""
            ### Quick Guide

            1.  **Select an image** from the sidebar file explorer (use search/filters). 2.  (Optional) **Rotate** 
            the image for better viewing using the `🔄 Rotate` button. 3.  **Draw bounding boxes** (optional) on 
            objects of interest on the canvas. Boxes are cleared on rotation. 4.  **Confirm annotation** (`✅ 
            Confirm`) to save boxes (relative to original orientation) and create/update the basic schema. 5.  
            **Generate Q/A with Gemini** (`🤖 Generate Q/A`) to add bilingual questions and answers (requires 
            confirmed schema). 6.  (Optional) Review and **edit the schema** using the interactive editor or the JSON 
            preview fields. Changes are saved automatically.

            ### Keyboard Shortcuts (Canvas)

            * `Esc`: Cancel current drawing action.
            * `Del` / `Backspace`: Delete selected rectangle (use the toolbar button).

            ### Output Location

            All annotations are saved to the `annotated_dataset/` folder with:
            * Schema JSON files in `schema_[category]/`
            * Annotated images (copies with boxes drawn) in `annotated_[category]/`
            """)


# --- Entry Point ---
if __name__ == "__main__":
    main()
