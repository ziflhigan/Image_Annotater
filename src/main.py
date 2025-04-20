"""Streamlit UI entry-point: Image annotation with hierarchical view and optional renaming."""

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any  # Added Dict, Any

import streamlit as st

# Import components
from components.canvas_box import draw as draw_canvas
from components.json_viewer import show_json, interactive_json_editor
from components.sidebar import image_selector  # Displays list and rename button
# Import utils
from utils.ai_utils import generate_qa, GeminiQA
from utils.file_utils import (
    save_annotated_image,
    save_schema,
    check_existing_annotation,  # Checks based on stem
    get_schema_stats,
    get_annotated_image_path,
    ANNOT_ROOT
)
# Import Pydantic model and BBox type
from utils.schema_utils import VLMSFTData, BBox, Metadata  # Import necessary submodels

# No longer need uuid here, file_utils handles it for renaming
# import uuid

# --- Page Config ---
st.set_page_config(
    page_title="Image_Annotater",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Extended state: Added flags for disabling buttons during processing
default_keys = {
    "rects": [],  # Drawn boxes (display coords)
    "schema": None,  # Current FixedSchema object for the displayed image
    "selected_image_path": None,  # Path selected via sidebar button click (transient)
    "current_image_path": None,  # Path being actively processed/displayed
    "rotation_angle": 0,  # Canvas rotation
    "image_scale_factor": 1.0,  # Canvas display scale
    "last_action_time": time.time(),
    "processing_qa": False,  # Flag to disable buttons during QA generation
    "displayed_image": None,  # Store the displayed (possibly rotated) image
    "qa_pairs": None,  # Store multiple QA pairs from Gemini
    "use_annotated_image": False,  # Flag to use annotated image for Gemini
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
            if len(category_list) > 100: category_list = category_list[:100] + "..."
            st.caption(
                f"📊 Stats: {stats['total']} annotations ({stats['with_boxes']} with boxes) "
                f"across {stats.get('category_count', 0)} categories. "
                f"Categories: {category_list if category_list else 'None'}"
            )
        else:
            st.caption("📊 Stats: No annotations found yet.")
    except Exception as e:
        st.warning(f"Could not retrieve annotation stats: {e}")
    st.markdown("---")


def check_and_load_annotation(img_path: str) -> bool:
    """Check if annotation exists (using stem), load if checkbox checked. Return True if loaded."""
    print(f"--- check_and_load_annotation --- Checking for: {img_path}")  # DEBUG
    existing_dict = check_existing_annotation(img_path)  # Checks based on stem
    loaded = False
    if existing_dict:
        # Use image path in the key for uniqueness
        checkbox_key = f"load_existing_{img_path}"
        # Default to True only if schema is None or for a different path
        load_checked = st.session_state.schema is None or st.session_state.schema.image_path != img_path

        if st.sidebar.checkbox("Load existing annotation?", value=load_checked, key=checkbox_key):
            if st.session_state.schema is None or st.session_state.schema.image_path != img_path:
                print(f"    Checkbox checked and need to load. Loading annotation for {img_path}...")  # DEBUG
                try:
                    schema = VLMSFTData.model_validate(existing_dict)  # Use V2 validation
                    st.session_state.schema = schema
                    st.session_state.rects = []  # Keep canvas empty initially when loading schema
                    st.sidebar.success(f"Loaded existing annotation for {Path(img_path).name}")
                    loaded = True
                except Exception as e:
                    print(f"    ERROR loading annotation: {e}")  # DEBUG
                    st.sidebar.error(f"Error loading annotation: {str(e)}")
                    st.session_state.schema = None  # Reset on error
        elif st.session_state.schema is None:
            print(f"    Existing annotation found for {img_path}, but checkbox unchecked.")  # DEBUG
            st.sidebar.info(f"Existing annotation found but not loaded.")
    else:
        print(f"    No existing annotation found for {img_path}.")  # DEBUG
    return loaded


def handle_confirm_annotation(img_path: str, scaled_back_boxes: List[BBox], rotated_img=None) -> Optional[VLMSFTData]:
    """Handle Confirm: create/update schema, save schema & image (using filename stem)."""
    st.session_state.last_action_time = time.time()
    saved_schema_obj = None
    print(f"--- handle_confirm_annotation --- Path: {img_path}")  # DEBUG
    with st.spinner("Processing annotation..."):
        try:
            img_path_str = str(Path(img_path))  # Ensure string path relative to CWD/Dataset
            image_stem = Path(img_path_str).stem  # ID is the stem

            # Core data for schema creation/update
            core_data = {
                # image_id will be set by validator from image_path if needed
                "image_path": img_path_str,
                "bounding_box": scaled_back_boxes,
                "task_type": random.choice(["vqa", "Instruction"]) if scaled_back_boxes else "captioning",
            }

            # If schema is already loaded for *this path*, merge relevant fields
            existing_schema = st.session_state.schema
            if existing_schema and existing_schema.image_path == img_path_str:
                print("    Merging with existing loaded schema...")  # DEBUG
                # Preserve fields not directly set by drawing boxes
                core_data["difficulty"] = existing_schema.difficulty
                core_data["tags"] = existing_schema.tags
                core_data["text_ms"] = existing_schema.text_ms
                core_data["answer_ms"] = existing_schema.answer_ms
                core_data["text_en"] = existing_schema.text_en
                core_data["answer_en"] = existing_schema.answer_en
                core_data["task_type"] = existing_schema.task_type  # Keep existing task type
                core_data["language"] = existing_schema.language
                core_data["split"] = existing_schema.split
                core_data["source"] = existing_schema.source
                # Metadata: Update timestamp within the existing object
                if existing_schema.metadata:
                    existing_schema.metadata.timestamp = datetime.now()  # Update timestamp
                    core_data["metadata"] = existing_schema.metadata
                else:
                    core_data["metadata"] = Metadata()  # Create new with current timestamp
            else:
                print("    Creating new schema from scratch or overwriting different image's schema.")  # DEBUG
                core_data["metadata"] = Metadata()  # Ensure metadata is created

            # Create/Validate the Schema Object using Pydantic V2
            # The validator will set image_id=image_stem if 'image_id' isn't in core_data
            schema_obj = VLMSFTData.model_validate(core_data)
            print(f"    Validated Schema. Image ID set to: {schema_obj.image_id}")  # DEBUG

            # Save the Schema JSON file (uses schema_obj.image_id which should be the stem)
            schema_file_path = save_schema(schema_obj)
            st.success(f"✅ Schema saved/updated: {schema_file_path.relative_to(Path.cwd())}")
            saved_schema_obj = schema_obj

            # Save Annotated Image (uses schema_obj.image_id) with rotation
            save_image_copy = True  # Always save copy on confirm for now
            if save_image_copy:
                try:
                    # Pass the rotated image if available
                    rot_angle = st.session_state.rotation_angle if rotated_img else 0
                    saved_img_path = save_annotated_image(
                        img_path_str,
                        schema_obj.image_id,
                        scaled_back_boxes,
                        rotated_img,
                        rot_angle
                    )
                    st.success(f"🖼️ Annotated image saved: {saved_img_path.relative_to(Path.cwd())}")
                except Exception as img_e:
                    st.error(f"Failed to save annotated image copy: {img_e}")

        except Exception as e:
            st.error(f"Error processing or saving annotation: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return None

        return saved_schema_obj


def qa_pair_selector(qa_pairs: List[GeminiQA], schema: VLMSFTData) -> Optional[VLMSFTData]:
    """Display a modal to select one QA pair from multiple options and update schema."""
    if not qa_pairs:
        st.error("No QA pairs available to select from.")
        return None

    # Create a container for the selection UI
    with st.container():
        st.subheader("🤖 Select a Question-Answer Pair")
        st.info("Choose one of the AI-generated QA pairs to use in your annotation:")

        # Create styled options for each QA pair
        st.markdown("""
        <style>
        .qa-option {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
            background-color: #f0f2f6;
            border-left: 4px solid #4b6cb7;
        }
        .qa-option-selected {
            background-color: #e6f7ff;
            border-left: 4px solid #1890ff;
        }
        .qa-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .qa-content {
            margin-bottom: 5px;
        }
        .qa-tags {
            color: #666;
            font-style: italic;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create radio buttons for each QA pair
        options = []
        for i, qa in enumerate(qa_pairs):
            display = f"**{i + 1}. {qa.task_type.upper()}** ({qa.difficulty})"
            options.append(display)

        # Radio buttons for selection
        selected_index = st.radio(
            "Select a QA pair:",
            range(len(options)),
            format_func=lambda i: options[i],
            key="qa_pair_selector"
        )

        # Display detailed view of each option
        for i, qa in enumerate(qa_pairs):
            # Apply different styling based on selection
            css_class = "qa-option qa-option-selected" if i == selected_index else "qa-option"

            st.markdown(f'<div class="{css_class}">', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="qa-header">Option {i + 1}: {qa.task_type.upper()} ({qa.difficulty}) - Score: {qa.language_quality_score}/5</div>
            <div class="qa-content">
                <p>🇬🇧 <strong>Q:</strong> {qa.text_en or ''}</p>
                <p>🇬🇧 <strong>A:</strong> {qa.answer_en}</p>
                <p>🇲🇾 <strong>Q:</strong> {qa.text_ms or ''}</p>
                <p>🇲🇾 <strong>A:</strong> {qa.answer_ms}</p>
            </div>
            <div class="qa-tags">Tags: {', '.join(qa.tags) if qa.tags else 'None'}</div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        selected_qa = qa_pairs[selected_index]

        # Preview the selected QA pair
        st.markdown("---")
        st.subheader("Preview Selected Pair")

        # Display the selected QA pair using Streamlit's native components
        with st.container():
            # Create a light gray background container
            st.markdown("""
            <style>
            .preview-container {
                background-color: #f8f9fa;
                border: 1px solid #c0c0c0;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 20px;
            }
            </style>
            """, unsafe_allow_html=True)

            with st.container():
                st.markdown('<div class="preview-container">', unsafe_allow_html=True)

                # Task, Difficulty, and Quality Score
                st.subheader(f"Task: {selected_qa.task_type.capitalize()}")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Difficulty:** {selected_qa.difficulty}")
                with col2:
                    st.markdown(f"**Quality Score:** {selected_qa.language_quality_score}/5")

                # English and Malay content in columns
                eng_col, my_col = st.columns(2)

                with eng_col:
                    st.markdown("#### English")
                    if selected_qa.text_en:
                        st.markdown(f"**Question:** {selected_qa.text_en}")
                    st.markdown(
                        f"**{'Answer'}:** {selected_qa.answer_en}")

                with my_col:
                    st.markdown("#### Malay")
                    if selected_qa.text_ms:
                        st.markdown(f"**Question:** {selected_qa.text_ms}")
                    st.markdown(
                        f"**{'Answer'}:** {selected_qa.answer_ms}")

                # Tags
                st.markdown(f"**Tags:** {', '.join(selected_qa.tags) if selected_qa.tags else 'None'}")

                st.markdown('</div>', unsafe_allow_html=True)

        # Confirm button
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Use Selected QA Pair", use_container_width=True, type="primary"):
                # Update schema with the selected QA pair
                updated_schema = schema.model_copy(deep=True)

                # Update schema fields
                updated_schema.task_type = selected_qa.task_type
                updated_schema.text_en = selected_qa.text_en or ""
                updated_schema.text_ms = selected_qa.text_ms or ""
                updated_schema.answer_en = selected_qa.answer_en
                updated_schema.answer_ms = selected_qa.answer_ms
                updated_schema.difficulty = selected_qa.difficulty
                updated_schema.tags = selected_qa.tags or []

                # Update metadata
                if updated_schema.metadata is None:
                    updated_schema.metadata = Metadata()
                updated_schema.metadata.language_quality_score = selected_qa.language_quality_score
                updated_schema.metadata.timestamp = datetime.now()

                # Save schema
                try:
                    save_schema(updated_schema)
                    st.success("✅ Schema updated with selected QA pair!")
                    # Clear QA pairs from session state
                    st.session_state.qa_pairs = None
                    return updated_schema
                except Exception as e:
                    st.error(f"Error saving schema: {e}")
                    return None

        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                # Clear QA pairs from session state
                st.session_state.qa_pairs = None
                return None

    # If we get here, no action was taken yet
    return None


def handle_gemini_qa(img_path: str, schema: VLMSFTData, use_annotated_image: bool) -> None:
    """Call Gemini, get multiple QA pairs, and store them in session state for selection."""
    st.session_state.last_action_time = time.time()
    st.session_state.processing_qa = True  # Set processing flag to disable buttons

    print(f"--- handle_gemini_qa --- Path: {img_path}, Schema ID: {schema.image_id}")  # DEBUG
    print(f"    Using annotated image: {use_annotated_image}")  # DEBUG

    try:
        with st.spinner("Calling Gemini AI for QA generation..."):
            # Get the existing schema as dict for context
            schema_dict = schema.model_dump()

            # Generate multiple QA pairs
            qa_pairs = generate_qa(
                img_path,
                existing_schema=schema_dict,
                use_annotated_image=use_annotated_image
            )

            # Store QA pairs in session state for selection
            st.session_state.qa_pairs = qa_pairs
            print(f"    Received {len(qa_pairs)} QA pairs from Gemini")  # DEBUG

    except Exception as e:
        st.error(f"Error generating Q/A: {str(e)}")
        st.session_state.qa_pairs = None
    finally:
        st.session_state.processing_qa = False  # Reset processing flag


# --- Main App ---
def main():
    # --- Initial setup & Static Sidebar Elements ---
    # print("\n\n=========================================\n--- Main Run Start ---") # DEBUG
    # ... (initial state print can be useful) ...

    render_header()

    # Define persistent sidebar widgets
    st.sidebar.header("Dataset Images")
    search_term = st.sidebar.text_input("🔍 Search images", key="search_images")
    filter_options = ["All", "Annotated", "Not Annotated"]
    selected_filter = st.sidebar.radio(
        "Filter by status:", filter_options, horizontal=True, key="filter_status"
    )

    # Display sidebar list (and rename button)
    # This function now also handles button clicks to update st.session_state.selected_image_path
    # print("--- Calling image_selector (Sidebar List Display) ---") # DEBUG
    image_selector(search_term, selected_filter)
    # print("--- Returned from image_selector ---") # DEBUG

    # --- Get state potentially updated by sidebar ---
    img_path_selected = st.session_state.selected_image_path
    # print(f"--- State values AFTER sidebar list display ---") # DEBUG
    # print(f"    img_path_selected: {img_path_selected}") # DEBUG
    # print(f"    current_image_path (before transition): {st.session_state.current_image_path}") # DEBUG

    # --- State transition: Image Change Detection ---
    # print("--- Checking for State Transition (Image Change) ---") # DEBUG
    rerun_needed = False
    if img_path_selected != st.session_state.current_image_path:
        print(
            f"    >>> State Transition DETECTED: '{st.session_state.current_image_path}' -> '{img_path_selected}' <<<")  # DEBUG
        st.session_state.current_image_path = img_path_selected
        # Reset state for the new image
        st.session_state.schema = None
        st.session_state.rects = []
        st.session_state.rotation_angle = 0
        st.session_state.image_scale_factor = 1.0
        st.session_state.displayed_image = None
        st.session_state.qa_pairs = None
        st.session_state.use_annotated_image = False
        rerun_needed = True  # Rerun to load the new image context

    # --- Load annotation IF image selected AND schema not loaded ---
    current_img_path = st.session_state.current_image_path
    annotation_loaded = False
    if current_img_path and st.session_state.schema is None:
        # print("    Image selected and schema is None, checking for existing annotation...") # DEBUG
        # Check/Load happens here, returns True if loaded
        annotation_loaded = check_and_load_annotation(current_img_path)
        if annotation_loaded:
            # print("    Annotation was loaded, signaling rerun needed.") # DEBUG
            rerun_needed = True  # Rerun to show loaded schema

    # --- Handle QA Pair Selection (if pairs exist in session state) ---
    if st.session_state.qa_pairs and st.session_state.schema:
        updated_schema = qa_pair_selector(st.session_state.qa_pairs, st.session_state.schema)
        if updated_schema is not None:
            st.session_state.schema = updated_schema
            # Force rerun to update UI with the selected QA pair
            rerun_needed = True

    # --- Trigger Rerun if needed AFTER state updates ---
    if rerun_needed:
        print("    <<< Triggering Rerun due to State Transition or Annotation Load >>>")  # DEBUG
        st.rerun()

    # --- Main Content Area (uses state potentially set above or in previous run) ---
    # print(f"--- Proceeding with current image (Post-Transition/Load Check) ---") # DEBUG
    # print(f"    current_img_path: {current_img_path}") # DEBUG

    if current_img_path:
        # print(f"--- Rendering Main Content for: {current_img_path} ---") # DEBUG
        current_img_stem = Path(current_img_path).stem

        # --- Layout Containers ---
        st.header("📜 Schema & Actions")
        schema_placeholder = st.container()
        st.markdown("---")
        st.header(f"🖼️ Canvas: {Path(current_img_path).name}")  # Use original name
        rotation_controls_placeholder = st.container()
        canvas_placeholder = st.container()

        # --- Rotation ---
        with rotation_controls_placeholder:
            col_rot_1, col_rot_2 = st.columns([1, 4])
            with col_rot_1:
                # Use image path in key for stability if filenames are unique
                rotate_button_disabled = st.session_state.processing_qa
                if st.button("🔄 Rotate 90° CW", key=f"rotate_{current_img_path}",
                             disabled=rotate_button_disabled):
                    st.session_state.rotation_angle = (st.session_state.rotation_angle + 90) % 360
                    st.session_state.rects = []  # Clear boxes on rotation
                    st.session_state.displayed_image = None  # Reset displayed image on rotation
                    st.warning("Bounding boxes cleared due to rotation.")
                    st.rerun()  # Rerun needed to redraw canvas rotated
            with col_rot_2:
                st.caption(f"Current display rotation: {st.session_state.rotation_angle}° clockwise")

        # --- Canvas ---
        with canvas_placeholder:
            # print(f"    Attempting to draw canvas for: {current_img_path} (Rot: {
            # st.session_state.rotation_angle})") # DEBUG
            try:
                # Key uses path and angle to reset on change
                boxes_display, scale_factor, displayed_image = draw_canvas(
                    current_img_path,
                    st.session_state.rotation_angle
                )
                st.session_state.image_scale_factor = scale_factor
                st.session_state.rects = boxes_display
                st.session_state.displayed_image = displayed_image  # Store the displayed image
            except Exception as e:
                st.error(f"Failed to render canvas: {e}")
                st.session_state.rects = []
                st.session_state.image_scale_factor = 1.0
                st.session_state.displayed_image = None

        # --- Schema and Actions ---
        with schema_placeholder:
            # print("    Rendering Schema & Actions section...") # DEBUG
            schema_changed_in_section = False  # Flag for changes in this section
            current_schema = st.session_state.schema

            if current_schema:
                # Check consistency: Schema ID should match current image stem
                if current_schema.image_id != current_img_stem:
                    st.error(
                        f"State inconsistency: Loaded schema ID '{current_schema.image_id}' does not match current image stem '{current_img_stem}'. Please re-select image.")
                    st.stop()  # Prevent further processing with inconsistent state

                # --- Schema Editor ---
                # print("    Rendering schema editor...") # DEBUG
                # Use image path in editor key
                updated_schema_obj = interactive_json_editor(
                    current_schema, key=f"editor_{current_img_path}"
                )
                if updated_schema_obj:
                    print("    Schema editor returned changes, updating state and saving...")  # DEBUG
                    st.session_state.schema = updated_schema_obj
                    try:
                        save_schema(updated_schema_obj)
                        st.success("Schema updated via editor and saved.")
                        schema_changed_in_section = True
                        current_schema = updated_schema_obj  # Use updated obj
                    except Exception as e:
                        st.error(f"Error saving schema after edit: {e}")

                # --- Schema Display ---
                # print(f"    Displaying schema preview for ID: {current_schema.image_id}") # DEBUG
                show_json(current_schema, label=f"Schema Preview ({current_schema.image_id})")

            else:
                # print("    No schema loaded, showing info message.") # DEBUG
                st.info("👆 Draw boxes (optional) and click Confirm to create the first schema.")

            # --- Action Buttons ---
            col1, col2 = st.columns(2)
            with col1:
                # Confirm button - Key uses image path
                confirm_key = f"confirm_{current_img_path}"
                # Disable button during QA processing
                confirm_button_disabled = st.session_state.processing_qa

                # print(f"    Rendering Confirm button (Key: {confirm_key})...") # DEBUG
                if st.button("✅ Confirm", use_container_width=True, type="primary",
                             key=confirm_key, disabled=confirm_button_disabled):
                    print(f"--- CONFIRM Button Clicked (Key: {confirm_key}) ---")  # DEBUG
                    current_drawn_boxes = st.session_state.rects
                    current_scale = st.session_state.image_scale_factor
                    scaled_back_boxes = []  # Calculate scaled boxes
                    if abs(current_scale - 1.0) > 1e-6:
                        for box in current_drawn_boxes:
                            if isinstance(box, list) and len(box) == 4:
                                scaled_back_boxes.append([
                                    (int(round(x * current_scale)), int(round(y * current_scale)))
                                    for x, y in box])
                            else:
                                st.warning(f"Skipping invalid box during scaling: {box}")
                    else:
                        scaled_back_boxes = current_drawn_boxes

                    print(f"    Calling handle_confirm_annotation with path='{current_img_path}'")  # DEBUG
                    # Pass the displayed (possibly rotated) image
                    new_or_updated_schema = handle_confirm_annotation(
                        current_img_path,
                        scaled_back_boxes,
                        st.session_state.displayed_image
                    )
                    if new_or_updated_schema:
                        print("    handle_confirm_annotation successful, updating state.")  # DEBUG
                        st.session_state.schema = new_or_updated_schema
                        schema_changed_in_section = True
                    else:
                        print("    handle_confirm_annotation failed or returned None.")  # DEBUG

            with col2:
                # Generate Q/A button - Key uses image path
                qa_key = f"qa_btn_{current_img_path}"
                # print(f"    Rendering Generate Q/A button (Key: {qa_key})...") # DEBUG
                qa_button_disabled = current_schema is None or st.session_state.processing_qa

                # If we have a schema, add image selection option for QA generation
                if current_schema:
                    # Check if annotated image exists
                    annotated_img_path = get_annotated_image_path(current_img_path, current_img_stem)
                    has_annotated_image = annotated_img_path is not None

                    # Only show the option if an annotated image exists
                    if has_annotated_image:
                        st.radio(
                            "Image to send to AI:",
                            ["Original Image", "Annotated Image (with boxes)"],
                            key="image_choice_radio",
                            index=1 if st.session_state.use_annotated_image else 0,
                            horizontal=True,
                            on_change=lambda: setattr(st.session_state, "use_annotated_image",
                                                      st.session_state.image_choice_radio == "Annotated Image (with boxes)")
                        )
                    else:
                        # If no annotated image yet, force original and show info
                        st.info("Save annotation first to use annotated image with AI.")
                        st.session_state.use_annotated_image = False

                if qa_button_disabled:
                    st.caption("Confirm annotation first.")

                if st.button("🤖 Generate Q/A", type="secondary", use_container_width=True,
                             disabled=qa_button_disabled, key=qa_key):
                    if not qa_button_disabled:
                        print(f"--- QA Button Clicked (Key: {qa_key}) ---")  # DEBUG
                        # Call Gemini to get multiple QA pairs
                        handle_gemini_qa(
                            current_img_path,
                            current_schema,
                            st.session_state.use_annotated_image
                        )
                        # Force rerun to show QA pair selection UI (will be handled at the beginning of main())
                        st.rerun()

            # --- Rerun if schema changed by Confirm/QA/Edit in this section ---
            if schema_changed_in_section:
                print("--- Schema Changed from actions: Triggering Rerun ---")  # DEBUG
                st.rerun()

    else:
        # No image selected
        # print("--- No image selected, showing initial info message. ---") # DEBUG
        st.info("👈 Select an image from the sidebar to start annotating.")
        with st.expander("📘 How to use this app", expanded=True):  # Default expanded
            st.markdown("""
             ### Quick Guide

             1.  **Select an image** from the sidebar. Use search/filters. Annotation status (`✅`/`⚪`) is shown.
             2.  (Optional) **Rotate** the image using `🔄 Rotate`.
             3.  **Draw bounding boxes** (optional) on the canvas.
             4.  **Confirm** (`✅ Confirm`) to save boxes and create/update the `<filename>.json` schema and `<filename>.jpg` annotated image copy. Output folders mirror the input structure.
             5.  **Generate Q/A** (`🤖 Generate Q/A`) via Gemini (requires confirmed schema).
                - You can choose to use the original image or the annotated image with boxes.
                - You'll be able to select from multiple AI-generated QA pairs.
             6.  (Optional) **Edit schema fields** (`✏️ Edit Schema Values`) for more options including language settings, metadata, and tags.
             7.  (Optional) **Rename** original dataset files to UUIDs using the sidebar button (⚠️ **BACKUP FIRST**).
             """)

            # Show a screenshot or demo image
            # st.image("https://via.placeholder.com/800x400?text=Image+Annotater+Demo",
            #          caption="Image Annotater application workflow")


# --- Entry Point ---
if __name__ == "__main__":
    print("***** Application Starting *****")  # DEBUG
    ANNOT_ROOT.mkdir(parents=True, exist_ok=True)  # Ensure output root exists
    main()
