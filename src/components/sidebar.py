"""Enhanced sidebar file-explorer component with search and filtering."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Dict, Set

import streamlit as st

from utils.file_utils import list_images, derive_category, ANNOT_ROOT


def get_annotated_images() -> Set[str]:
    """Returns set of image ids that have been annotated already."""
    annotated = set()
    if not ANNOT_ROOT.exists():
        return annotated

    # Check every schema folder for existing JSONs
    for schema_dir in ANNOT_ROOT.glob("schema_*"):
        if schema_dir.is_dir():
            for json_file in schema_dir.glob("*.json"):
                # Store just the image id (filename without extension)
                annotated.add(json_file.stem)

    return annotated


def image_selector() -> None:  # Changed return type to None
    """Enhanced sidebar with search, filtering, and annotation status indicators.
    Updates session state directly when a button is clicked.
    """
    st.sidebar.header("Dataset Images")

    # Initialize the session state key if it doesn't exist
    if "selected_image_path" not in st.session_state:
        st.session_state.selected_image_path = None

    # Search box
    search_term = st.sidebar.text_input("🔍 Search images", key="search_images")

    # Filter options
    filter_options = ["All", "Annotated", "Not Annotated"]
    selected_filter = st.sidebar.radio("Filter by status:", filter_options, horizontal=True, key="filter_status")

    # Get all images and annotated status
    all_images = list_images()
    annotated_ids = get_annotated_images()

    # Apply search filter if provided
    if search_term:
        all_images = [img for img in all_images if search_term.lower() in img.lower()]

    # Apply annotation status filter
    filtered_images = []
    for img_path_str in all_images:  # Iterate over strings
        img_id = Path(img_path_str).stem
        is_annotated = img_id in annotated_ids

        if selected_filter == "All" or \
                (selected_filter == "Annotated" and is_annotated) or \
                (selected_filter == "Not Annotated" and not is_annotated):
            filtered_images.append((img_path_str, is_annotated))  # Store path as string

    # Group by category for better organization
    images_by_category: Dict[str, List[tuple]] = {}
    for img_path_str, is_annotated in filtered_images:
        category = derive_category(img_path_str)  # Use string path
        if category not in images_by_category:
            images_by_category[category] = []
        images_by_category[category].append((img_path_str, is_annotated))

    # Display message if no images match filters
    if not filtered_images:
        st.sidebar.warning("No images found matching your criteria.")
        # Do not return None here, let main handle no selection

    # Display images grouped by category with annotation status indicators
    # selected_image = None # We won't return this directly

    for category, images in sorted(images_by_category.items()):  # Sort categories
        # Use the category string (which is now a path) directly in the label
        # Use an f-string to ensure it's treated as a string
        expander_label = f"{category} ({len(images)})"
        with st.sidebar.expander(expander_label, expanded=True):
            # Sort images within category by name
            sorted_images = sorted(images, key=lambda item: os.path.basename(item[0]))
            for img_path_str, is_annotated in sorted_images:
                img_name = os.path.basename(img_path_str)
                status_icon = "✅ " if is_annotated else "🔘 "
                if st.button(f"{status_icon}{img_name}", key=f"btn_{img_path_str}", use_container_width=True):
                    st.session_state.selected_image_path = img_path_str
                    # No rerun here, main loop handles state change

    # REMOVED the selectbox to simplify and avoid conflicts
    # choice = st.sidebar.selectbox(...)

    # Display image count statistics
    st.sidebar.caption(f"Showing {len(filtered_images)} of {len(all_images)} images")

    # Keyboard navigation hint
    with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
        - `Esc`: Deselect/cancel current action
        - `Del`: Delete selected rectangle (Use toolbar)
        - `Ctrl+S`: Save annotation (Use Confirm button)
        - `Ctrl+G`: Generate Q/A (Use Generate button)
        """)

    # No return value needed, selection is handled via session state
