"""Sidebar file-explorer component with hierarchical view, search, filtering, and rename button."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, DefaultDict

import streamlit as st
# Use the functions from file_utils directly
from utils.file_utils import (
    list_images,
    derive_full_relative_path,
    get_annotated_image_stems,  # Use stem-based check
    rename_dataset_files_to_uuid  # Import renaming function
)
from utils.logger import get_logger

# Get logger for this module
logger = get_logger("sidebar")


def image_selector(search_term: str, selected_filter: str) -> None:
    """Displays hierarchical image list, handles selection, includes rename button."""

    # Initialize session state keys if they don't exist
    if "selected_image_path" not in st.session_state:
        st.session_state.selected_image_path = None

    logger.debug(f"Image selector called with search_term='{search_term}', filter='{selected_filter}'")

    # --- Display Utilities at the top ---
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Utilities")
    utilities_col1, utilities_col2 = st.sidebar.columns(2)

    with utilities_col1:
        rename_tooltip = "WARNING: Renames original dataset images and attempts to update annotations. BACK UP FIRST!"
        if st.button("Rename to UUIDs", help=rename_tooltip):
            logger.info("Rename to UUIDs button clicked")
            st.info("Starting renaming process... This may take a while.")
            progress_bar = st.sidebar.progress(0.0)
            try:
                # Call the renaming function from file_utils
                success, annot_updated, errors = rename_dataset_files_to_uuid(progress_bar)
                progress_bar.progress(1.0)  # Ensure progress bar reaches 100%
                if errors > 0:
                    logger.warning(f"Renaming completed with {errors} errors")
                    st.error(f"{errors} errors occurred during renaming. Check logs for details.")
                else:
                    logger.info(f"Renaming completed successfully: {success} images, {annot_updated} annotations")
                    st.success("Renaming completed successfully.")
                # Force a rerun to refresh the sidebar with new filenames
                st.info("Refreshing file list...")
                st.rerun()
            except Exception as e:
                logger.error(f"Error during renaming: {e}", exc_info=True)
                st.error(f"An error occurred during renaming: {e}")
                progress_bar.progress(1.0)  # Clear progress bar on error

    # Add another utility button in the second column if needed
    with utilities_col2:
        # Refresh button
        if st.button("Refresh List", help="Refresh the file list"):
            logger.debug("Refresh list button clicked")
            st.rerun()

    st.sidebar.markdown("---")

    # Get all images and annotated status (using stems)
    all_images = list_images()
    annotated_stems = get_annotated_image_stems()  # Get stems of annotated files
    logger.debug(f"Found {len(all_images)} total images, {len(annotated_stems)} annotated")

    # --- Pre-process images: Filter ---
    processed_images: List[Tuple[str, str, bool]] = []  # (img_path_str, img_stem, is_annotated)

    for img_path_str in all_images:
        img_path_obj = Path(img_path_str)
        img_stem = img_path_obj.stem

        # Apply search filter first
        if search_term and search_term.lower() not in img_path_str.lower():
            continue

        # Check annotation status using stem
        is_annotated = img_stem in annotated_stems

        # Apply annotation status filter
        filter_passed = (selected_filter == "All" or
                         (selected_filter == "Annotated" and is_annotated) or
                         (selected_filter == "Not Annotated" and not is_annotated))

        if filter_passed:
            processed_images.append((img_path_str, img_stem, is_annotated))

    # --- Group images by hierarchical path ---
    images_by_hierarchy: DefaultDict[str, List[Tuple[str, str, bool]]] = defaultdict(list)
    for img_path_str, img_stem, is_annotated in processed_images:
        hierarchy_key = derive_full_relative_path(img_path_str)
        if hierarchy_key == "(error_deriving_path)":
            logger.warning(f"Skipping image due to path derivation error: {img_path_str}")
            continue
        if hierarchy_key == "":
            hierarchy_key = "(Root Level)"

        images_by_hierarchy[hierarchy_key].append((img_path_str, img_stem, is_annotated))

    logger.debug(f"Grouped images into {len(images_by_hierarchy)} hierarchical groups")

    # --- Display Hierarchical Tree ---
    if not processed_images:
        logger.info("No images found matching criteria")
        st.sidebar.warning("No images found matching your criteria.")
    else:
        sorted_categories = sorted(images_by_hierarchy.keys())

        for category_path in sorted_categories:
            images_in_category = images_by_hierarchy[category_path]
            images_in_category.sort(key=lambda item: Path(item[0]).name)  # Sort by filename

            expander_label = f"{category_path} ({len(images_in_category)})"

            # Default to collapsed (expanded=False)
            with st.sidebar.expander(expander_label, expanded=False):
                for img_path_str, img_stem, is_annotated in images_in_category:
                    img_name = Path(img_path_str).name
                    status_icon = "✅" if is_annotated else "⚪"

                    # Use image path string in button key for uniqueness until renamed
                    button_key = f"btn_{img_path_str}"
                    if st.button(f"{status_icon} {img_name}", key=button_key, use_container_width=True):
                        logger.info(f"Image selected: {img_name}")
                        st.session_state.selected_image_path = img_path_str

    # --- Display Statistics Below Tree ---
    st.sidebar.markdown("---")  # Separator

    # Image count statistics
    st.sidebar.caption(f"Showing {len(processed_images)} of {len(all_images)} images")

    # Keyboard navigation hint
    with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
         - Use canvas toolbar for drawing/deleting boxes.
         - Use buttons for Confirm / Generate Q/A.
         """)
