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


# No longer need uuid here unless for the button logic, keep it for now


# No longer need get_annotated_image_uuids

def image_selector(search_term: str, selected_filter: str) -> None:
    """Displays hierarchical image list, handles selection, includes rename button."""

    # Initialize session state keys if they don't exist
    if "selected_image_path" not in st.session_state:
        st.session_state.selected_image_path = None
    # No longer need UUID state here
    # if "image_path_to_uuid" not in st.session_state: ...
    # if "current_image_uuid" not in st.session_state: ...

    print("--- image_selector --- Start")  # DEBUG
    print(f"    Received search_term='{search_term}', selected_filter='{selected_filter}'")  # DEBUG

    # Get all images and annotated status (using stems)
    all_images = list_images()
    annotated_stems = get_annotated_image_stems()  # Get stems of annotated files

    # --- Pre-process images: Filter ---
    # No UUID generation here anymore
    processed_images: List[Tuple[str, str, bool]] = []  # (img_path_str, img_stem, is_annotated)

    print(f"    Processing {len(all_images)} total images found by list_images().")  # DEBUG
    for i, img_path_str in enumerate(all_images):
        img_path_obj = Path(img_path_str)
        img_stem = img_path_obj.stem

        # Apply search filter first
        if search_term and search_term.lower() not in img_path_str.lower():
            continue

        # Check annotation status using stem
        is_annotated = img_stem in annotated_stems
        # print(f"    Image: {img_path_obj.name}, Stem: {img_stem}, Annotated: {is_annotated}") # DEBUG

        # Apply annotation status filter
        filter_passed = (selected_filter == "All" or
                         (selected_filter == "Annotated" and is_annotated) or
                         (selected_filter == "Not Annotated" and not is_annotated))

        if filter_passed:
            processed_images.append((img_path_str, img_stem, is_annotated))

    # --- Group images by hierarchical path ---
    images_by_hierarchy: DefaultDict[str, List[Tuple[str, str, bool]]] = defaultdict(list)
    # print("\n    Grouping images by hierarchy...") # DEBUG
    for img_path_str, img_stem, is_annotated in processed_images:
        hierarchy_key = derive_full_relative_path(img_path_str)
        # print(f"        Grouping: Path='{img_path_str}', Key='{hierarchy_key}'") # DEBUG
        if hierarchy_key == "(error_deriving_path)":
            print(f"        WARNING: Skipping image due to path derivation error: {img_path_str}")
            continue
        if hierarchy_key == "":
            hierarchy_key = "(Root Level)"

        images_by_hierarchy[hierarchy_key].append((img_path_str, img_stem, is_annotated))

    # print(f"\n    Found {len(images_by_hierarchy)} hierarchical groups.") # DEBUG

    # --- Display Hierarchical Tree ---
    if not processed_images:
        st.sidebar.warning("No images found matching your criteria.")
    else:
        sorted_categories = sorted(images_by_hierarchy.keys())
        # print("\n    Displaying expanders...") # DEBUG

        for category_path in sorted_categories:
            images_in_category = images_by_hierarchy[category_path]
            images_in_category.sort(key=lambda item: Path(item[0]).name)  # Sort by filename

            expander_label = f"{category_path} ({len(images_in_category)})"
            # print(f"    Creating Expander: '{expander_label}'") # DEBUG
            # Determine if expander should be expanded by default (e.g., if it contains the selected image)
            # Default to True for simplicity now, could be made smarter
            is_expanded = True
            # Check if the currently selected image belongs to this category path
            # This logic needs refinement if selection persistence across runs is needed
            # if st.session_state.selected_image_path:
            #      selected_cat_path = derive_full_relative_path(st.session_state.selected_image_path)
            #      if selected_cat_path == "" : selected_cat_path = "(Root Level)"
            #      is_expanded = (category_path == selected_cat_path)

            with st.sidebar.expander(expander_label, expanded=is_expanded):  # Default expanded
                for img_path_str, img_stem, is_annotated in images_in_category:
                    img_name = Path(img_path_str).name
                    status_icon = "✅" if is_annotated else "⚪"

                    # Use image path string in button key for uniqueness until renamed
                    button_key = f"btn_{img_path_str}"
                    if st.button(f"{status_icon} {img_name}", key=button_key, use_container_width=True):
                        print(f"\n--- BUTTON CLICKED ---")  # DEBUG
                        print(f"    Button Key: {button_key}")  # DEBUG
                        print(f"    Image Name: {img_name}")  # DEBUG
                        print(f"    Assigning selected_image_path = '{img_path_str}'")  # DEBUG
                        st.session_state.selected_image_path = img_path_str
                        # No UUID assignment needed here anymore
                        # st.session_state.current_image_uuid = img_uuid
                        # ...

                        print(f"    State AFTER click: selected_path='{st.session_state.selected_image_path}'")  # DEBUG

    # --- Display Utilities Below Tree ---
    st.sidebar.markdown("---")  # Separator

    # Image count statistics
    st.sidebar.caption(f"Showing {len(processed_images)} of {len(all_images)} images")

    # Keyboard navigation hint
    with st.sidebar.expander("⌨️ Keyboard Shortcuts"):
        st.markdown("""
         - Use canvas toolbar for drawing/deleting boxes.
         - Use buttons for Confirm / Generate Q/A.
         """)

    # --- Renaming Button ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dataset Utilities")
    st.sidebar.warning("Use with caution!")
    if st.sidebar.button("🧬 Rename Dataset Files to UUIDs",
                         help="WARNING: Renames original dataset images and attempts to update annotations. "
                              "BACK UP FIRST!"):
        print("--- Rename Button Clicked ---")  # DEBUG
        st.sidebar.info("Starting renaming process... This may take a while.")
        progress_bar = st.sidebar.progress(0.0)
        try:
            # Call the renaming function from file_utils
            success, annot_updated, errors = rename_dataset_files_to_uuid(progress_bar)
            progress_bar.progress(1.0)  # Ensure progress bar reaches 100%
            st.sidebar.success(f"Renaming finished! {success} images renamed, {annot_updated} annotations updated.")
            if errors > 0:
                st.sidebar.error(f"{errors} errors occurred during renaming. Check console logs.")
            else:
                st.sidebar.success("Renaming completed successfully.")
            # Force a rerun to refresh the sidebar with new filenames
            st.sidebar.info("Refreshing file list...")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"An error occurred during renaming: {e}")
            progress_bar.progress(1.0)  # Clear progress bar on error

    print("--- image_selector --- End")  # DEBUG
