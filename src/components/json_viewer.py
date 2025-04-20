"""JSON display with edit capabilities and syntax highlighting."""

import json
from typing import Any, Dict, Optional, List, Literal, Union

import streamlit as st
from pydantic import BaseModel  # Import BaseModel if schema objects are passed directly

from utils.schema_utils import VLMSFTData, LanguageInfo


def show_json(obj: Any, label: str = "Schema preview", editable: bool = False) -> Optional[Dict]:
    """Display JSON with syntax highlighting and optional editing capability.

    Args:
        obj: The object (dict or Pydantic model) to display as JSON.
        label: Label for the text area/expander.
        editable: Whether to allow editing the JSON (Not recommended for complex schemas).

    Returns:
        Updated JSON object as dict if edited, otherwise None.
    """
    txt = ""
    try:
        # Handle Pydantic models using model_dump
        if isinstance(obj, BaseModel):
            # Convert Pydantic model to dict first, then to JSON string
            data_dict = obj.model_dump(mode='json')  # mode='json' handles types like datetime
            txt = json.dumps(data_dict, indent=2, ensure_ascii=False)
        elif isinstance(obj, dict):
            # Dump dict directly
            txt = json.dumps(obj, indent=2, ensure_ascii=False)
        elif isinstance(obj, str):
            # Try to parse/reformat if it's already a JSON string
            try:
                parsed = json.loads(obj)
                txt = json.dumps(parsed, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                txt = obj  # Display as is if not valid JSON string
        else:
            # Fallback for other types
            txt = json.dumps(obj, indent=2, ensure_ascii=False, default=str)  # Add default=str

    except Exception as e:
        st.error(f"Error formatting object as JSON: {str(e)}")
        txt = str(obj)  # Display raw string representation on error

    # Basic syntax highlighting (can be improved)
    # ... (keep existing highlighting logic or remove if using st.code)

    if not editable:
        with st.expander(label, expanded=True):
            # Use st.code for better built-in JSON display
            st.code(txt, language="json")
        return None  # Not editable, return None

    # --- Editing Logic (Simplified - Use interactive_json_editor instead) ---
    # Direct text area editing is prone to errors for complex schemas.
    # Recommend using the field-based editor below or disabling direct edit.
    st.warning("Direct JSON editing is disabled. Use 'Edit Schema Values' expander.")
    with st.expander(label, expanded=True):
        st.code(txt, language="json")
    # edited_txt = st.text_area(label, txt, height=400, key=f"json_edit_{label}")
    # if edited_txt != txt:
    #     try:
    #         return json.loads(edited_txt)
    #     except json.JSONDecodeError:
    #         st.error("Invalid JSON syntax in text area.")
    #         return None # Return None on error
    return None


def interactive_json_editor(schema_model: VLMSFTData, key: str = "json_editor") -> Optional[VLMSFTData]:
    """Interactive JSON editor using Pydantic model fields.

    Args:
        schema_model: The Pydantic VLMSFTData object to edit.
        key: A unique key prefix for Streamlit components.

    Returns:
        Updated schema object if changed and validated, otherwise None.
    """
    edited = False
    # Work on a copy to compare changes
    updated_data = schema_model.model_dump()  # Get data as dict

    with st.expander("✏️ Edit Schema Values", expanded=False):
        st.caption("Edit individual fields. Changes are saved on Confirm or Generate Q/A.")

        # Use tabs to organize the editor into logical sections
        tab1, tab2, tab3 = st.tabs(["Basic Info", "Text & Answers", "Metadata"])

        with tab1:
            # Task type, difficulty, source, split
            task_options = ["captioning", "vqa", "instruction"]
            current_task = updated_data.get("task_type", "vqa")
            new_task = st.selectbox("Task Type",
                                    task_options,
                                    index=task_options.index(current_task) if current_task in task_options else 0,
                                    key=f"{key}_task_type")
            if new_task != current_task:
                updated_data["task_type"] = new_task
                edited = True

            # Source
            source_val = updated_data.get("source", "Image_Annotater")
            new_source = st.text_input("Source", value=source_val, key=f"{key}_source")
            if new_source != source_val:
                updated_data["source"] = new_source
                edited = True

            # Difficulty
            diff_options = ["easy", "medium", "hard"]
            current_diff = updated_data.get("difficulty", "medium")
            new_diff = st.selectbox("Difficulty",
                                    diff_options,
                                    index=diff_options.index(current_diff) if current_diff in diff_options else 1,
                                    key=f"{key}_difficulty")
            if new_diff != current_diff:
                updated_data["difficulty"] = new_diff
                edited = True

            # Split
            split_options = ["train", "val", "test"]
            current_split = updated_data.get("split", "train")
            new_split = st.selectbox("Split",
                                     split_options,
                                     index=split_options.index(current_split) if current_split in split_options else 0,
                                     key=f"{key}_split")
            if new_split != current_split:
                updated_data["split"] = new_split
                edited = True

            # Tags
            tags_str = ", ".join(updated_data.get("tags", []))
            new_tags_str = st.text_input("Tags (comma-separated)", value=tags_str, key=f"{key}_tags")
            if new_tags_str != tags_str:
                updated_data["tags"] = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
                edited = True

        with tab2:
            # Text fields - text_en, text_ms, answer_en, answer_ms
            text_en = updated_data.get("text_en", "")
            new_text_en = st.text_area("Text (English)", value=text_en, key=f"{key}_text_en")
            if new_text_en != text_en:
                updated_data["text_en"] = new_text_en
                edited = True

            text_ms = updated_data.get("text_ms", "")
            new_text_ms = st.text_area("Text (Malay)", value=text_ms, key=f"{key}_text_ms")
            if new_text_ms != text_ms:
                updated_data["text_ms"] = new_text_ms
                edited = True

            answer_en = updated_data.get("answer_en", "")
            new_answer_en = st.text_area("Answer (English)", value=answer_en, key=f"{key}_answer_en")
            if new_answer_en != answer_en:
                updated_data["answer_en"] = new_answer_en
                edited = True

            answer_ms = updated_data.get("answer_ms", "")
            new_answer_ms = st.text_area("Answer (Malay)", value=answer_ms, key=f"{key}_answer_ms")
            if new_answer_ms != answer_ms:
                updated_data["answer_ms"] = new_answer_ms
                edited = True

        with tab3:
            # Metadata - annotator_id, language settings
            # Get current metadata or create empty dict
            metadata = updated_data.get("metadata", {})

            # Annotator ID
            annotator_id = metadata.get("annotator_id", "")
            new_annotator_id = st.text_input("Annotator ID", value=annotator_id, key=f"{key}_annotator_id")
            if new_annotator_id != annotator_id:
                metadata["annotator_id"] = new_annotator_id
                updated_data["metadata"] = metadata
                edited = True

            # License
            license_val = metadata.get("license", "CC-BY")
            new_license = st.text_input("License", value=license_val, key=f"{key}_license")
            if new_license != license_val:
                metadata["license"] = new_license
                updated_data["metadata"] = metadata
                edited = True

            # Language Quality Score
            quality_score = metadata.get("language_quality_score", 0)
            new_quality_score = st.slider("Language Quality Score",
                                          min_value=0.0,
                                          max_value=5.0,
                                          value=float(quality_score if quality_score is not None else 0.0),
                                          step=0.1,
                                          key=f"{key}_quality_score")
            if new_quality_score != quality_score:
                metadata["language_quality_score"] = new_quality_score
                updated_data["metadata"] = metadata
                edited = True

            # Language settings
            st.subheader("Language Settings")

            # Get current language info
            language_info = updated_data.get("language", {})

            # Language source options
            lang_source_options = [
                ["ms", "en"],  # Both languages
                ["ms"],  # Malay only
                ["en"]  # English only
            ]

            # Convert current source to list if it's not already
            current_source = language_info.get("source", ["ms", "en"])
            # Find the index in options list
            current_source_index = 0
            for i, opt in enumerate(lang_source_options):
                if sorted(opt) == sorted(current_source):
                    current_source_index = i
                    break

            source_labels = ["Both (ms, en)", "Malay (ms) only", "English (en) only"]
            selected_source_label = st.radio("Source Language",
                                             source_labels,
                                             index=current_source_index,
                                             key=f"{key}_lang_source")

            # Convert the selected label back to an index
            new_source_index = source_labels.index(selected_source_label)

            if new_source_index != current_source_index:
                language_info["source"] = lang_source_options[new_source_index]
                updated_data["language"] = language_info
                edited = True

            # Language target options (similar to source)
            if "target" in language_info and language_info["target"]:
                current_target = language_info.get("target", ["ms", "en"])
                current_target_index = 0
                for i, opt in enumerate(lang_source_options):
                    if sorted(opt) == sorted(current_target):
                        current_target_index = i
                        break
            else:
                current_target_index = 0  # Default to both

            target_labels = ["Both (ms, en)", "Malay (ms) only", "English (en) only", "None"]
            selected_target_label = st.radio("Target Language",
                                             target_labels,
                                             index=current_target_index if "target" in language_info and language_info[
                                                 "target"] else 3,
                                             key=f"{key}_lang_target")

            # Handle target language setting
            if selected_target_label != "None":
                # Convert the selected label back to an index
                new_target_index = target_labels.index(selected_target_label)
                if "target" not in language_info or language_info.get("target") is None or \
                        sorted(language_info.get("target", [])) != sorted(lang_source_options[new_target_index]):
                    language_info["target"] = lang_source_options[new_target_index]
                    updated_data["language"] = language_info
                    edited = True
            elif "target" in language_info and language_info["target"] is not None:
                # Set target to None if "None" selected
                language_info["target"] = None
                updated_data["language"] = language_info
                edited = True

    if edited:
        try:
            # Re-validate the updated dictionary using the Pydantic model
            validated_schema = VLMSFTData.model_validate(updated_data)
            return validated_schema  # Return the validated Pydantic object
        except Exception as e:  # Catch Pydantic ValidationError
            st.error(f"Schema validation error after edit: {e}")
            return None  # Return None if validation fails
    else:
        return None  # No changes detected
