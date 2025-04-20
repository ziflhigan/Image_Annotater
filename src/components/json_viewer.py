"""JSON display with edit capabilities and syntax highlighting."""

import json
from typing import Any, Dict, Optional, List, Callable

import streamlit as st
from pydantic import BaseModel
from utils.ai_utils import GeminiQA
from utils.logger import get_logger
from utils.schema_utils import VLMSFTData

logger = get_logger("json_viewer")


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
        logger.error(f"Error formatting object as JSON: {e}", exc_info=True)
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
            logger.info("Schema edited and validated successfully")
            return validated_schema  # Return the validated Pydantic object
        except Exception as e:  # Catch Pydantic ValidationError
            logger.error(f"Schema validation error after edit: {e}", exc_info=True)
            st.error(f"Schema validation error after edit: {e}")
            return None  # Return None if validation fails
    else:
        return None  # No changes detected


def qa_card_selector(qa_pairs: List[GeminiQA], on_select_callback: Callable[[GeminiQA], None]) -> None:
    """Display QA pairs as horizontal cards with individual selection buttons.

    Args:
        qa_pairs: List of GeminiQA objects to display
        on_select_callback: Callback function that takes the selected GeminiQA
    """
    if not qa_pairs:
        st.error("No QA pairs available to select from.")
        return

    st.subheader("🤖 Select a Question-Answer Pair")
    st.info("Choose one of the AI-generated QA pairs to add to your annotation:")

    # Add custom CSS for card layout
    st.markdown("""
    <style>
    .qa-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: flex-start;
        margin-bottom: 20px;
    }
    .qa-card-wrapper {
        flex: 0 0 calc(33.333% - 15px);
        margin-bottom: 15px;
        display: flex;
        flex-direction: column;
    }
    .qa-card {
        position: relative;
        border: 1px solid #ddd;
        border-radius: 10px 10px 0 0;
        padding: 15px;
        background-color: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
        flex-grow: 1;
        height: 100%;
    }
    .qa-card:hover {
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .qa-card-button {
        border: 1px solid #ddd;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 8px 0;
        background-color: #f8f9fa;
        text-align: center;
        font-weight: bold;
        cursor: pointer;
        transition: background-color 0.2s;
    }
    .qa-card-button:hover {
        background-color: #e9ecef;
    }
    .qa-card-header {
        font-weight: bold;
        border-bottom: 1px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 10px;
        color: #333;
    }
    .qa-card-tag {
        display: inline-block;
        background-color: #f0f0f0;
        border-radius: 15px;
        padding: 2px 8px;
        margin-right: 5px;
        margin-bottom: 5px;
        font-size: 0.8em;
        color: #555;
    }
    .qa-card-difficulty-easy {
        color: #28a745;
    }
    .qa-card-difficulty-medium {
        color: #fd7e14;
    }
    .qa-card-difficulty-hard {
        color: #dc3545;
    }
    .qa-card-score {
        position: absolute;
        top: 15px;
        right: 15px;
        background-color: #f8f9fa;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .qa-content {
        font-size: 0.9em;
    }
    @media (max-width: 1200px) {
        .qa-card-wrapper {
            flex: 0 0 calc(50% - 15px);
        }
    }
    @media (max-width: 768px) {
        .qa-card-wrapper {
            flex: 0 0 100%;
        }
    }
    </style>
    """, unsafe_allow_html=True)

    # Create a responsive grid layout using Streamlit columns
    # Determine number of columns based on number of QA pairs
    num_cols = min(3, len(qa_pairs))
    if num_cols == 0:
        return

    # Use Streamlit's column system which works more reliably
    cols = st.columns(num_cols)

    # Distribute cards among columns
    for i, qa in enumerate(qa_pairs):
        col_idx = i % num_cols

        with cols[col_idx]:
            # Determine difficulty color and CSS class
            difficulty_class = qa.difficulty
            difficulty_color = {
                "easy": "#28a745",
                "medium": "#fd7e14",
                "hard": "#dc3545"
            }.get(difficulty_class, "#333333")

            # Calculate score color (red to green gradient)
            score = qa.language_quality_score

            # Create card with styled elements
            st.markdown(f"""
            <div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 15px; position: relative; background: white;">
                <div style="position: absolute; top: 15px; right: 15px; background: #f8f9fa; border-radius: 50%; width: 40px; height: 40px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: {'green' if score > 3.5 else 'orange' if score > 2 else 'red'};">
                    {score}
                </div>
                <div style="font-weight: bold; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 10px;">
                    {qa.task_type.upper()} <span style="color: {difficulty_color};">({difficulty_class})</span>
                </div>
                <div style="font-size: 0.9em; margin-bottom: 10px;">
                    <div><strong>🇬🇧 Q:</strong> {qa.text_en[:100] + '...' if len(qa.text_en) > 100 else qa.text_en}</div>
                    <div><strong>🇲🇾 Q:</strong> {qa.text_ms[:100] + '...' if len(qa.text_ms) > 100 else qa.text_ms}</div>
                </div>
                <div style="font-size: 0.9em; margin-bottom: 10px;">
                    <div><strong>🇬🇧 A:</strong> {qa.answer_en[:100] + '...' if len(qa.answer_en) > 100 else qa.answer_en}</div>
                    <div><strong>🇲🇾 A:</strong> {qa.answer_ms[:100] + '...' if len(qa.answer_ms) > 100 else qa.answer_ms}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Standard Streamlit button that's fully visible and functional
            if st.button(f"Use This QA", key=f"use_qa_{i}", use_container_width=True):
                logger.info(f"QA pair {i + 1} selected: {qa.task_type} ({qa.difficulty})")
                on_select_callback(qa)

    # Add detailed view of each QA pair in expandable sections
    for i, qa in enumerate(qa_pairs):
        with st.expander(f"Details for QA #{i + 1}: {qa.task_type.upper()} ({qa.difficulty})"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### English")
                st.markdown(f"**Question:** {qa.text_en}")
                st.markdown(f"**Answer:** {qa.answer_en}")

            with col2:
                st.markdown("#### Malay")
                st.markdown(f"**Question:** {qa.text_ms}")
                st.markdown(f"**Answer:** {qa.answer_ms}")

            st.markdown(f"**Tags:** {', '.join(qa.tags) if qa.tags else 'None'}")
            st.markdown(f"**Quality Score:** {qa.language_quality_score}/5")
