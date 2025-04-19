"""JSON display with edit capabilities and syntax highlighting."""

import json
from typing import Any, Dict, Optional, List, Literal, Union

import streamlit as st
from pydantic import BaseModel  # Import BaseModel if schema objects are passed directly

from utils.schema_utils import FixedSchema


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


def interactive_json_editor(schema_model: FixedSchema, key: str = "json_editor") -> Optional[FixedSchema]:
    """Interactive JSON editor using Pydantic model fields.

    Args:
        schema_model: The Pydantic FixedSchema object to edit.
        key: A unique key prefix for Streamlit components.

    Returns:
        Updated schema object if changed and validated, otherwise None.
    """
    edited = False
    # Work on a copy to compare changes
    updated_data = schema_model.model_dump()  # Get data as dict

    with st.expander("✏️ Edit Schema Values", expanded=False):
        st.caption("Edit individual fields. Changes are saved on Confirm or Generate Q/A.")

        # Iterate through fields defined in the Pydantic model for type hints
        for field_name, field_info in schema_model.model_fields.items():
            current_value = updated_data.get(field_name)

            # Skip complex fields like bounding_box, language, metadata for this simple editor
            # (Metadata sub-fields could be handled similarly if needed)
            if field_name in ["bounding_box", "language", "metadata", "image_id", "image_path", "source"]:
                continue  # Skip non-editable or complex fields here

            label = f"{field_name.replace('_', ' ').capitalize()}"
            widget_key = f"{key}_{field_name}"
            new_value = current_value  # Initialize with current value

            # --- Create appropriate input widget based on type ---
            field_type = field_info.annotation

            # Handle Optional types
            is_optional = getattr(field_type, '__origin__', None) is Optional or \
                          getattr(field_type, '__origin__', None) is Union and \
                          type(None) in getattr(field_type, '__args__', [])

            actual_type = field_type
            if is_optional:
                # Get the non-None type from Optional[T] or Union[T, None]
                non_none_args = [arg for arg in getattr(field_type, '__args__', []) if arg is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]

            # Literal dropdowns (task_type, difficulty, split)
            if getattr(actual_type, '__origin__', None) is Literal:
                options = list(getattr(actual_type, '__args__', []))
                # Find index, default to 0 if current_value is not in options
                try:
                    current_index = options.index(current_value) if current_value in options else 0
                except ValueError:
                    current_index = 0  # Fallback if value isn't valid literal
                new_value = st.selectbox(label, options, index=current_index, key=widget_key)

            # String fields (text_ms, answer_ms, etc.)
            elif actual_type is str:
                # Use text_area for potentially longer text
                if field_name in ["text_ms", "answer_ms", "text_en", "answer_en"]:
                    new_value = st.text_area(label, value=current_value or "", key=widget_key)
                else:  # Use text_input for shorter strings like tags
                    new_value = st.text_input(label, value=current_value or "", key=widget_key)

            # List of strings (tags)
            elif field_type == List[str] or actual_type == list[str]:  # Check both notations
                tags_str = ", ".join(current_value or [])
                new_tags_str = st.text_input(label + " (comma-separated)", value=tags_str, key=widget_key)
                if new_tags_str != tags_str:
                    new_value = [tag.strip() for tag in new_tags_str.split(",") if tag.strip()]
                else:
                    new_value = current_value  # No change

            # Add other type handlers if needed (int, float, bool)
            # Example:
            # elif actual_type is float:
            #     new_value = st.number_input(label, value=float(current_value or 0.0), key=widget_key)
            # elif actual_type is int:
            #      new_value = st.number_input(label, value=int(current_value or 0), step=1, key=widget_key)
            # elif actual_type is bool:
            #      new_value = st.checkbox(label, value=bool(current_value or False), key=widget_key)

            # --- Check if value changed ---
            # Need careful comparison, especially for lists/empty strings
            if new_value != current_value:
                # Handle case where empty text area becomes None vs empty string
                if isinstance(new_value, str) and not new_value and current_value is None:
                    pass  # Treat empty string input as None if original was None
                elif new_value != current_value:
                    updated_data[field_name] = new_value
                    edited = True

    if edited:
        try:
            # Re-validate the updated dictionary using the Pydantic model
            validated_schema = FixedSchema.model_validate(updated_data)
            return validated_schema  # Return the validated Pydantic object
        except Exception as e:  # Catch Pydantic ValidationError
            st.error(f"Schema validation error after edit: {e}")
            return None  # Return None if validation fails
    else:
        return None  # No changes detected
