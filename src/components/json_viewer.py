"""Enhanced JSON display with edit capabilities and syntax highlighting."""

import json
from typing import Any, Dict, Optional

import streamlit as st


def show_json(obj: Any, label: str = "Schema preview", editable: bool = False) -> Optional[Dict]:
    """Display JSON with syntax highlighting and optional editing capability.
    
    Args:
        obj: The object to display as JSON
        label: Label for the text area
        editable: Whether to allow editing the JSON
        
    Returns:
        Updated JSON object if edited, otherwise None
    """
    # Convert to pretty-printed JSON
    try:
        if isinstance(obj, str):
            # Try to parse if it's already a string
            parsed = json.loads(obj)
            txt = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            # Otherwise, just dump the object
            txt = json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error formatting JSON: {str(e)}")
        txt = str(obj)
    
    # Add syntax highlighting (basic)
    highlighted = txt.replace(
        '"', '<span style="color: #a31515">"</span>'
    ).replace(
        ',', '<span style="color: #000000">,</span>'
    ).replace(
        ':', '<span style="color: #0451a5">:</span>'
    ).replace(
        '{', '<span style="color: #000000">{</span>'
    ).replace(
        '}', '<span style="color: #000000">}</span>'
    ).replace(
        '[', '<span style="color: #000000">[</span>'
    ).replace(
        ']', '<span style="color: #000000">]</span>'
    )
    
    # Display with syntax highlighting if not editable
    if not editable:
        with st.expander(label, expanded=True):
            st.code(txt, language="json")
        return None
    
    # Show as editable text area if requested
    edited_txt = st.text_area(label, txt, height=400)
    
    # Return parsed object if edited
    if edited_txt != txt:
        try:
            return json.loads(edited_txt)
        except json.JSONDecodeError:
            st.error("Invalid JSON syntax")
            return None
    
    return None


def interactive_json_editor(schema: Dict, key: str = "json_editor") -> Optional[Dict]:
    """Interactive JSON editor with field-by-field editing capabilities.
    
    Args:
        schema: The schema dictionary to edit
        key: A unique key for Streamlit components
        
    Returns:
        Updated schema if changed, otherwise None
    """
    edited = False
    updated_schema = schema.copy()
    
    with st.expander("Edit Schema Values", expanded=False):
        st.caption("Edit individual fields without changing the structure")
        
        # Handle top-level fields, excluding nested objects
        for field, value in schema.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                new_value = None
                
                # Create appropriate input widget based on value type
                if isinstance(value, bool):
                    new_value = st.checkbox(field, value, key=f"{key}_{field}")
                elif isinstance(value, int):
                    new_value = st.number_input(field, value=value, key=f"{key}_{field}")
                elif isinstance(value, float):
                    new_value = st.number_input(field, value=value, step=0.1, key=f"{key}_{field}")
                elif isinstance(value, str):
                    new_value = st.text_input(field, value, key=f"{key}_{field}")
                elif value is None:
                    if st.checkbox(f"Set value for {field}", False, key=f"{key}_{field}_check"):
                        new_value = st.text_input(f"{field} value", "", key=f"{key}_{field}_value")
                
                # Check if value changed
                if new_value is not None and new_value != value:
                    updated_schema[field] = new_value
                    edited = True
        
        # Handle simple lists (of strings)
        for field, value in schema.items():
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                st.subheader(f"{field} (list)")
                tags_str = ", ".join(value)
                new_tags = st.text_input(f"Enter comma-separated values", 
                                        tags_str, 
                                        key=f"{key}_{field}_list")
                
                if new_tags != tags_str:
                    # Split by comma and strip whitespace
                    updated_schema[field] = [tag.strip() for tag in new_tags.split(",") if tag.strip()]
                    edited = True
    
    # Return updated schema if changed, otherwise None
    if edited:
        return updated_schema
    return None