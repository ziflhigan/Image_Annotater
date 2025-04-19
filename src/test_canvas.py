# test_canvas.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import os

# Create a dummy image path (replace with a real one)
image_path = "dataset/Buildings/8dfb82916f8f4484e66ff2c58126f90.jpg"

if os.path.exists(image_path):
    img = Image.open(image_path)
    st.write(f"Image loaded: {img.width}x{img.height}")

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.1)",
        stroke_width=3,
        stroke_color="#FF0000",
        background_image=img,
        height=img.height,
        width=img.width,
        drawing_mode="rect",
        key="canvas_test",
    )
    st.write("Canvas rendered.")
    if canvas_result.json_data:
        st.json(canvas_result.json_data)
else:
    st.error(f"Test image not found at: {image_path}")