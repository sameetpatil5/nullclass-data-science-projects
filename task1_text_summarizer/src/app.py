import streamlit as st
from text_summarizer import summarize_text

# Title
st.title("Text Summarizer")

# Input container
input_text_box_container = st.container(border=True)

# Text area for input
input_text_box = input_text_box_container.text_area(
    "Input text to summarize", height=300
)

# Button to summarize text
summarize_btn = st.button("Summarize")

# Summarize the text
if summarize_btn:
    if input_text_box.strip():
        summary = summarize_text(input_text_box)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.warning("Please provide some text to summarize.")
