import streamlit as st

st.set_page_config(page_title="Document Classification", page_icon="🙈")

st.title("Document Classification")

allowed_types = [
    "pdf",
]
uploaded_file = st.file_uploader(f"Upload a file {allowed_types} to classify.", type=allowed_types)

