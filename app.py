import streamlit as st
import fitz

st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Legal Assistant")
st.write("Welcome to our RAG Project!")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success("PDF Uploaded Successfully ✅")

    st.write("### File Details")
    st.write("Filename:", uploaded_file.name)
    st.write("File Size:", uploaded_file.size, "bytes")

    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")

    text = ""

    for page in pdf:
        text += page.get_text()

    st.write("## Extracted Text")
    st.text(text[:3000])