import streamlit as st
import fitz
from utils import chunk_text
from embeddings import create_embeddings
from vectordb import store_embeddings

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

    # Create Chunks
    chunks = chunk_text(text)

    st.write("## Number of Chunks")
    st.success(len(chunks))

    st.write("## First 5 Chunks")

    for i, chunk in enumerate(chunks[:5]):
        st.write(f"### Chunk {i+1}")
        st.write(chunk)

    # Create Embeddings
        # Create Embeddings
    embeddings = create_embeddings(chunks)

    try:
        collection = store_embeddings(chunks, embeddings)

        st.success("Embeddings stored successfully!")

        st.write("Total Records:", collection.count())

    except Exception as e:
        st.error(f"Error: {e}")

    st.write("## Embedding Information")

    st.success(f"Total Embeddings: {len(embeddings)}")

    st.write("Embedding Dimension:", len(embeddings[0]))

    st.write("First 10 Values of First Embedding:")

    st.write(embeddings[0][:10])

    st.write("## ChromaDB")

    st.success("Embeddings Stored Successfully! ✅")

    st.write("Total Records:", collection.count())