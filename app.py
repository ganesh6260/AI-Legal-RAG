import streamlit as st
import fitz
from utils import chunk_text
from embeddings import create_embeddings
from vectordb import store_embeddings
from retriever import search_documents

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

    # Read PDF
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
    embeddings = create_embeddings(chunks)

    try:
        collection = store_embeddings(chunks, embeddings)

        st.success("Embeddings Stored Successfully! ✅")
        st.write("Total Records:", collection.count())

    except Exception as e:
        st.error(f"Error: {e}")

    # Embedding Information
    st.write("## Embedding Information")

    st.success(f"Total Embeddings: {len(embeddings)}")

    st.write("Embedding Dimension:", len(embeddings[0]))

    st.write("First 10 Values of First Embedding:")

    st.write(embeddings[0][:10])

    # ChromaDB
    st.write("## ChromaDB")

    st.success("Embeddings Stored Successfully! ✅")

    st.write("Total Records:", collection.count())

    # ==============================
    # Ask Questions From PDF
    # ==============================

    st.write("## 💬 Ask Questions From PDF")

    question = st.text_input(
        "Ask your question"
    )

    if question:

        results = search_documents(question)

        st.write("## 🔍 Top Matching Chunks")

        documents = results["documents"][0]

        for i, doc in enumerate(documents):
            st.write(f"### Result {i+1}")
            st.write(doc)