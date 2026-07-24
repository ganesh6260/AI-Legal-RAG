import streamlit as st
import fitz
import time
from utils import chunk_text
from embeddings import create_embeddings
from vectordb import store_embeddings
from retriever import search_documents
from llm import generate_answer

st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Legal Assistant")
st.write("Welcome to our RAG Project!")

# -------------------------------
# Session State
# -------------------------------

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None

# -------------------------------
# Sidebar
# -------------------------------

uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

# -------------------------------
# Detect New PDF
# -------------------------------

if uploaded_file is not None:

    if st.session_state.current_pdf != uploaded_file.name:

        st.session_state.current_pdf = uploaded_file.name

        st.session_state.chat_history = []

        st.session_state.pdf_processed = False

# -------------------------------
# Clear Chat Button
# -------------------------------

if st.sidebar.button("🗑 Clear Chat"):

    st.session_state.chat_history = []

    st.rerun()

# -------------------------------
# Process PDF ONLY ONE TIME
# -------------------------------

if uploaded_file is not None and not st.session_state.pdf_processed:

    st.sidebar.success("✅ PDF Uploaded")
    st.sidebar.write("📄", uploaded_file.name)
    st.sidebar.write(f"📦 {uploaded_file.size/1024:.1f} KB")

    pdf = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf:
        text += page.get_text()

    st.write("## Extracted Text")
    st.text(text[:3000])

    # Chunking
    chunks = chunk_text(text)

    st.write("## Number of Chunks")
    st.success(len(chunks))

    # Embeddings
    embeddings = create_embeddings(chunks)

    st.write("## Embedding Information")
    st.success(f"Total Embeddings: {len(embeddings)}")

    # Store in ChromaDB
    collection = store_embeddings(chunks, embeddings)

    st.success("Embeddings Stored Successfully ✅")
    st.write("Total Records:", collection.count())

    st.session_state.pdf_processed = True


# -------------------------------
# Prepare Chat History for Download
# -------------------------------

chat_text = "========== AI LEGAL ASSISTANT CHAT ==========\n\n"

for message in st.session_state.chat_history:

    if message["role"] == "user":
        chat_text += f"👤 User:\n{message['content']}\n\n"

    else:
        chat_text += f"🤖 AI:\n{message['content']}\n\n"

chat_text += "============================================"

# -------------------------------
# Chat Section
# -------------------------------

if st.session_state.pdf_processed:

    st.divider()

    st.subheader("💬 Chat with your PDF")

    st.download_button(
        label="📥 Download Chat History",
        data=chat_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

    # Display Previous Chat
    for message in st.session_state.chat_history:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input
    question = st.chat_input("Ask anything about your PDF...")

    if question:

        # Save User Message
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        # Retrieve Documents
        results = search_documents(question)

        context = ""

        for doc in results["documents"][0]:
            context += doc + "\n\n"

        source_chunks = results["documents"][0]

        # Generate Answer
        start_time = time.time()

        with st.spinner("🤖 AI is thinking..."):
            answer = generate_answer(
                question,
                context
            )

        end_time = time.time()

        response_time = round(end_time - start_time, 2)   

        # Save Assistant Response
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Display Answer
        with st.chat_message("assistant"):
            st.write(answer)
            st.caption(f"⏱ Response Time: {response_time} sec")

        # Source Chunks
        with st.expander("📄 View Source Chunks"):

            st.caption(
                "The following document sections were retrieved and used by the AI to generate the answer."
            )

            for i, chunk in enumerate(source_chunks):

                st.markdown(f"### 🔹 Source {i+1}")

                st.info(chunk)

                st.divider()