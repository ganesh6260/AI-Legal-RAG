import streamlit as st
import fitz
import time
import logging

from utils.utils import chunk_text
from rag.vectordb import store_embeddings
from rag.chat_graph import ask as chat_ask  # <-- memory-aware RAG + LLM in one call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-legal-assistant")

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------
# Light-touch custom styling
# (colors/background/text now come from .streamlit/config.toml, so this
# only adds a few extra visual touches — it never fights Streamlit's own
# text-contrast handling, which is what caused the dark-on-dark text)
# -------------------------------
st.markdown("""
<style>
    .hero {
        padding: 1.75rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #1e3a5f 0%, #14324f 100%);
        border: 1px solid rgba(148, 163, 184, 0.25);
        margin-bottom: 1.5rem;
    }
    .hero h1 {
        color: #f8fafc;
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 1rem;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        border-radius: 12px;
        padding: 0.6rem 0.9rem;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Hero Header
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>⚖️ AI Legal Assistant</h1>
    <p>Upload a legal document and ask questions grounded in its actual text — powered by RAG.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Session State
# -------------------------------
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "chat_history" not in st.session_state:
    # keyed per PDF name, so each document keeps its own visible transcript
    st.session_state.chat_history = {}
if "current_pdf" not in st.session_state:
    st.session_state.current_pdf = None
if "num_chunks" not in st.session_state:
    st.session_state.num_chunks = {}
if "preview_text" not in st.session_state:
    st.session_state.preview_text = {}
if "thread_versions" not in st.session_state:
    # bump this per PDF when "Clear Chat" is hit, to start a fresh memory thread
    st.session_state.thread_versions = {}

# -------------------------------
# Sidebar — top (upload / clear)
# -------------------------------
with st.sidebar:
    st.markdown("### 📁 Document")
    uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"], label_visibility="collapsed")

    st.divider()
    if st.button("🗑 Clear Chat for this PDF", use_container_width=True):
        if st.session_state.current_pdf:
            pdf_name = st.session_state.current_pdf
            st.session_state.chat_history[pdf_name] = []
            # bumping the thread version starts a brand-new LangGraph memory
            # thread, so the model "forgets" prior turns for this PDF
            st.session_state.thread_versions[pdf_name] = (
                st.session_state.thread_versions.get(pdf_name, 0) + 1
            )
        st.rerun()


def get_thread_id(pdf_name: str) -> str:
    """One LangGraph memory thread per uploaded PDF (per its clear-chat version)."""
    version = st.session_state.thread_versions.get(pdf_name, 0)
    return f"{pdf_name}::v{version}"


def render_sidebar_history(pdf_name: str):
    """
    Renders document stats + collapsible Q&A history in the sidebar.
    IMPORTANT: this is called at the END of the script (after the current
    question has been answered and appended to history), not at the top —
    otherwise it always shows state from one run behind the latest answer.
    """
    with st.sidebar:
        st.divider()
        st.markdown("### 📊 Document Stats")
        c1, c2 = st.columns(2)
        c1.metric("Chunks", st.session_state.num_chunks.get(pdf_name, 0))
        c2.metric("Messages", len(st.session_state.chat_history.get(pdf_name, [])))

        st.divider()
        st.markdown("### 🕑 Chat History")
        sidebar_history = st.session_state.chat_history.get(pdf_name, [])
        if not sidebar_history:
            st.caption("No messages yet.")
            return

        # Pair up each user question with the assistant answer that follows it
        qa_pairs = []
        pending_question = None
        for message in sidebar_history:
            if message["role"] == "user":
                pending_question = message["content"]
            elif message["role"] == "assistant" and pending_question is not None:
                qa_pairs.append((pending_question, message["content"]))
                pending_question = None
        if pending_question is not None:
            qa_pairs.append((pending_question, "⏳ Waiting for answer..."))

        with st.container(height=350):
            for i, (q, a) in enumerate(qa_pairs, start=1):
                with st.expander(f"Q{i}: {q}"):
                    st.markdown(a)


# -------------------------------
# Detect New / Switched PDF
# -------------------------------
if uploaded_file is not None:
    is_new_pdf = st.session_state.current_pdf != uploaded_file.name
    st.session_state.current_pdf = uploaded_file.name

    if uploaded_file.name not in st.session_state.chat_history:
        st.session_state.chat_history[uploaded_file.name] = []

    if is_new_pdf and uploaded_file.name not in st.session_state.num_chunks:
        st.session_state.pdf_processed = False
    elif uploaded_file.name in st.session_state.num_chunks:
        # already indexed earlier in this session — its memory thread is
        # still alive, so we can skip re-embedding and go straight to chat
        st.session_state.pdf_processed = True

# -------------------------------
# Process PDF
# -------------------------------
if uploaded_file is not None and not st.session_state.pdf_processed:
    pdf_name = uploaded_file.name
    with st.status("Processing document...", expanded=True) as status:
        st.write("📄 Extracting text...")
        pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in pdf)

        if not text.strip():
            status.update(label="No extractable text found", state="error")
            st.error(
                "This PDF doesn't contain selectable text (it may be a scanned image). "
                "Try an OCR'd version of the file."
            )
            st.stop()

        st.write("✂️ Splitting into chunks...")
        chunks = chunk_text(text)

        st.write(f"🧠 Generating embeddings for {len(chunks)} chunks...")
        try:
            # NOTE: if store_embeddings persists to a single shared collection,
            # make sure it either (a) namespaces by pdf_name/thread_id, or
            # (b) clears previous vectors — otherwise a second PDF's questions
            # could retrieve chunks from the first PDF.
            store_embeddings(chunks)
        except Exception as e:
            logger.exception("Embedding/storage failed")
            status.update(label="Indexing failed", state="error")
            st.error(f"Couldn't index this document: {e}")
            st.stop()

        st.session_state.num_chunks[pdf_name] = len(chunks)
        st.session_state.preview_text[pdf_name] = text[:3000]
        st.session_state.pdf_processed = True
        status.update(label="✅ Document indexed", state="complete")

    st.rerun()

# -------------------------------
# Chat Section
# -------------------------------
if st.session_state.pdf_processed and st.session_state.current_pdf:
    pdf_name = st.session_state.current_pdf
    thread_id = get_thread_id(pdf_name)
    history = st.session_state.chat_history.setdefault(pdf_name, [])

    st.subheader(f"💬 Chat with: {pdf_name}")

    with st.expander("📄 Extracted text preview"):
        st.text(st.session_state.preview_text.get(pdf_name, ""))

    st.divider()

    # Previous chat (this PDF's transcript only)
    for message in history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Ask anything about your document...")

    if question:
        history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("Thinking..."):
                try:
                    # `ask()` handles retrieval + trimmed multi-turn history +
                    # generation in one call, scoped to this PDF's thread_id
                    answer, source_chunks = chat_ask(question, thread_id)
                except Exception as e:
                    logger.exception("Chat graph invocation failed")
                    answer, source_chunks = (
                        f"Sorry, something went wrong generating a response: {e}",
                        [],
                    )
            response_time = round(time.time() - start_time, 2)
            st.write(answer)
            st.caption(f"⏱ Response time: {response_time}s · {len(source_chunks)} source chunk(s) used")

            if source_chunks:
                with st.expander("📄 View source chunks"):
                    for i, chunk in enumerate(source_chunks):
                        st.markdown(f"**🔹 Source {i + 1}**")
                        st.info(chunk)

        history.append({"role": "assistant", "content": answer})

    # Rendered LAST, after the current question's answer has already been
    # appended to history above — this is what fixes the "one behind" lag.
    render_sidebar_history(pdf_name)

else:
    st.info("👈 Upload a PDF from the sidebar to get started.")