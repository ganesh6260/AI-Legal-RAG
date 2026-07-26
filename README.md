# ⚖️ AI Legal Assistant (AI-Legal-RAG)

> Upload a legal document and have a grounded, multi-turn conversation about it — powered by Retrieval-Augmented Generation (RAG) and Google Gemini.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![LangGraph](https://img.shields.io/badge/LangGraph-Memory-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini-orange)

---

## 📖 Overview

**AI Legal Assistant** is a Retrieval-Augmented Generation (RAG) application that lets you upload a legal PDF (contracts, agreements, policies, etc.) and ask natural-language questions about it. Instead of relying on the LLM's general knowledge, every answer is **grounded in the actual text of your document** — reducing hallucination and keeping responses traceable to their source.

The assistant also supports **multi-turn conversations with per-document memory**, so follow-up questions like *"what about clause 4?"* or *"does that apply to both parties?"* are understood in context — without mixing up memory across different uploaded documents.

🔗 **Live Demo:** https://ai-legal-rag-ganesh.streamlit.app/

---

## ✨ Features

- 📄 **PDF ingestion** — extracts text directly from uploaded PDFs using PyMuPDF
- ✂️ **Smart chunking** — splits documents into overlapping chunks for accurate retrieval
- 🧠 **Semantic search** — embeds and indexes chunks in a vector database (ChromaDB) for relevant-passage retrieval
- 💬 **Multi-turn chat memory** — built on LangGraph, with a separate conversation thread per uploaded PDF
- 🔍 **Source transparency** — every answer shows exactly which chunks of the document it was generated from
- 🖥️ **Clean chat interface** — built with Streamlit, including a collapsible per-document Q&A history sidebar
- ⚠️ **Graceful error handling** — friendly messages for rate limits, server errors, and unreadable/scanned PDFs

---

## 🧰 Tech Stack

| Layer              | Technology                                      |
|--------------------|--------------------------------------------------|
| UI                 | [Streamlit](https://streamlit.io)                |
| PDF Parsing        | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io) |
| Orchestration      | [LangChain 1.x](https://python.langchain.com)    |
| Conversation Memory| [LangGraph](https://langchain-ai.github.io/langgraph/) |
| Vector Store       | [ChromaDB](https://www.trychroma.com)            |
| Embeddings         | Sentence-Transformers (`all-MiniLM-L6-v2`)        |
| LLM                | Google Gemini (`gemini-flash-latest`)            |
| Language           | Python 3.11                                      |

---

## 📁 Project Structure

```
AI-Legal-RAG/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .env                    # API keys (not committed)
├── llm/
│   └── llm.py               # Gemini LLM wrapper + prompt invocation
├── rag/
│   ├── vectordb.py          # Embedding generation + Chroma storage
│   ├── retriever.py         # Semantic search over stored chunks
│   ├── chat_graph.py        # LangGraph memory graph (per-PDF threads)
│   └── prompt_template.py   # Prompt template (context + chat history)
└── utils/
    └── utils.py              # Text chunking utilities
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ganesh6260/AI-Legal-RAG.git
cd AI-Legal-RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 💡 Usage

1. Upload a legal PDF from the sidebar.
2. Wait for the document to be processed (extracted, chunked, and embedded).
3. Ask questions in the chat box — e.g. *"What is the termination notice period?"*
4. Expand **📄 View source chunks** under any answer to see exactly which parts of the document it came from.
5. Ask follow-up questions naturally — the assistant remembers the conversation for that document.
6. Use **🗑 Clear Chat for this PDF** in the sidebar to reset memory for the current document.

---

## 🗺️ Roadmap

- [ ] Support for multiple simultaneous documents in one conversation
- [ ] Persistent chat memory across app restarts (SQLite-backed checkpointer)
- [ ] OCR support for scanned PDFs
- [ ] Export answers with citations to PDF/Word
- [ ] Support for additional LLM providers (OpenAI, Claude, local models)

---

## ⚠️ Known Limitations

- Only PDFs with selectable (non-scanned) text are supported.
- Conversation memory is in-memory by default and resets on app restart.
- Answers are only as accurate as the retrieved chunks — always verify critical legal conclusions with a qualified professional.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Open a pull request

---

## 🙏 Disclaimer

This tool is intended for informational and research purposes only and does **not** constitute legal advice. Always consult a qualified legal professional for decisions involving real legal documents.