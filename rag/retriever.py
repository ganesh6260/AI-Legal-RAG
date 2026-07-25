from rag.vectordb import vector_store


def search_documents(query, top_k=5):
    """
    Retrieve the most relevant documents using LangChain Chroma.
    """

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k
    )

    documents = []
    distances = []

    for doc, score in results:
        documents.append(doc.page_content)
        distances.append(score)

    return {
        "documents": [documents],
        "distances": [distances]
    }