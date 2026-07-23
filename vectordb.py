import chromadb

# Create Chroma Client
client = chromadb.Client()

# Create Collection
collection = client.get_or_create_collection(
    name="legal_documents"
)

def store_embeddings(chunks, embeddings):
    """
    Store text chunks and embeddings into ChromaDB.
    """

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return collection