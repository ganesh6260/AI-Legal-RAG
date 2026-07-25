from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

# Load model only once
model = SentenceTransformer("all-MiniLM-L6-v2")


class LocalEmbedding(Embeddings):

    def embed_documents(self, texts):
        return model.encode(texts).tolist()

    def embed_query(self, text):
        return model.encode(text).tolist()


embedding_function = LocalEmbedding()

vector_store = Chroma(
    collection_name="legal_documents",
    embedding_function=embedding_function
)


def store_embeddings(chunks):
    """
    Store LangChain Documents into Chroma Vector Store.
    """

    vector_store.add_documents(chunks)

    return vector_store