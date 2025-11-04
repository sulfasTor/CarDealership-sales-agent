from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_core.embeddings import Embeddings

def init_store(embeddings: Embeddings) -> VectorStore:
    vector_store = InMemoryVectorStore(embeddings)

    return vector_store
