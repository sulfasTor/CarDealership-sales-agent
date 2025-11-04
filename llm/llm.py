import os

if not os.environ.get("OPENAI_API_KEY"):
    raise Exception("OPENAI_API_KEY is not set")

from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def init_store() -> VectorStore:
    vector_store = InMemoryVectorStore(embeddings)

    return vector_store
