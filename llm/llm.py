import os

from langchain_core.embeddings import Embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain_core.vectorstores import VectorStore
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from llm.doc_loader import load_docs
from llm.store import init_store


def init_agent():
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY is not set")

    model = (
        ChatOllama(model="phi3:mini", temperature=0, validate_model_on_init=True)
        if os.environ.get("ENV") == "dev"
        else ChatOpenAI(model="gpt-3.5-turbo")
    )

    tools = [retrieve_context]
    prompt = (
        "Eres un agente de Kavak que es una empresa mexicana que se desempeña como marketplace en linea del mercado de carros usado."
        "Debes brindar recomendaciones de los autos disponibles en extracto del catálogo dadas las preferencias del cliente."
        "Debes otorgar planes de financiamiento tomando como base el enganche, el precio del auto, una tasa de interés del 10% y plazos de financiamiento de entre 3 y 6 años."
    )

    # Init store
    embeddings = init_embeddings()
    store = init_store(embeddings)
    docs_ids = load_docs(store, csv_path="sample_caso_ai_engineer.csv")

    print(f"Loaded {len(docs_ids)} docs.\n")

    # Init agent
    agent = create_agent(model, tools, system_prompt=prompt)

    return agent


def init_embeddings() -> Embeddings:
    embeddings = (
        OllamaEmbeddings(model="phi3:mini")
        if os.environ.get("ENV") == "dev"
        else OpenAIEmbeddings(model="text-embedding-3-large")
    )

    return embeddings


# LLM tools
@tool(response_format="content_and_artifact")
def retrieve_context(vector_store: VectorStore, query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def test():
    agent = init_agent()

    query = "Hola?\n\nQue vehiculos estan disponibles?"

    for event in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


test()
