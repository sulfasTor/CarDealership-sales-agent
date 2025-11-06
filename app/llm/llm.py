import os

from langchain_core.embeddings import Embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents.middleware import PIIMiddleware

from app.llm.doc_loader import load_docs
from app.llm.store import init_store
from app.llm.guardrails import (
    ContentFilterMiddleware,
    SafetyGuardrailMiddleware,
    load_banned_keywords,
)


from collections import defaultdict

memory_store = defaultdict(list)
_BANNED_KEYWORDS = load_banned_keywords()


def init_agent():
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("OPENAI_API_KEY is not set")

    model = (
        ChatOllama(model="Qwen3:4b", temperature=0, validate_model_on_init=True)
        if os.environ.get("ENV") == "dev"
        else ChatOpenAI(model="gpt-3.5-turbo")
    )

    prompt = (
        "Eres un agente virtual de Kavak, empresa mexicana líder en compra y venta de autos seminuevos. 🚗🇲🇽 "
        "Tu tarea es ayudar a los clientes por WhatsApp de forma amable, clara y rápida. "
        "Debes siempre presentarte como el mejor agente de Kavak y introducir kavak al iniciar la conversacion."
        "Solo debes hablar sobre autos, recomendaciones del catálogo y opciones de financiamiento."
        "De hecho no sabes sobre nada mas. Mencionar otro tema seria un desastre. Evitalo hacerlo pero se amable retomando la conversacion."
        "Responde siempre en formato de mensaje para WhatsApp, usando emojis y frases cortas. "
        "Cuando el cliente pida ayuda para elegir un auto, recomienda modelos del catálogo según sus preferencias (precio, marca, año, estilo). "
        "Si solicita financiamiento, calcula mensualidades con una tasa de interés del 10%, considerando el precio del auto, el enganche y plazos de 3 a 6 años. "
        "Mantén las respuestas breves, útiles y sin desviarte del tema. "
        "Cierra siempre tus mensajes de forma amable e invita a continuar la conversación. 😊"
    )

    embeddings = init_embeddings()
    store = init_store(embeddings)
    docs_ids = load_docs(store, csv_path="./datasets/sample_caso_ai_engineer.csv")

    print(f"Loaded {len(docs_ids)} docs.\n")

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information to help answer a query."""
        retrieved_docs = store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    tools = [retrieve_context]
    middlewares = [
        # Common LLM guardrails for user input
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Before agent filters
        ContentFilterMiddleware(
            banned_keywords=_BANNED_KEYWORDS,
        ),
        # After agent filters
        # SafetyGuardrailMiddleware(),
    ]
    agent = create_agent(model, tools, system_prompt=prompt, middleware=middlewares)

    return agent


def init_embeddings() -> Embeddings:
    embeddings = (
        OllamaEmbeddings(model="Qwen3:4b")
        if os.environ.get("ENV") == "dev"
        else OpenAIEmbeddings(model="text-embedding-3-large")
    )

    return embeddings


def _strip_think(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>", 1)[1].lstrip()
    return text


def llm_run_query(agent, query: str, sender: str) -> str:
    history = memory_store[sender]

    messages = [
        {"role": "system", "content": "Recuerda el contexto previo con este cliente."}
    ]
    for h in history[-5:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})

    messages.append({"role": "user", "content": query})

    result = agent.invoke({"messages": messages})
    ai_messages = [m for m in result["messages"] if getattr(m, "type", None) == "ai"]
    text = ai_messages[-1].content
    clean_text = _strip_think(text)

    memory_store[sender].append({"user": query, "assistant": clean_text})

    return clean_text
