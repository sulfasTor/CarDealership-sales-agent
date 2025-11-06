from typing import Any, List

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI


class ContentFilterMiddleware(AgentMiddleware):
    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [str(kw).strip().lower() for kw in banned_keywords if kw]

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print("checking for banned keywords")

        messages = state.get("messages", [])
        if not messages:
            return None

        human_messages = [m for m in messages if getattr(m, "type", None) == "human"]
        if not human_messages:
            return None

        last_message = human_messages[-1]
        content = (last_message.content or "").lower()

        for keyword in self.banned_keywords:
            if keyword and keyword in content:
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "😕 Prefiero mantener una conversación respetuosa para poder ayudarte bien.\n\n"
                                "Reformulemos tu mensaje y con gusto te ayudo con opciones de autos o financiamiento en Kavak. 🚗"
                            )
                        )
                    ],
                    "jump_to": "end",
                }

        return None


class SafetyGuardrailMiddleware(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.safety_model = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    @hook_config(can_jump_to=["end"])
    def after_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        print("Running after agent")

        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None

        safety_prompt = f"""Evalúa si esta respuesta es apropiada o no para un agente de Kavak que está ayudando a un cliente por un chat de WhatsApp y le ofrece catálogo y lo guía en los planes de financiamiento.

Responde solamente con 'SAFE' o 'UNSAFE'.

Response: {last_message.content}
"""
        result = self.safety_model.invoke(safety_prompt)

        verdict = (result.content or "").upper()
        if "UNSAFE" in verdict:
            return {
                "messages": [
                    AIMessage(
                        content=(
                            "No puedo dar esa respuesta. 🙏\n"
                            "Tal vez me ayudaría si reformulas tu petición y la enfocamos en autos seminuevos o financiamiento en Kavak. 🚗"
                        )
                    )
                ],
                "jump_to": "end",
            }

        return None


def load_banned_keywords() -> List[str]:
    try:
        with open("./datasets/banned_words.txt", "r", encoding="utf-8") as f:
            banned = [line.strip() for line in f if line.strip()]
    except Exception:
        print("Failed to load banned words. Skipping...")
        return []

    print(f"Loaded {len(banned)} banned words!")
    return banned
