"""
agent_graph.py
---------------
LangGraph multi-agent orchestration layer for Celestia.

A router agent classifies the user's message (chat / compatibility / horoscope)
and hands off to the matching specialist worker agent. Each worker performs its
own FAISS retrieval (via retrieve_with_confidence in server.py) and its own LLM
generation. This sits on top of the existing single-purpose Flask routes and
reuses their retrieval/generation logic instead of duplicating it.

Run as a CLI smoke test:
    python agent_graph.py
"""

import json
import re
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from server import (
    ZODIAC_TONES,
    get_llm,
    get_zodiac_from_date,
    map_message_to_zodiac_topic,
    parse_dob_to_month_day,
    retrieve_with_confidence,
)


# --- Shared state passed between agents ---
class AgentState(TypedDict, total=False):
    message: str
    zodiac: Optional[str]
    sign2: Optional[str]
    intent: Literal["chat", "compatibility", "horoscope"]
    context: str
    confidence: int
    response: str


# --- Router agent: classifies intent + extracts zodiac signs ---
ROUTER_PROMPT = """Classify the user's astrology request into exactly one category
and extract any zodiac signs mentioned.

Categories:
- "compatibility": asks how two signs get along, match, or are compatible
- "horoscope": asks for today's/daily horoscope or forecast
- "chat": anything else (general questions, advice, feelings, follow-ups)

User message: "{message}"
Known zodiac sign from session (may be empty): "{zodiac}"

Return ONLY a JSON object, no extra text:
{{"intent": "chat" | "compatibility" | "horoscope", "sign1": "<zodiac sign or empty>", "sign2": "<second sign or empty, only for compatibility>"}}
"""


def router_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = ROUTER_PROMPT.format(message=state["message"], zodiac=state.get("zodiac", ""))
    raw = llm.invoke([{"role": "user", "content": prompt}]).content.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not match:
        return {**state, "intent": "chat"}

    parsed = json.loads(match.group())
    intent = parsed.get("intent", "chat")
    if intent not in ("chat", "compatibility", "horoscope"):
        intent = "chat"

    return {
        **state,
        "intent": intent,
        "zodiac": parsed.get("sign1") or state.get("zodiac", ""),
        "sign2": parsed.get("sign2") or state.get("sign2", ""),
    }


def route_decision(state: AgentState) -> str:
    return state["intent"]


# --- Chat worker agent ---
def chat_agent(state: AgentState) -> AgentState:
    zodiac = state.get("zodiac", "")
    topic_keywords = map_message_to_zodiac_topic(state["message"], zodiac)
    context, confidence, is_fallback = retrieve_with_confidence(f"{zodiac} zodiac {topic_keywords}")
    if is_fallback:
        context, confidence, _ = retrieve_with_confidence(
            f"{zodiac} zodiac traits strengths weaknesses life advice"
        )

    tone = ZODIAC_TONES.get(zodiac, "Friendly and helpful astrology assistant.")
    prompt = f"""You are Celestia, a warm astrology guide. The user's sign is {zodiac}.
Personality style: {tone}
Zodiac profile:
{context}

User: {state['message']}

Reply in 3-5 warm, conversational sentences, weaving in one relevant trait."""

    reply = get_llm().invoke([{"role": "user", "content": prompt}]).content.strip()
    return {**state, "context": context, "confidence": confidence, "response": reply}


# --- Compatibility worker agent ---
def compatibility_agent(state: AgentState) -> AgentState:
    sign1, sign2 = state.get("zodiac", ""), state.get("sign2", "")
    ctx1, conf1, _ = retrieve_with_confidence(f"{sign1} zodiac love relationships traits element")
    ctx2, conf2, _ = retrieve_with_confidence(f"{sign2} zodiac love relationships traits element")
    context = f"{sign1} Profile:\n{ctx1}\n\n{sign2} Profile:\n{ctx2}"

    prompt = f"""Using ONLY the profiles below, write a 4-5 sentence compatibility
reading between {sign1} and {sign2}, then end with a new line: "Compatibility Rating: High/Medium/Low".

{context}"""

    reply = get_llm().invoke([{"role": "user", "content": prompt}]).content.strip()
    return {**state, "context": context, "confidence": max(conf1, conf2), "response": reply}


# --- Horoscope worker agent ---
def horoscope_agent(state: AgentState) -> AgentState:
    zodiac = state.get("zodiac", "")
    ctx, conf, _ = retrieve_with_confidence(f"{zodiac} zodiac strengths weaknesses life advice lucky")

    prompt = f"""Write a warm, 4-5 sentence daily horoscope for {zodiac} using ONLY this profile:
{ctx}"""

    reply = get_llm().invoke([{"role": "user", "content": prompt}]).content.strip()
    return {**state, "context": ctx, "confidence": conf, "response": reply}


# --- Build the graph: router -> chat|compatibility|horoscope -> END ---
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_agent)
    graph.add_node("chat", chat_agent)
    graph.add_node("compatibility", compatibility_agent)
    graph.add_node("horoscope", horoscope_agent)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"chat": "chat", "compatibility": "compatibility", "horoscope": "horoscope"},
    )
    graph.add_edge("chat", END)
    graph.add_edge("compatibility", END)
    graph.add_edge("horoscope", END)

    return graph.compile()


celestia_agent = build_graph()


if __name__ == "__main__":
    print("Celestia multi-agent CLI - router decides chat / compatibility / horoscope")
    zodiac = input("Your zodiac sign (e.g. Scorpio): ").strip()
    while True:
        message = input("\nYou: ").strip()
        if message.lower() in ("quit", "exit", "q"):
            break
        result = celestia_agent.invoke({"message": message, "zodiac": zodiac})
        print(f"[routed to: {result['intent']}] (confidence: {result.get('confidence')}%)")
        print(f"Celestia: {result['response']}")
