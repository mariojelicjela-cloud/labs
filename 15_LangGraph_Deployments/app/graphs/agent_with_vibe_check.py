from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END


class GraphState(TypedDict):
    messages: List[BaseMessage]
    draft_answer: str
    vibe_ok: bool
    feedback: str
    attempts: int


llm = ChatOpenAI(model="gpt-4o-mini")


def generate_answer(state: GraphState) -> GraphState:
    messages = state["messages"]
    attempts = state.get("attempts", 0)
    feedback = state.get("feedback", "")

    if feedback:
        prompt = (
            "You are a helpful assistant. Improve your previous answer using this feedback:\n"
            f"{feedback}\n\n"
            "Return only the improved answer."
        )
        response = llm.invoke(messages + [HumanMessage(content=prompt)])
    else:
        response = llm.invoke(messages)

    return {
        **state,
        "draft_answer": response.content,
        "attempts": attempts + 1,
    }


def vibe_check(state: GraphState) -> GraphState:
    answer = state["draft_answer"]

    prompt = f"""
You are a vibe checker for assistant responses.

Evaluate the answer below on:
1. clarity
2. friendliness
3. usefulness

If the answer is good enough, respond exactly like this:
PASS

If the answer is not good enough, respond exactly like this:
FAIL: <short feedback>

Answer to evaluate:
{answer}
""".strip()

    result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    if result.startswith("PASS"):
        return {
            **state,
            "vibe_ok": True,
            "feedback": "",
        }

    feedback = result.replace("FAIL:", "").strip()
    return {
        **state,
        "vibe_ok": False,
        "feedback": feedback or "Make the answer clearer, friendlier, and more useful.",
    }


def should_continue(state: GraphState) -> str:
    if state.get("vibe_ok"):
        return "end"

    if state.get("attempts", 0) >= 3:
        return "end"

    return "retry"


def finalize(state: GraphState) -> GraphState:
    final_text = state["draft_answer"]
    return {
        **state,
        "messages": state["messages"] + [HumanMessage(content=f"Final answer: {final_text}")]
    }


builder = StateGraph(GraphState)

builder.add_node("generate_answer", generate_answer)
builder.add_node("vibe_check", vibe_check)
builder.add_node("finalize", finalize)

builder.set_entry_point("generate_answer")
builder.add_edge("generate_answer", "vibe_check")

builder.add_conditional_edges(
    "vibe_check",
    should_continue,
    {
        "retry": "generate_answer",
        "end": "finalize",
    },
)

builder.add_edge("finalize", END)

graph = builder.compile()