import os
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def load_model():
    return ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API"))


# ---------- structured output models ----------

class Hotel(BaseModel):
    hotel_name: str = Field(..., description="Hotel name only.")
    stay_fee: str = Field(..., description="Hotel stay fee per day.")
    hotel_type: Literal["Luxury", "Mid_scale", "Budget"] = Field(
        ...,
        description="""Which category this hotel falls into:
        1) Luxury: highest comfort, personalized service, usually 5-star.
        2) Mid_scale: good quality at moderate prices, usually 3-4 star.
        3) Budget: basic, affordable, usually 1-2 star.
        If none fit exactly, pick whichever is closest."""
    )


class Hotels(BaseModel):
    hotels: list[Hotel]


# ---------- tools ----------

@tool
def hotel_search(query: str):
    """Searches hotels from the internet according to the city and constraints specified by the user."""
    tavily = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"), max_results=4)
    result = tavily.invoke(query)
    items = result.get("results", []) if isinstance(result, dict) else result
    norm = [
        {"url": i.get("url", ""), "title": i.get("title", ""), "content": i.get("content", "")}
        for i in items
    ]
    return {"internet_search": norm}


@tool
def HumanInTheLoop(que: str):
    """Call this when there isn't enough context to search, more specification
    is needed, or the request is ambiguous — this pauses and asks the human."""
    answer = interrupt(que)   # actual pause happens here
    return answer


def load_tool_model():
    model = load_model()
    return model.bind_tools([HumanInTheLoop, hotel_search])


# ---------- state ----------

class State(TypedDict):
    query: str
    messages: Annotated[list, add_messages]
    hotel_list: Hotels
    is_satisfied: bool
    is_confirmed: bool


# ---------- nodes ----------

def user_query(state: State):
    model = load_tool_model()
    messages = state["messages"] if state.get("messages") else [HumanMessage(content=state["query"])]
    result = model.invoke(messages)
    return {"messages": [result]}


def make_res(state: State):
    """Once no more tool calls are needed, turn the accumulated search
    results into a structured list of Hotel objects."""
    model = load_model().with_structured_output(Hotels)
    context = "\n".join(m.content for m in state["messages"] if getattr(m, "content", None))
    prompt = (
        f"User's original request: {state['query']}\n\n"
        f"Conversation and search results so far:\n{context}\n\n"
        "Extract the hotels mentioned into structured form."
    )
    hotels = model.invoke(prompt)
    return {"hotel_list": hotels}


def confirm_booking(state: State):
    """Second HITL point: show the found hotels and ask the human to
    confirm before treating anything as booked."""
    decision = interrupt({
        "question": "Here are the hotels found. Confirm one to book, or say 'no' to go back.",
        "hotels": [h.model_dump() for h in state["hotel_list"].hotels],
    })
    if isinstance(decision, dict) and decision.get("action") == "confirm":
        return {"is_confirmed": True}
    return {"is_confirmed": False}


def route_after_confirm(state: State) -> str:
    return "confirmed" if state.get("is_confirmed") else "not_confirmed"


# ---------- graph ----------

policy = RetryPolicy(max_attempts=4)
tool_node = ToolNode([HumanInTheLoop, hotel_search])

graph = StateGraph(State)
graph.add_node("user_query", user_query, retry_policy=policy)
graph.add_node("tool_node", tool_node ,retry_policy=policy)
graph.add_node("make_res", make_res, retry_policy=policy)
graph.add_node("confirm_booking", confirm_booking, retry_policy=policy)

graph.add_edge(START, "user_query")
graph.add_conditional_edges(
    "user_query",
    tools_condition,
    {
        "tools": "tool_node",
        END: "make_res",
    },
)
graph.add_edge("tool_node", "user_query")
graph.add_edge("make_res", "confirm_booking")
graph.add_conditional_edges(
    "confirm_booking",
    route_after_confirm,
    {
        "confirmed": END,
        "not_confirmed": "user_query",
    },
)

memory = InMemorySaver()
workflow = graph.compile(checkpointer=memory)

# ---------- display helper ----------

def show(response, label):
    """Prints either the interrupt payload (if paused) or the last message
    content / pending tool call, depending on what state actually holds."""
    print(f"\n=== {label} ===")

    if isinstance(response, dict) and "__interrupt__" in response:
        interrupt_obj = response["__interrupt__"][0]
        print("PAUSED — waiting on human input")
        print("Payload:", interrupt_obj.value)
        return

    last_msg = response["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        print("Model wants to call tool:", last_msg.tool_calls[0]["name"])
        print("With args:", last_msg.tool_calls[0]["args"])
    else:
        print("Message:", last_msg.content)

    if response.get("hotel_list"):
        print("Hotels found:")
        for h in response["hotel_list"].hotels:
            print(f"  - {h.hotel_name} | {h.stay_fee} | {h.hotel_type}")

    if "is_confirmed" in response:
        print("Confirmed:", response["is_confirmed"])


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hotel_1"}}
    query = "What are the best hotels in Pakistan?"

    print("USER QUERY:", query)

    response = workflow.invoke({"query": query, "messages": []}, config=config)
    show(response, "First run (expect: paused, ambiguous city)")

    response = workflow.invoke(Command(resume="Karachi"), config=config)
    show(response, "After clarifying city (expect: paused, confirm_booking)")

    response = workflow.invoke(Command(resume={"action": "confirm"}), config=config)
    show(response, "After confirming")