from langchain_core.tools import tool
import requests, json, os
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Optional, Annotated
from dotenv import load_dotenv
from langgraph.types import RetryPolicy
load_dotenv()

from langchain_groq import ChatGroq


def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API"),
    )


@tool
def calculator(number1: int, number2: int, operator: str):
    """Takes 2 numbers and an operator (+, -, *, /) and returns the result."""
    n1, n2 = int(number1), int(number2)
    if operator == "+":
        return {"calculated_res": str(n1 + n2)}
    if operator == "-":
        return {"calculated_res": str(n1 - n2)}
    if operator == "*":
        return {"calculated_res": str(n1 * n2)}          # <- was missing `return`
    if operator == "/":
        if n2 == 0:
            return {"calculated_res": "divisor should be greater than 0"}
        return {"calculated_res": str(n1 // n2)}
    return {"calculated_res": f"unsupported operator: {operator}"}


@tool
def weather_condition(city: str):
    """Takes a city name and returns temperature, feels-like temp, humidity, and condition."""
    api_key = os.getenv("WEATHER_API")
    params = {"q": city, "appid": api_key, "units": "metric"}
    response = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params)

    if response.status_code != 200:
        return {"error": f"weather API error: {response.status_code}"}

    data = response.json()
    return {
        "Temperature": data["main"]["temp"],
        "Feels_like_temp": data["main"]["feels_like"],
        "Humidity": data["main"]["humidity"],
        "Weather_cond": data["weather"][0]["main"],
    }


@tool
def unit_converter(value: float, from_unit: str, to_unit: str):
    """
    Convert a numeric value between units.
    Length: km<->m, cm<->m, mile<->km, inch<->cm, ft<->m
    Weight: kg<->g, kg<->lb
    Temperature: C<->F
    """
    conversions = {
        ("km", "m"): 1000, ("m", "km"): 0.001,
        ("cm", "m"): 0.01, ("m", "cm"): 100,
        ("kg", "g"): 1000, ("g", "kg"): 0.001,
        ("kg", "lb"): 2.20462, ("lb", "kg"): 0.453592,
        ("mile", "km"): 1.60934, ("km", "mile"): 0.621371,
        ("inch", "cm"): 2.54, ("cm", "inch"): 0.393701,
        ("ft", "m"): 0.3048, ("m", "ft"): 3.28084,
    }

    if from_unit == "C" and to_unit == "F":
        return {"converted_value": value * 9 / 5 + 32}
    if from_unit == "F" and to_unit == "C":
        return {"converted_value": (value - 32) * 5 / 9}

    factor = conversions.get((from_unit, to_unit))
    if factor is None:
        return {"converted_value": f"Unsupported conversion: {from_unit} -> {to_unit}"}
    return {"converted_value": value * factor}


from langchain_tavily import TavilySearch


@tool
def internet_search(query: str):
    """Search the internet when the answer needs current info (stock prices, rankings, news, etc.)."""
    tavily = TavilySearch(tavily_api_key=os.getenv("TAVILY_API_KEY"), max_results=4)
    result = tavily.invoke(query)
    items = result.get("results", []) if isinstance(result, dict) else result

    norm = [
        {
            "url": i.get("url", ""),
            "title": i.get("title", ""),
            "content": i.get("content", ""),
        }
        for i in items
    ]
    return {"internet_search": norm}

tools = [internet_search, weather_condition, calculator, unit_converter]
tool_node = ToolNode(tools)


def load_tool_model():
    return load_model().bind_tools(tools)


# --- THE KEY FIX ---
# tools_condition / ToolNode both hinge on a "messages" list with the
# add_messages reducer. That reducer is what appends (not overwrites)
# each new AIMessage / ToolMessage as the graph runs.
class State(TypedDict):
    query: str
    tool_calls:int
    messages: Annotated[list, add_messages]
    result: str


def router(state: State):
    
    """The LLM sees the 4 tool schemas and decides which one (if any) to call.
    This *is* your multi-tool routing — you don't write the branching logic,
    bind_tools + tools_condition do it based on tool_calls being present or not."""
    calls =  state["tool_calls"]
    if calls >= 5:
        # force the model to answer NOW, no tools allowed
        model = load_model()
        out = model.invoke(state["messages"])
        return {"messages": [out], "tool_calls": calls + 1}

    tool_model = load_tool_model()
    out = tool_model.invoke(state["messages"])
    return {"messages": [out]  , "tool_calls": calls+1}


def no_tool_reducer(state: State):
    """Runs once the loop ends (router produced an AIMessage with no tool_calls).
    Because router always sees the full message history — including every
    ToolMessage from earlier passes — this final AIMessage is already the
    model's synthesized answer. No separate summarizing node needed."""
    return {"result": state["messages"][-1].content}
policy = RetryPolicy(max_attempts=4)

graph = StateGraph(State)
graph.add_node("router", router , retry_policy=policy)
graph.add_node("tool_node", tool_node , retry_policy=policy)
graph.add_node("no_tool_reducer", no_tool_reducer)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    tools_condition,
    {
        "tools": "tool_node",
        END: "no_tool_reducer",
    },
)
graph.add_edge("tool_node", "router")
graph.add_edge("no_tool_reducer", END)

workflow = graph.compile()

if __name__ == "__main__":
    query = "what thing in ai is trending in 2026?"

    init_state = {"query": query, "messages": [("user", query)], "result": "", "tool_calls": 0}
    res = workflow.invoke(init_state)
    print(res["result"])