from langchain_core.tools import tool
import requests, os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API"),
    )


@tool
def HumanResponse(que: str):
    """Call this when the system needs to ask the user something before it
    can continue — e.g. missing city, missing date, any required info."""
    # Body is basically a placeholder — the LLM just needs this schema to
    # know the tool exists and what argument to pass. The actual pause
    # happens in run_agent, not here.
    return que


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
        "weather_cond": [{
            "Temperature": data["main"]["temp"],
            "Feels_like_temp": data["main"]["feels_like"],
            "Humidity": data["main"]["humidity"],
            "Weather_cond": data["weather"][0]["main"],
        }]
    }


def model_with_tools():
    model = load_model()
    return model.bind_tools([weather_condition, HumanResponse])


class State(TypedDict):
    query: str
    weather_cond: list[dict]
    human_resp: str
    rsp: str


def run_agent(state: State):
    model = model_with_tools()
    resp = model.invoke(state["query"])

    if resp.tool_calls and resp.tool_calls[0]["name"] == "HumanResponse":
        question = resp.tool_calls[0]["args"]["que"]
        print(resp.tool_calls)
        print(question)
        # ACTUAL pause happens here. Execution of this node stops entirely
        # at this line until the graph is resumed with Command(resume=...).
        city = interrupt(question)

        wet = weather_condition.invoke(city)
        return {"human_resp": city, "weather_cond": wet["weather_cond"]}

    return {"rsp": resp.content}


graph = StateGraph(State)
graph.add_node("run_agent", run_agent)
graph.add_edge(START, "run_agent")
graph.add_edge("run_agent", END)

memory = InMemorySaver()
workflow = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "user_1"}}

    response = workflow.invoke(
        {"query": "Ask the user where they are, then look up the weather there"},
        config=config,
    )
    config = {"configurable": {"thread_id": "user_1"}}
    response = workflow.invoke(Command(resume="karachi"), config=config)
    print("\n=== After resume (should show actual weather data) ===")
    print(response)   