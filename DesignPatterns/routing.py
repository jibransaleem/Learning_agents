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



class Router(BaseModel):
    profile : Literal["Github" , "Medium ,none"] =  Field(default="none" , description="The output destionation is about Github or Medium ?Answer name only.If neither return none only")
    user_handle :str= Field(default="none" , description="What is user handle either for github or medium mentioned.If non mentioned return none")
    
class State(TypedDict):
    query : str 
    summary : str 
    route:Literal["Medium" ,"Github" ,"none"]
    user_handle : str
    
from langchain_core.messages import HumanMessage , SystemMessage
def profile_(state):
    sy = SystemMessage(content=  """You are an expert social media assistant.You have to decide On the user query wheather to pull user gihtub or medoum profile details  """)
    hu = HumanMessage(content = state["query"] )
    model = load_model()
    struc_ =model.with_structured_output(Router) 
    response = struc_.invoke([sy ,hu])
    return {"route" : response.profile,
            "user_handle"  : response.user_handle}
    
def router(state):
    if state["route"].lower() == "medium":
        return "MediumNode"
    if state["route"].lower() == "gitub":
        return "GithubNode"
    return "end" 
def MediumNode(state):
    return state
def GithubNode(state):
    return state
graph =StateGraph(State)
graph.add_node("github_node",GithubNode)
graph.add_node("medium_node",MediumNode)
graph.add_node("user" ,profile_)

graph.add_edge(START , "user")
graph.add_conditional_edges("user"  , router,{"MediumNode":"medium_node" ,"GithubNode": "github_node" , "end":END})
graph.add_edge("github_node" , END)
graph.add_edge("medium_node" , END)
workflow =  graph.compile()
# print(workflow.get_graph().draw_mermaid())

res = workflow.invoke({"query":"Can u summerize my github profile for user Jibran_8"})
