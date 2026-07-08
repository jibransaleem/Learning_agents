from langchain_core.tools import tool
import requests, json, os
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import TypedDict, Optional, Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from langgraph.types import RetryPolicy
load_dotenv()

from langchain_groq import ChatGroq

def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API"),
    )
import operator
import sqlite3
class state(TypedDict):
    query : str
    genrated_qry : Annotated[list , operator.add]
    un_correct_query:bool
    fetched_data : Annotated[list , operator.add]
    issue_with_query : str
    feeback: str
    result : list
    counter :int
    break_:bool
    
def get_schema():
    db_path = r"C:\Users\ADIL TRADERS\Desktop\agentic_learn\Learning_agents\project2_text_to_sql\company.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schema = "\n".join(row[0] for row in cursor.fetchall() if row[0])
    conn.close()
    return schema


def fetch_query(query):    
    try:
        conn = sqlite3.connect("company.db")
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        # print(rows)
        return True,[rows]
    except Exception as e:
        return False , str(e)


def QueryGenrator(state):
    schema =  get_schema()
    counter = state["counter"]
    if counter >=5:
        return {
            "counter" :"counter+1",
            "break_":True
        }   
    
    prev_query  =state["genrated_qry"][-1] if state["genrated_qry"] else ""
    sys_msg = SystemMessage(content = f"You are an expert Database Manager.Your task is to write correct and optimized sql queries for the database with following schema :{schema}")
    hmn_msg = HumanMessage(
    content=f"""
        You are an expert SQLite query generator.

        ## Database Schema
        {schema}

        ## User Request
        {state["query"]}

        ## Instructions
        1. Carefully understand the database schema.
        2. Think step by step about which tables and columns are required.
        3. Generate a correct and efficient SQLite query.
        4. Only use tables and columns that exist in the schema.
        5. Do not make assumptions or invent columns.
        6. Optimize the query where possible.

        ## Previous Attempt (if any)
        Query:
        {prev_query}

        Validation Status:
        {state["un_correct_query"]}

        Validation Feedback:
        {state["issue_with_query"]}

        Additional Feedback:
        {state["feeback"]}

        ## Task
        - If `Validation Status` indicates the previous query is incorrect, fix the previous query using all of the feedback provided.
        - Otherwise, generate a new SQL query from scratch based on the user's request.

        ## Output
        Return ONLY the SQL query.
        Do not include explanations, markdown, or code fences.
        """
        )
    model =  load_model()
    result = model.invoke([sys_msg , hmn_msg])
    print(result.content)
    return {
        "counter": counter+1,
        "genrated_qry": [result.content]      
    }
def query_exec(state):
    query = state["genrated_qry"][-1]
    
    is_exe  , result = fetch_query(query)
    if not is_exe:
        return {
            "un_correct_query":True,
            "issue_by_wrong_query": result
        }
    return {
        "un_correct_query":False,
        "fetched_data" : result
    }
def query_Debugger(state):
    last_query = state["genrated_qry"][-1]
    sy = SystemMessage(content = f"You are an expert Debugger for sql queries. Given the schema {get_schema()} you look the issue step by step and give a feedback for correcting the query")
    hu = HumanMessage(content = f"Given the query genrated {last_query} it is causing the issue :{state["issue_with_query"]}.Do debug and genrate feedback step by step")
    model = load_model()
    out = model.invoke([sy,hu])
    return {
        "feedback" : out.content,
        
    }
def router(state):
    if state["un_correct_query"]:
        return "query_debug"
    return "end_"
def break_cond(state):
    if  not state["break_"]:
        return "query_exec"
def end_(state):
    return {"result":state["fetched_data"][-1]}
    
graph = StateGraph(state)
graph.add_node("query_genrator"  , QueryGenrator)
graph.add_node("query_exec" , query_exec)
graph.add_node("query_debug" , query_Debugger)
graph.add_node("end_" , end_)
graph.add_edge(START , "query_genrator")
graph.add_conditional_edges("query_genrator" , break_cond , {"query_exec":"query_exec" , END:END})
# graph.add_edge("query_genrator" , "query_exec")
graph.add_conditional_edges("query_exec" , router ,{"query_debug" : "query_debug", "end_":"end_"})
graph.add_edge("query_debug" , "query_genrator")
graph.add_edge("end_" , END)
workflow =  graph.compile()
query ="""Show me all completed orders from customers based in Karachi, including who handled each order, sorted by amount from highest to lowest."""
query = {"query":query ,"un_correct_query":False , "issue_with_query" : "" ,"feeback":"" , "counter":0 , "break_":False}
res = workflow.invoke(query)
if len(res["result"])>0:
    for i in res["result"]:
        print(i)
else:
    print("LLM Fails to write efficient query")
    
    