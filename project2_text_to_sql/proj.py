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

    ## Previous Query (if any)
    {prev_query}

    ## Validation Status
    {state["un_correct_query"]}

    ## Validation Feedback
    {state["issue_with_query"]}

    ## Review Feedback
    {state["feeback"]}

    ## Instructions

    Carefully analyze the database schema before writing any SQL.

    Depending on the situation, perform ONE of the following:

    ### Case 1: No previous query
    Generate a new SQL query that satisfies the user's request.

    ### Case 2: Previous query is incorrect
    If Validation Status indicates the query is incorrect:
    - Fix the query using ALL validation feedback.
    - Correct syntax errors.
    - Correct joins.
    - Correct table or column names.
    - Correct filters, grouping, ordering, or aggregations.
    - Ensure the final query satisfies the user's request.

    ### Case 3: Previous query is correct but reviewer suggests improvements
    If the previous query is valid but the feedback indicates:
    - missing required data,
    - incomplete results,
    - unnecessary joins,
    - inefficient logic,
    - possible optimization,
    - or any other improvement,

    then modify the previous query accordingly while preserving correctness.

    ## General Rules

    - Only use tables and columns present in the schema.
    - Never invent tables or columns.
    - Produce an efficient SQLite query.
    - Avoid unnecessary joins or subqueries.
    - Return complete results that satisfy the user's request.
    - If multiple improvements are suggested, apply all of them.

    ## Output

    Return ONLY the final SQL query.

    Do NOT include:
    - explanations
    - markdown
    - code fences
    - comments
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
    return "evaluator"
def break_cond(state):
    if  not state["break_"]:
        return "query_exec"
def end_(state):
    return {"result":state["fetched_data"][-1]}
from pydantic import BaseModel , Field
from typing import Literal
from typing import Literal
from pydantic import BaseModel, Field

class Eval(BaseModel):
    feedback: str = Field(
        description=(
            "If improvements are needed, provide concise and specific feedback "
            "explaining what should be changed in the SQL query. "
            "If no improvements are needed, return an empty string."
        )
    )

    need_imp: Literal["Yes", "No"] = Field(
        description=(
            "Return 'Yes' if the SQL query needs any modification, including "
            "correctness, completeness of results, or query optimization. "
            "Otherwise, return 'No'."
        )
    )  
def query_evaluator(state):
    sy = SystemMessage(content = f"You are an expert Sql query Evaluator.You See , analyize and optimize the sql queries for the given schema \n {get_schema()}")
    hu = HumanMessage(
        content=f"""
    You are reviewing a generated SQLite query.

    ## User Request
    {state["query"]}

    ## Generated SQL
    {state["genrated_qry"][-1]}

    ## Query Result
    {state["fetched_data"][-1]}

    Analyze step by step.

    Answer the following:

    1. Does the returned data completely satisfy the user's request?
    Answer: Yes or No.

    2. If No, explain exactly what information is missing or incorrect.

    3. Is the SQL query logically correct?
    Answer: Yes or No.

    4. Can the SQL query be optimized?
    Consider:
    - unnecessary joins
    - unnecessary subqueries
    - redundant conditions
    - inefficient filtering
    - better aggregation
    - simpler SQL

    5. Should the SQL query be modified for ANY reason
    (correctness, completeness, or optimization)?

    Answer ONLY:
    Yes
    or
    No

    6. If the answer to (5) is Yes, provide concise feedback describing every required change.
    """
    )
    model =  load_model()
    ev_model  = model.with_structured_output(Eval)
    
    res = ev_model.invoke([sy , hu])
    if (res.need_imp).lower()== "yes":
        return{
            "break_":True,
            "feedback" : res.feedback
            
        }
    return{
        "break_":False,
        "feedback" : res.feedback
    }
        
def need_optimization(state):
    if state["break_"]:
        return "end_"
    return  "query_genrator"
    
graph = StateGraph(state)
policy = RetryPolicy(max_attempts=4)
graph.add_node("query_genrator"  , QueryGenrator , retry_policy=policy)
graph.add_node("query_exec" , query_exec)
graph.add_node("query_debug" , query_Debugger , retry_policy=policy)
graph.add_node("end_" , end_)
graph.add_node("evaluator" ,query_evaluator,retry_policy=policy)
graph.add_edge(START , "query_genrator")
graph.add_conditional_edges("query_genrator" , break_cond , {"query_exec":"query_exec" , END:END})
graph.add_conditional_edges("query_exec" , router ,{"query_debug" : "query_debug", "evaluator":"evaluator"})
graph.add_conditional_edges("evaluator" ,need_optimization  ,{"end_":"end_" ,"query_genrator":"query_genrator" })
graph.add_edge("query_debug" , "query_genrator")
graph.add_edge("end_" , END)
workflow =  graph.compile()
# query ="""Show me all completed orders from customers based in Karachi, including who handled each order, sorted by amount from highest to lowest."""

Q = ["Which employees have never handled an order?" ,"List employees who earn more than the average salary in their own department","Show total completed order revenue grouped by department, ordered highest to lowest.","For each employee, show their single largest order — only for employees who have at least one order","How many orders were placed each month in 2024, and what was the total revenue for each month?"]


for qry in Q:  
    print(f"\n {qry} \n")  
    try :
        query = {"query":qry ,"un_correct_query":False , "issue_with_query" : "" ,"feeback":"" , "counter":0 , "break_":False}

        res = workflow.invoke(query)

        if len(res["result"])>0:
            for i in res["result"]:
                print(i)
        else:
            print("LLM Fails to write efficient query")
    except Exception as e:
        print(str(e))    