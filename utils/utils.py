import psycopg2
import streamlit as st  
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits import create_sql_agent
from langchain.llms import Clarifai
from langchain.globals import set_debug

set_debug(True)
def connect_db(connect_uri: str):
    try:
        conn = psycopg2.connect(connect_uri)
        conn.close()
    except Exception as e:
        st.write(e)
        

def init_db(init_uri: str):
    if init_uri.startswith("postgres://"):
       init_uri ="postgresql+psycopg2" + init_uri[len("postgres"):]
    #st.write(init_uri)  
    try:
        db = SQLDatabase.from_uri(init_uri)
        return db
    except Exception as e:
        st.write(e)
    
def llm_model(url: str, pat):
    clarifai_llm = Clarifai(model_url=url, pat=pat)
    return clarifai_llm
    
def prompt():
    PROMPT="""
You are a PostgreSQL expert tasked with answering questions based on a database. Given an input question, your task is to:

1) Create a syntactically correct PostgreSQL query to retrieve the necessary information.
2)Execute the query and return the results to answer the input question.
3)Ensure that the query selects only the required columns from the relevant tables.
5)Utilize only the column names present in the provided tables.
6)Avoid querying for columns that do not exist.
7) If the question involves "today", use the CURRENT_DATE function to get the current date.
8) Order the results to return the most informative data in the database.
9) Dont add code blocks for sql query.
10) Please adhere to the following format while giving outputs:
11)Consider what type of datatype is there in each column and match it with user query.

Important! You should provide the postgres code along with query output from database for the following question, do not add backticks or codeblock while giving sql query as it might lead to error

Final answer from database

{question}
"""
    return PROMPT

def sql_chain(llm_model, database):
    db_chain = SQLDatabaseChain.from_llm(llm=llm_model, db=database, top_k=500)
    return db_chain

def sql_agent(llm_model, database):
    db_agent = create_sql_agent(llm_model, db=database, verbose=True)
    return db_agent

def chain_response(db_chain, question, prompt):
    response = db_chain.run(prompt.format(question=question))
    return response

def agent_response(db_agent, question):
    response = db_agent.invoke(question)
    return response["output"]
