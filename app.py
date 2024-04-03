import time
import streamlit as st
from clarifai.modules.css import ClarifaiStreamlitCSS
import streamlit.components.v1 as components
from langchain.schema import HumanMessage, AIMessage
from utils.utils import (connect_db, init_db, 
                         llm_model, prompt, 
                         sql_chain, sql_agent, 
                         agent_response, chain_response)


st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

st.title("Chat with your Structured data 📚")
st.write("##")
config_type=st.checkbox("Advanced configuration", key="config_type")

with open('./styles.css') as f:
  st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)
  
with st.sidebar:
  with st.form(key="DB_config"):
    st.title("Database Configuration")
    st.write("##")
    
    # Database connection details
    connection_uri = st.text_input("**Connection URI(PostgreSQL database URI):**",type="password", placeholder="Ex postgresql://user:password@localhost:5432/dbname")
    st.write("##")
    
    CLARIFAI_PAT = st.text_input("**Clarifai PAT:**",type="password", placeholder="Ex: CLARIFAI_PAT")
    st.write("##")
    
    llm=st.text_input("**LLM Model URL (optional):**", placeholder="Ex: https://clarifai.com/openai/chat-completion/models/gpt-4-turbo")
    configure = st.form_submit_button(label='Configure')
      
if connection_uri and CLARIFAI_PAT:
  if "connect_uri" not in st.session_state.keys():
    st.session_state["connect_uri"] = connection_uri

if "chat_history" not in st.session_state.keys():
  st.session_state['chat_history'] = [{"role": "assistant", "content": "How may I help you?"}]
  
if not llm:
  llm = "https://clarifai.com/openai/chat-completion/models/gpt-4-turbo" 
  
if connection_uri and CLARIFAI_PAT:   
  connect_db(connection_uri)
  clarifai_llm=llm_model(llm, pat=CLARIFAI_PAT)
  agent=sql_agent(clarifai_llm, init_db(connection_uri))
  chain=sql_chain(clarifai_llm, init_db(connection_uri))

def previous_chats():
  chat_list = []
  for message in st.session_state['chat_history']:
    with st.chat_message(message["role"]):
      if message["role"] == 'user':
        msg = HumanMessage(content=message["content"])
      else:
        msg = AIMessage(content=message["content"])
      chat_list.append(msg)
      st.write(message["content"])

def chatbot(agent, chain):
  if message := st.chat_input(key="input"):
    st.chat_message("user").write(message)
    st.session_state['chat_history'].append({"role": "user", "content": message})
    
    with st.chat_message("assistant"):
      with st.spinner("Thinking..."):
        try:
          start_time = time.time()
          if config_type:
            response = agent_response(agent, message)
          else:
            response = chain_response(chain, message, prompt())
            
          end_time = time.time()
          time_taken = end_time - start_time
          response = response + "\n\n" + f"{time_taken:.2f} seconds"
          st.write(response)
          message = {"role": "assistant", "content": response}
          st.session_state['chat_history'].append(message)
        except Exception as e:
          response = f"An error occurred: {e}"

if "connect_uri" in st.session_state.keys():
  previous_chats()
  chatbot(agent, chain)


