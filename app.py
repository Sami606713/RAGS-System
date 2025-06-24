import streamlit as st
from agent.agent import RunAgent
from utils.helper import Query_Optimizer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Streamlit layout
st.set_page_config(page_title="Document Chat App", layout="wide")
st.title("📄 Document Chat App")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for speaker, message in st.session_state.chat_history:
    with st.chat_message(name=speaker):
        st.markdown(message)

# Chat input
user_input = st.chat_input("Ask something about your document...")

if user_input:
    # Show user message
    with st.chat_message("You"):
        st.markdown(user_input)

    # Optimize query
    # user_input = Query_Optimizer(user_input)
    # Run agent
    response = RunAgent(query=user_input)

    # Show bot response
    with st.chat_message("Bot"):
        st.markdown(response)

    # Save to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
