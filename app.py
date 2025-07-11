import streamlit as st
from agent.agent import RunAgent
from agno.agent import Agent, RunResponse
from vectorStore.vectorStore import GetQueryContext
from langchain_openai import ChatOpenAI
from typing import Iterator
# from generator import generate_answer
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# fun for generating answers
# Load LLM for generation
llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")


def generate_answer(query: str) -> str:
    """
    Retrieve context and generate an answer using the LLM. 
    """
    context_text = GetQueryContext(query, faiss_index_path="my_faiss_index")
    
    prompt = f"""
    You are a knowledgeable assistant. Follow these instructions strictly:

    Context Usage Only:
    Answer all user questions using only the information provided in the context.

    Out-of-Context Questions:
    If the user's question is not covered by the context, respond:
    "I have knowledge related to this context, which covers the following areas:"
    (Then provide a brief summary of the context.)
    "Please ask a question within this scope."

    Answer Style:

    Use the context to explain the answer.

    Keep responses concise but informative—not too short, not too long.

    Do not add any information that is not explicitly found in the context.

    Context:
    {context_text}

    Question: {query}
    Answer:

    Source: 
    (List the sources of the information used to answer the question, if applicable.)
    """
    response = llm.invoke(prompt)
    return response.content

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
    print(f"Original Query: {user_input}")
    # Run agent

    # Show bot response
    with st.chat_message("Bot"):
        with st.spinner("Thinking..."):
            response = generate_answer(user_input)
            st.markdown(response)
    # Save to chat history
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response))
