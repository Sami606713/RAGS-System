import streamlit as st
from workflow.workflow import create_workflow
import os
from uuid import uuid4

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set Streamlit layout
st.set_page_config(page_title="Document Chat App", layout="wide")
st.title("📄 Document Chat App")

# Initialize workflow app
app = create_workflow()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of dicts: {"role": "user"/"assistant", "content": str}

# Function to stream response
def stream_response():
    def response_generator():
        for msg, metadata in app.stream({"question": user_input}, stream_mode="messages"):
            node_name = metadata.get("langgraph_node", "Unknown Node")
            status.update(label=f"{node_name}", state="running", expanded=True)
            if msg.content and metadata["langgraph_node"] == "Generator":
                yield msg.content
    return response_generator()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask something about your document...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.status("Thinking…", expanded=True) as status:
            full_response = st.write_stream(stream_response())

    # Save assistant response
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
