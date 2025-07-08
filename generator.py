# generator/generator.py

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
from vectorStore.vectorStore import GetQueryContext
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

if __name__ == "__main__":
    user_query ="can you create a table for fuels rates"

    answer = generate_answer(user_query)
    print(f"\nAnswer:\n{answer}")
