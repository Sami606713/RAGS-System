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
    You are an expert assistant. Use ONLY the following context to answer the user's question. If the answer is not in the context, say you don't know.
    you can use the context to explain the answer but note that answer is not too long not too short.
    but note that the answer should be in the context.

    Context:
    {context_text}

    Question: {query}
    Answer:
    """
    response = llm.invoke(prompt)
    return response

if __name__ == "__main__":
    user_query ="can you create a table for fuels rates"

    answer = generate_answer(user_query)
    print(f"\nAnswer:\n{answer}")
