# generator/generator.py

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load LLM for generation
llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
