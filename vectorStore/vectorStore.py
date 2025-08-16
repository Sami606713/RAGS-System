from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from langchain.retrievers import EnsembleRetriever
from utils.helper import get_bm25_retriever,Query_Optimizer
# from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
from typing import List
import os
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain_cohere.chat_models import ChatCohere
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
# from langchain_voyageai import VoyageAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uuid import uuid4

load_dotenv()
# Initialize OpenAI embeddings

def get_embeddings():
    """
    Initialize OpenAI embeddings with the API key from environment variables.
    """
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")
    return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"),skip_empty=True,show_progress_bar=True)
    # if "VOYAGE_API_KEY" not in os.environ:
    #     raise ValueError("VOYAGE_API_KEY is not set")
    # voyage = VoyageAIEmbeddings(voyage_api_key=os.getenv("VOYAGE_API_KEY"), model="voyage-context-3")
    


# embeddings = get_embeddings()
# Initialize FlashRank reranker
# llm = ChatCohere(temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
compressor = CohereRerank(model="rerank-english-v3.0",cohere_api_key =os.getenv("COHERE_API_KEY"))


def add_to_vector_store(docs_chunks: List[Document], batch_size: int = 32, vector_store_path = "my_faiss_index3"):
    successful_docs = []
    failed_docs = []
    try:
        embeddings = get_embeddings()
        print(f">> Starting embedding for {len(docs_chunks)} documents...\n")
        if os.path.exists(vector_store_path):
            print(">> Loading the index <<")
            vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        else:
            print(">> Creating the index  <<")
            dimension = len(embeddings.embed_query("hello world"))
            index = faiss.IndexFlatL2(dimension)
            vector_store = FAISS(
                embedding_function=embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        uuids = [str(uuid4()) for _ in docs_chunks]
        print(f"\n📦 Preparing to insert {len(docs_chunks)} documents into FAISS...\n")
        for i in tqdm(range(0, len(docs_chunks), batch_size), desc="🔍 Embedding & Inserting", unit="batch"):
            try:
                batch_docs = docs_chunks[i:i+batch_size]
                batch_ids = uuids[i:i+batch_size]
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                successful_docs.extend(batch_docs)
            except Exception as e:
                print(f"Error during batch insertion: {str(e)}")
                failed_docs.extend(docs_chunks[i:i+batch_size])

        vector_store.save_local(vector_store_path)
        print("✅ Data insertion successful!\n")
        print(f"📦 Successfully inserted {len(successful_docs)} documents.")
        print(f"❌ Failed to insert {len(failed_docs)} documents.")
        return {
            "status": "success",
            "vector_store": vector_store,
            "num_documents": len(docs_chunks),
            "successful_docs": len(successful_docs),
            "failed_docs": len(failed_docs)
        }
    except Exception as e:
        raise Exception(f"Error in add_to_vector_store: {str(e)}")
