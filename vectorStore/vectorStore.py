from pymilvus import MilvusClient
# from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

from typing import List

# =============Fais Setup============#
from typing import List
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from tqdm import tqdm


def add_to_vector_store(docs_chunks: List[Document], batch_size: int = 64, vector_store_path = "my_faiss_index"):
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
        batch_docs = docs_chunks[i:i+batch_size]
        batch_ids = uuids[i:i+batch_size]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    vector_store.save_local(vector_store_path)
    print("✅ Data insertion successful!\n")
    return {
        "status": "success",
        "vector_store": vector_store,
        "num_documents": len(docs_chunks)
    }


from langchain.retrievers import EnsembleRetriever
from utils.helper import get_bm25_retriever

def get_hybrid_retriever(faiss_index_path: str, docs: List[Document], alpha: float = 0.5):
    faiss_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = get_bm25_retriever(docs, k=10)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[1 - alpha, alpha]
    )
    return hybrid_retriever


def GetContext(query: str, docs: List[Document]):
    hybrid_retriever = get_hybrid_retriever("my_faiss_index", docs, alpha=0.5)
    results = hybrid_retriever.invoke(query)
    source = [doc.metadata.get("source", "Unknown") for doc in results]
    return {
        "query": query,
        "results": [
            {
                "page_content": res.page_content,
                "metadata": res.metadata,
                "score" : res.metadata.get("score", None),
                "source":source
            } for res in results
        ]
    }



if __name__ == "__main__":
   pass
