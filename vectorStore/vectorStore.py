from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from langchain.retrievers import EnsembleRetriever
from utils.helper import get_bm25_retriever
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
from typing import List
import os
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from uuid import uuid4

load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
# Initialize FlashRank reranker
llm = Cohere(temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
compressor = CohereRerank(model="rerank-english-v3.0",cohere_api_key =os.getenv("COHERE_API_KEY"))

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


def get_hybrid_retriever(faiss_index_path: str, docs: List[Document], alpha: float = 0.5):
    faiss_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 5})
    bm25_retriever = get_bm25_retriever(docs, k=10)
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[1 - alpha, alpha]
    )
    return hybrid_retriever

# Add a query expension by using MultiQueryRetriever
def get_multiquery_hybrid_retriever(docs: List[Document], faiss_index_path: str, alpha: float = 0.5):
    # Build hybrid retriever (BM25 + FAISS)
    hybrid_retriever = get_hybrid_retriever(faiss_index_path, docs, alpha=alpha)

    # Add MultiQueryRetriever on top of hybrid retriever
    llm = ChatOpenAI(temperature=0.0, model="gpt-4o-mini")  # or "gpt-3.5-turbo"
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )

    return multi_query_retriever

# get context
def GetContext(query: str, docs: List[Document]):
    if not docs:
        return {
            "query": query,
            "results": [],
            "error": "❌ No documents provided to GetContext(). Ensure docs are passed correctly."
        }

    # ✅ Step 1: Optimize queryx
    print(">> Applying Query Expansion...")
    # ✅ Step 2: Create hybrid + multiquery retriever
    multiquery_hybrid_retriever = get_multiquery_hybrid_retriever(docs, faiss_index_path="my_faiss_index", alpha=0.5)
    print(">> Creating MultiQuery Hybrid Retriever...")

    # ✅ Step 3: Wrap with compression retriever for reranking
    print(">> Creating Contextual Compression Retriever...")
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multiquery_hybrid_retriever,
        base_compressor=compressor
    )

    # ✅ Step 4: Retrieve
    results = compression_retriever.invoke(query)

  
    source = [doc.metadata.get("source", "Unknown") for doc in results]

    return {
        "query": query,
        "results": [
            {
                "page_content": res.page_content,
                "metadata": res.metadata,
                "source": res.metadata.get("source", "Unknown"),
                "score": res.metadata.get("score", 0.0),
                "souces2": source
            } for res in results
        ]
    }


if __name__ == "__main__":
   pass
