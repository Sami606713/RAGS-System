from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from langchain.retrievers import EnsembleRetriever
from utils.helper import get_bm25_retriever,Query_Optimizer
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
from langchain_cohere import ChatCohere
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from uuid import uuid4

load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
# Initialize FlashRank reranker
llm = ChatCohere(temperature=0, cohere_api_key=os.getenv("COHERE_API_KEY"))
compressor = CohereRerank(model="rerank-english-v3.0",cohere_api_key =os.getenv("COHERE_API_KEY"))

compressor = FlashrankRerank()

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


def GetQueryContext(query: str,faiss_index_path: str="index.faiss"):
    faiss_vector_store = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

    print(">> Creating FAISS Retriever...")
    faiss_retriever = faiss_vector_store.as_retriever(search_kwargs={"k": 510})
    # Use the retriever to get the context for the query

    print(">> Creating MultiQuery Retriever...")
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=faiss_retriever,
        llm=llm
    )
    # results = multi_query_retriever.invoke(query)

    # Apply reranker
    print(">> Applying Reranker...")
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=multi_query_retriever,
        base_compressor=compressor
    )
    print(">> Query ",query)
    refine_query = Query_Optimizer(query)

    print(">> After Query Optimizer ",refine_query)
    results = compression_retriever.invoke(refine_query)

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
    context = GetQueryContext("What is the price of methanol?")
    print("Context Retrieved:")
    for res in context['results']:
         print(f"Source: {res['source']}, Content: {res['page_content'][:100]}...")  # Print first 100 chars of content
