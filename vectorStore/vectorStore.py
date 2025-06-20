from pymilvus import MilvusClient
# from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

embeddings = OpenAIEmbeddings()

from typing import List

# =============Fais Setup============#
from typing import List
from langchain_core.documents import Document
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from tqdm import tqdm


def add_to_vector_store(docs_chunks: List[Document],batch_size:int = 64,vector_store_path = "my_faiss_index"):
    """
    Embeds document chunks and stores them in a FAISS vector store.
    
    Args:
        docs_chunks (List[Document]): List of LangChain Document objects.
    
    Returns:
        dict: Status message and vector store.
    """
    print(f">> Starting embedding for {len(docs_chunks)} documents...\n")

    

    if os.path.exists(vector_store_path):
        print(">> Loading the index <<")
        vector_store = FAISS.load_local(vector_store_path, embeddings,allow_dangerous_deserialization=True)
    else:
        print(">> Creating the index  <<")
        # Create an index using the dimensionality of one sample embedding
        dimension = len(embeddings.embed_query("hello world"))
        index = faiss.IndexFlatL2(dimension)
        # Initialize vector store
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    # Generate unique IDs for documents
    uuids = [str(uuid4()) for _ in docs_chunks]

    print(f"\n📦 Preparing to insert {len(docs_chunks)} documents into FAISS...\n")
    # Loop over documents in batches
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

def GetContext(query:str):
    vector_store = FAISS.load_local("my_faiss_index", embeddings,allow_dangerous_deserialization=True)

    results = vector_store.similarity_search(
    query,
    k=2,
    # filter={"source": "tweet"},
    )
    # for res in results:
    #     print(f"* {res.page_content} [{res.metadata}]")
    
    return {"Context":results}



if __name__ == "__main__":
   pass
