from pymilvus import MilvusClient
# from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from tqdm import tqdm
from dotenv import load_dotenv
from typing import List
import os

load_dotenv()

# Get credentials from environment
URI = os.getenv("ZILLOW_URI")
TOKEN = os.getenv("ZILLOW_PASSWORD")  

# Initialize Milvus client
client = MilvusClient(
    uri=URI,
    token=f"db_94eed7ebf0f2204:{TOKEN}"
)

if client.has_collection(collection_name="Rags"):
    print(">> Collection Already Existis")
else:
    client.create_collection(
        collection_name="Rags",
        dimension=1536,  # OpenAIEmbeddings uses 1536 dimensions
    )

embeddings = OpenAIEmbeddings()

from typing import List

# def add_to_vector_store(docs_chunks: List[Document]):
#     """
#     Embeds document chunks using OpenAI and stores them in Milvus.
#     Args:
#         docs_chunks (List[Document]): List of LangChain Document objects.
#     Returns:
#         dict: Success message.
#     """
   

#     vectors = []
#     print(f">> Starting embedding for {len(docs_chunks)} documents...\n")
#     for chunk in tqdm(docs_chunks, desc="🔍 Embedding Documents", unit="doc"):
#         vector = embeddings.embed_documents([chunk.page_content])[0]
#         vectors.append(vector)

#     data = [
#         {
#             "id": i,
#             "vector": vectors[i],
#             "text": docs_chunks[i].page_content,
#             "subject": "history"
#         }
#         for i in range(len(vectors))
#     ]

#     print(f"\n📦 Preparing to insert {len(data)} documents into Milvus...\n")

#     for i in tqdm(range(1), desc="📤 Inserting into Milvus"):
#         res = client.insert(collection_name="Rags", data=data)

#     print("✅ Data insertion successful!")
#     return {"status": "success"}


# def GetContext(query:str):
#     query_vectors = embeddings.embed_documents([query])

#     res = client.search(
#         collection_name="Rags",  # target collection
#         data=query_vectors,  # query vectors
#         limit=5,  # number of returned entities
#         output_fields=["text", "subject"],  # specifies fields to be returned
#     )
#     print(">> Context: ",res)
#     return res[0][0].entity

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
