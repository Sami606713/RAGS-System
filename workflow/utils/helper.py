from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever

from vectorStore.vectorStore import get_embeddings
def load_vector_store(vector_store_path: str = "my_faiss_index3"):
    """
    Load the FAISS vector store from the specified path.
    """
    try:
        embeddings = get_embeddings()
        print(f">> Loading vector store from {vector_store_path}...")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded successfully!")
        return vector_store
    except Exception as e:
        raise Exception(f"Error loading vector store: {str(e)}")
    

def load_tfidf_store(vector_store_path: str = "tf_idf"):
    """
    Load the TF-IDF vector store from the specified path.
    """
    try:
        print(f">> Loading TF-IDF vector store from {vector_store_path}...")
        tfidf_retriever = TFIDFRetriever.load_local(vector_store_path, allow_dangerous_deserialization=True)
        print("✅ TF-IDF vector store loaded successfully!")
        return tfidf_retriever
    except Exception as e:
        raise Exception(f"Error loading TF-IDF vector store: {str(e)}")