from langchain_community.vectorstores import FAISS
from vectorStore.vectorStore import get_embeddings
def load_vector_store(vector_store_path: str = "my_faiss_index2"):
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