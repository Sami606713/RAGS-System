from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store,load_tfidf_store

# Get Context Using Refine Query
def get_relevant_doc(state:AgentState)->AgentState:
    try:
        print("Get Relevant...")
        query = state['rewrite_question']

        vector_store = load_vector_store()
        tf_idf_vector_store = load_tfidf_store()

        # Retrieve results from both
        vector_results = vector_store.similarity_search(query, k=5)
        tfidf_results = tf_idf_vector_store.invoke(query)

        # Combine results
        combined_results = vector_results + tfidf_results  

        return {
            "context":combined_results
        }
    except Exception as e:
        print("Error in answer generation:", str(e))
        return str(e)