from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store

# Get Context Using Refine Query
def get_relevant_doc(state:AgentState)->AgentState:
    try:
        print("Get Relevant...")
        query = state['rewrite_question']

        vector_store = load_vector_store()

        # Retrieve results from both
        context = vector_store.similarity_search(query, k=5)
        
        return {
            "context":context
        }
    except Exception as e:
        print("Error in answer generation:", str(e))
        return {'error': str(e)}