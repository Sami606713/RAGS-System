from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store

# Get Context Using Refine Query
def get_relevant_doc(state:AgentState)->AgentState:
    try:
        print("Get Relevant...")
        query = state['rewrite_question']

        vector_store = load_vector_store()
        results = vector_store.similarity_search(
            query,
            k=5
        )

        return {
            "context":results
        }
    except Exception as e:
        print(str(e))
        return str(e)