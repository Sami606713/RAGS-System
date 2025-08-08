from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store
from workflow.models.loadModel import get_query_decomposer_model

def query_decomposition(state: AgentState):
    try:
        print("Query Decomposition...")

        # Get the user query
        question = state['rewrite_question']
        vector_store = load_vector_store()

        # Generate sub-queries (assumes composer_llm returns a list of strings)
        sub_queries_obj = get_query_decomposer_model().invoke(question)
        sub_queries = sub_queries_obj.compose_query

        sub_query_context = []

        for idx, query in enumerate(sub_queries):
            print(f"Processing sub-query {idx + 1}: {query}")
            context = vector_store.similarity_search(query, k=5)
            sub_query_context.extend(context) # Use extend to flatten the list

        return {
            "context": sub_query_context
        }
    except Exception as e:
        print(f"Error in query decomposition: {str(e)}")
        return {
            "error": str(e)
        }