from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store,load_tfidf_store
from workflow.models.loadModel import get_query_decomposer_model

def query_decomposition(state: AgentState):
    try:
        print("Query Decomposition...")

        # Get the user query
        question = state['rewrite_question']
        vector_store = load_vector_store()
        tf_idf_vector_store = load_tfidf_store()

        # Generate sub-queries (assumes composer_llm returns a list of strings)
        sub_queries_obj = get_query_decomposer_model().invoke(question)
        sub_queries = sub_queries_obj.compose_query

        sub_query_context = []

        for idx, query in enumerate(sub_queries):
            print(f"Processing sub-query {idx + 1}: {query}")
            context1 = vector_store.similarity_search(query, k=5)
            context2 = tf_idf_vector_store.invoke(query)
            sub_query_context.extend(context1)  # Use extend to flatten the list
            sub_query_context.extend(context2)

        return {
            "context": sub_query_context
        }
    except Exception as e:
        print("Error in answer generation:", str(e))
        return str(e)