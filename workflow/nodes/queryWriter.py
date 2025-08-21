from workflow.states.states import AgentState
from workflow.models.loadModel import get_query_refine_model
# query refine node
def query_rewriter(state:AgentState)-> AgentState:
    """
    Query refine
    """
    try:
        print("Query Refiner....")
        query = state['question']

        prompt = f"""You are a helpful assistant your task is to refine the query but refine the query in this way donot loss the context of original query maintain the context of original query.
        Original Query: {query}
        """
        refine_query = get_query_refine_model().invoke(prompt)

        print("Query Refiner: ",refine_query)
        state['rewrite_question'] = refine_query.refine_query

        return state
    except Exception as e:
        print("Error in answer generation:", str(e))
        return str(e)