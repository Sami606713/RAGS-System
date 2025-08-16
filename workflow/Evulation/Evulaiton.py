from workflow.states.states import AgentState
from workflow.models.loadModel import evulation_model

def evaluate_response(state:AgentState):
    """ 
    Evaluate the generated response
    """

    answer = state['answer']
    query = state['rewrite_question']

    evulation = evulation_model().invoke(f"""
    User Query: {query}
    Response: {answer}
    """)
    print("Evaluation Result: ", evulation.evaluation_result)
    if evulation.evaluation_result.lower() == "no":
        return "no"
    
    if evulation.evaluation_result.lower() == "yes":
        return "yes"