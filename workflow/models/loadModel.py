from langchain_openai import ChatOpenAI
from workflow.states.states import RewriterQuery,QueryDecomposer,Evaluation

def load_model() -> ChatOpenAI:
    llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
    return llm

def get_query_refine_model() -> RewriterQuery:
    """
    Get the query refinement model using the provided LLM.
    """
    llm = load_model()
    return llm.with_structured_output(RewriterQuery)


def get_query_decomposer_model() -> RewriterQuery:
    """
    Get the query decomposition model using the provided LLM.
    """
    llm = load_model()
    return llm.with_structured_output(QueryDecomposer)

def evulation_model():
    """
    Get the evaluation model using the provided LLM.
    """
    llm = load_model()
    return llm.with_structured_output(Evaluation)