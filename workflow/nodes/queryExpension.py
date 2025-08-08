from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store
from workflow.models.loadModel import load_model
from langchain.retrievers.multi_query import MultiQueryRetriever


# Apply Query Expansion and get results for each query
def query_expansion(state:AgentState):
       """
       Query Expansion using MultiQueryRetriever.
       This function retrieves relevant documents based on the refined query.
       Args:
              state (AgentState): The current state containing the refined query.
       Returns:
              dict: A dictionary containing the context retrieved for the query.
       """
       try:
              vector_store = load_vector_store()
              retriever_from_llm = MultiQueryRetriever.from_llm(
              retriever=vector_store.as_retriever(), llm=load_model()
              )

              # get the question
              question = state['rewrite_question']

              context = retriever_from_llm.invoke(question)

              return {
                     "context":context
              }
       except Exception as e:
           print(f"Error in query expansion: {str(e)}")
           return {'error': str(e)}