# Apply Long Context Reorder
from langchain_community.document_transformers import LongContextReorder
from workflow.states.states import AgentState

def ReOrderingDocument(state:AgentState):
     try:
          context = state['context']

          reordering = LongContextReorder()
          reordered_docs = reordering.transform_documents(context)

          return {
               "context":reordered_docs
          }
     except Exception as e:
        return {'error': str(e)}
