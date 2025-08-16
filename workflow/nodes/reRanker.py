# Reranker
from workflow.states.states import AgentState
from langchain.retrievers import ContextualCompressionRetriever
# from langchain_community.document_compressors import FlashrankRerank
from langchain_cohere import CohereRerank
from workflow.utils.helper import load_vector_store
import os



def ReRanker(state:AgentState):
    try:
        print("Reranking....")
        updated_context = state['context']
        query = state['rewrite_question']
        sources = list(set([doc.metadata.get('source', 'Unknown') for doc in updated_context]))

        # reranked docs
        compressor = CohereRerank(model="rerank-english-v3.0", cohere_api_key=os.getenv("COHERE_API_KEY"))

        # Rerank documents
        reranked_docs = compressor.compress_documents(updated_context, query)

        # Sort by score descending (higher score = more relevant)
        reranked_docs_sorted = sorted(
            reranked_docs, key=lambda doc: doc.metadata.get("score", 0), reverse=True
        )

        # Option 1: Return top 3 (or top-k)
        top_docs = reranked_docs_sorted[:40]
        return {
            "reranked_context":top_docs,
            "sources":sources
        }
    except Exception as e:
        print(f"Error in reranking: {str(e)}")
        return {
            "error": str(e)
        }
