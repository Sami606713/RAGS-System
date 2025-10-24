from workflow.nodes.queryWriter import query_rewriter
from workflow.nodes.getContext import get_relevant_doc
from workflow.nodes.reRanker import ReRanker
from workflow.nodes.docReordering import ReOrderingDocument
from workflow.nodes.queryDecomposer import query_decomposition
from workflow.nodes.ragFusion import rag_fusion  # Updated: Use RAG Fusion instead
from workflow.nodes.generator import generate_answer
from workflow.nodes.queryUpdater import query_updater
from workflow.Evulation.Evulaiton import evaluate_response
from langgraph.graph import StateGraph,START,END
from workflow.states.states import AgentState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableLambda
from langchain.embeddings import init_embeddings
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
embeddings = init_embeddings("openai:text-embedding-3-small")

def create_workflow():
    """
    Create the workflow for the RAGS system.
    """
    # define the nodes
    workflow = StateGraph(AgentState)

    # Add the nodes (using correct function/class names from imports)
    workflow.add_node("Query Rewriter", RunnableLambda(query_rewriter))
    workflow.add_node("RAG Fusion", RunnableLambda(rag_fusion))  # Updated: RAG Fusion node
    workflow.add_node("Generator", RunnableLambda(generate_answer))

    # Define the edges - Simplified workflow with RAG Fusion
    workflow.add_edge(START, "Query Rewriter")
    workflow.add_edge("Query Rewriter", "RAG Fusion")
    workflow.add_edge("RAG Fusion", "Generator")
    workflow.add_edge("Generator", END)

    # checkpointer = InMemorySaver()

    return workflow.compile()


if __name__ == "__main__":
    # Create the workflow
    app = create_workflow()
    question =  "What type of fuel do most ships use?"
    results = app.invoke({"question":question})

    # Print the final result
    if 'answer' in results:
        print(results['answer'])
    else:
        print(results['error'])