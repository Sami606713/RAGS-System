from workflow.nodes.queryWriter import query_rewriter
from workflow.nodes.getContext import get_relevant_doc
from workflow.nodes.reRanker import ReRanker
from workflow.nodes.docReordering import ReOrderingDocument
from workflow.nodes.queryDecomposer import query_decomposition
from workflow.nodes.queryExpension import query_expansion
from workflow.nodes.generator import generate_answer
from langgraph.graph import StateGraph,START,END
from workflow.states.states import AgentState
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.runnables import RunnableLambda
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def create_workflow():
    """
    Create the workflow for the RAGS system.
    """
    # define the nodes
    workflow = StateGraph(AgentState)

    # Add the nodes (using correct function/class names from imports)
    workflow.add_node("Query Rewriter", RunnableLambda(query_rewriter))
    workflow.add_node("Get Context", RunnableLambda(get_relevant_doc))
    workflow.add_node("Query Expansion", RunnableLambda(query_expansion))
    workflow.add_node("Query Decomposition", RunnableLambda(query_decomposition))
    workflow.add_node("Doc ReOrdering", RunnableLambda(ReOrderingDocument))
    workflow.add_node("ReRanking", RunnableLambda(ReRanker))
    workflow.add_node("Generator", RunnableLambda(generate_answer))

    # Define the edges
    workflow.add_edge(START, "Query Rewriter")

    workflow.add_edge("Query Rewriter", "Get Context")
    workflow.add_edge("Query Rewriter", "Query Expansion")
    workflow.add_edge("Query Rewriter", "Query Decomposition")

    workflow.add_edge("Get Context", "Doc ReOrdering")
    workflow.add_edge("Query Expansion", "Doc ReOrdering")
    workflow.add_edge("Query Decomposition", "Doc ReOrdering")

    workflow.add_edge("Doc ReOrdering", "ReRanking")
    workflow.add_edge("ReRanking", "Generator")

    workflow.add_edge("Generator", END)


    # checkpointer = InMemoryDocstore()

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