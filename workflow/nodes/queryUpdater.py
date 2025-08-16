from langgraph.types import Command
from workflow.states.states import AgentState
from workflow.models.loadModel import get_query_refine_model
from langgraph.graph import END

def query_updater(state: AgentState):
    # Extract original query and top context
    original_query = state["rewrite_question"]
    top_context = state["reranked_context"][:10]
    count = state.get("count",0)
    print(f"Reasning....{count}")
    if(count>=2):
        return Command(
            goto=END,
            update={
                "answer":"I Can't find the answer form the given context"
            }
        )
    else:
        # Build prompt with clearer instructions
        prompt = (
            "You are a helpful assistant. Your task is to refine the user's query "
            "to make it more precise and aligned with the provided context.\n\n"
            f"Context:\n{top_context}\n\n"
            f"Original Query:\n{original_query}\n\n"
            "Refined Query:"
        )

        # Invoke the query refinement model
        refined_query = get_query_refine_model().invoke(prompt)

        # Update state with the refined query
        state['rewrite_question'] = refined_query.refine_query

        return {
            "rewrite_question":refined_query.refine_query,
            "context":[],
            "reranked_context":[],
            "count":count+1
        }