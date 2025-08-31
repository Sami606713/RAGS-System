from workflow.states.states import AgentState
from workflow.prompt.prompt import generator_prompt
from workflow.models.loadModel import load_model
from workflow.utils.helper import format_context


def generate_answer(state: AgentState):
    """
    Generate structured academic-style answers based on reranked context.
    """
    try:
        print("Ans Generation")
        context = state['context']
        query = state['question']

        print("Formatting context...")
        combined_context = format_context(context)
        print("Formatting Done..")

        # Prompt template
        prompt = generator_prompt().format(
            question=query,
            combined_context=combined_context
        )

        answer = load_model().invoke(prompt)
        print("Answer Generated")

        return {'answer': answer.content}

    except Exception as e:
        print("Error in answer generation:", str(e))
        return {'error': str(e)}