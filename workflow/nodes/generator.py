from workflow.states.states import AgentState
from workflow.prompt.prompt import generator_prompt
from workflow.models.loadModel import load_model
def generate_answer(state: AgentState):
    try:
        print("Ans Generation")
        updated_context = state['context']
        query = state['rewrite_question']
        sources = state['sources']

        sources_str = ", ".join(sources)
        # Combine documents into a structured block
        print("Combining..")
        combined_context = "\n\n".join([
            f"""
**Summary**: {doc.metadata.get('summary', 'N/A')}

{doc.page_content.strip()}
""" for doc in updated_context
        ])

        print("Combined Context: ", combined_context)
        # Prompt template (cleaner, more guided)
        prompt = generator_prompt().format(
            query=query,
            combined_context=combined_context,
            sources=sources_str
        )
        # Generate the answer
        answer = load_model().invoke(prompt)

        return {
            'answer': answer.content
        }

    except Exception as e:
        return {'error': str(e)}