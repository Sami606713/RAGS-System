from agno.agent import Agent
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from vectorStore.vectorStore import GetContext
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv

load_dotenv()
def RunAgent(query):
    """
    This agent can run hte query and return the response
    retriever_tool can accept query and user_id and return the response
    """
    try:
        agent = Agent(
            tools=[GetContext,
                   ReasoningTools(add_instructions=True),
                    ThinkingTools(add_instructions=True)
                    ],
            description = "This agent strictly processes user queries using ONLY the provided context. It must not use external knowledge or assumptions beyond the context. "
            "If the exact answer is not found, it must reason based on the available information to generate a helpful response. "
            "If reasoning is not possible from the given context, the agent must clearly state that it cannot answer the query and prompt the user to try a related query. "
            "At no point should the agent fabricate information or rely on knowledge not present in the provided context.",
            instructions = [
                """
                ROLE:
                - You are a context-restricted assistant representing <BOT_NAME>.
                - Your sole purpose is to assist users using ONLY the information provided in the given context/document.

                RESPONSE RULES:
                1. Use ONLY the information found in the provided context. Do not use external knowledge, prior training data, or general facts.
                2. Your answers must be clear, concise, and accurate — but not overly long. Focus on providing complete responses in a compact form.
                3. DO NOT assume, guess, or invent information. If a detail is not clearly present or inferable from the context, do NOT include it.
                4. If the question cannot be answered from the context alone, respond exactly with:
                "Apologies; I am not sure about that. Please head over to <SUPPORT_URL> for some additional help from our team."
                5. NEVER imply or suggest that you have knowledge outside the context.
                6. Each answer MUST include the source/document name at the end (e.g., "— Source: Hydrogen Bunkering at Ports").

                COMPLIANCE POLICY:
                - This is a ZERO-TOLERANCE instruction set.
                - Any response containing non-contextual knowledge or fabricated content is strictly prohibited.
                """
            ],

            show_tool_calls=True,
            markdown=True,
            model=OpenAIChat(id="gpt-4o")
        )


        # Run Agent
        response: RunResponse = agent.run(query, stream=False,structured_outputs=True)
        
        return response.content
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    query = "tell me about enery and climate changes?"
    response = RunAgent(query=query)
    print(">> Response: ",response)
