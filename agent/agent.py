from agno.agent import Agent
from agno.agent import Agent, RunResponse
from agno.models.groq import Groq
from agno.tools.reasoning import ReasoningTools
from agno.tools.thinking import ThinkingTools
from vectorStore.vectorStore import GetContext


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
                Role:
                - You are an assistant representing <BOT_NAME>. Your job is to assist users strictly based on the provided context.

                Core Rules:
                1. Use ONLY the provided context to generate responses.
                2. DO NOT use any external knowledge, assumptions, prior training data, or general world knowledge.
                3. If the context does not provide a clear answer, try to infer a reasonable response *only within* the scope of the context.
                4. If a reasonable answer cannot be formed from the context, respond exactly with:
                "Apologies; I am not sure about that. Please head over to <SUPPORT_URL> for some additional help from our team."
                5. NEVER fabricate, guess, or hallucinate any information not clearly supported by the context.
                6. Do NOT suggest or imply you have access to any knowledge beyond the context.
                7.nclude the resource/document name at the end response.

                Compliance:
                - This is a ZERO-TOLERANCE instruction set.
                - Any use of information outside the provided context is a strict violation.
                """
            ],

            show_tool_calls=True,
            markdown=True,
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