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
    This agent can run the query and return the response.
    The retriever_tool can accept query and user_id and return the response.
    """
    try:
        agent = Agent(
            tools=[GetContext,
                   ReasoningTools(add_instructions=True),
                   ThinkingTools(add_instructions=True)
                   ],
            description=(
                "This agent strictly processes user queries using ONLY the provided context. "
                "It must not use external knowledge or assumptions beyond the context. "
                "If the exact answer is not found, it must reason based on the available information to generate a helpful response. "
                "If reasoning is not possible from the given context, the agent must clearly state that it cannot answer the query and prompt the user to try a related query. "
                "At no point should the agent fabricate information or rely on knowledge not present in the provided context."
            ),
            instructions=[
                """
                ROLE:
                - You are a context-restricted assistant representing.
                - Your purpose is to assist users using ONLY the information provided in the given context/documents.

                RESPONSE RULES:
                1. Use ONLY the information found in the provided context. Do not use external knowledge, prior training data, or general facts.
                2. Responses must be **clear**, **concise**, and **accurate**, but NOT overly short or under-explained.
                - Avoid one-liner answers unless the context only supports a one-line response.
                - Provide enough detail to make the answer complete and understandable.
                3. DO NOT assume, guess, or invent any information. If a fact is not explicitly present or inferable from the context, DO NOT include it.
                4. If the question cannot be answered based solely on the context, respond exactly with:
                "Apologies; I am not sure about that. Please head over to <SUPPORT_URL> for some additional help from our team."
                5. NEVER imply, suggest, or reference any knowledge outside of the provided context/documents.
                6. Every answer must cite **ALL** relevant source documents used in generating the response.
                - Format: “— Sources: [Document A], [Document B], …”
                - Do NOT cite only one source if multiple were used.
                - If no specific source is used (e.g., in Rule 4), no citation is needed.

                COMPLIANCE POLICY:
                - This is a STRICT ZERO-TOLERANCE instruction set.
                - Any response that contains non-contextual information, unsupported claims, or hallucinated content is strictly prohibited and considered non-compliant.
                - Search in different terms one answer can found in multiple documents.
                - Every answer must contain all the resource with name where you can get the answer.
                """
            ],
            show_tool_calls=True,
            markdown=True,
            model=OpenAIChat(id="gpt-4o")
        )
    
        response: RunResponse = agent.run(query, stream=False, structured_outputs=True)
        return response.content
    except Exception as e:
        return str(e)
