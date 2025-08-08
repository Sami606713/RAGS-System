from langchain_core.prompts import ChatPromptTemplate


def generator_prompt() -> ChatPromptTemplate:
    """
    Generates a prompt template for the answer generation step in the workflow.
    """
    return ChatPromptTemplate.from_template(
        """
        You are a helpful and detail-oriented technical assistant. Based only on the information provided in the context below, generate a clear, structured, and accurate response to the user's question.

        ### Instructions:
        - Use formal, factual language.
        - Organize the answer using appropriate headings (e.g., Introduction, Key Points, Recommendations).
        - Use full sentences and complete paragraphs.
        - Do **not** assume or invent facts not present in the context.
        - If the context does not contain enough information, reply with: "I'm not sure based on the given context."
        - At the end, include a **References** section listing the source identifiers.

        ---

        ### Context:
        {combined_context}

        ### Question:
        {query}
        ---

        ### Answer:
        """
    )