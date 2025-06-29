from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain.schema import Document
from dotenv import load_dotenv
load_dotenv()


def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64


def LoadAndExtractData(file_path):
    try:
        tables = []
        texts = []
        print(">> Extracting Data")
        data = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        print(">> Extracting Text and tables...")
        for chunk in data:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type((chunk))):
                texts.append(chunk)
        print(">> Chunks are: ", data)
        print(">> Extracting Images...")
        images = get_images_base64(data)
        return tables, texts, images
    except Exception as e:
        print("Error is: ", str(e))
        return [], [], str(e)
    


# Summarizer Function
def Summarizer(prompt_template, data, config=True, set_messages=False):
    """
    This function summarizes documents using a prompt template and the ChatOpenAI model.
    
    Args:
        prompt_template (str): Template string for the prompt.
        data (List[Dict] or List[str]): Input data to be summarized.
        config (bool): Whether to run the chain with concurrency limit.
        set_messages (bool): Whether to set messages as chat messages with an image.

    Returns:
        List[str]: List of summaries.
    """
    try:
        if set_messages:
            messages = [
                (
                    "user",
                    [
                        {"type": "text", "text": prompt_template},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                    ],
                )
            ]
            prompt = ChatPromptTemplate.from_messages(messages)
            model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
            summarize_chain = {"image": lambda x: x} | prompt | model | StrOutputParser()
        else:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
            summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
        

        if config:
            return summarize_chain.batch(data, {"max_concurrency": 3})
        else:
            return summarize_chain.batch(data)
    except Exception as e:
        return str(e)


def Query_Optimizer(query):
    """
    This function optimizes a query by removing unnecessary words and phrases.

    Args:
        query (str): The input query to be optimized.

    Returns:
        str: The optimized query.
    """
    try:
        prompt_template = """
        You are a query expansion expert working on improving search relevance.

        Your task is to rewrite the following user query to make it more informative and clear, without changing its intent. You can:
        - Add missing context (e.g., expand abbreviations, clarify vague terms).
        - Add synonyms or related keywords to improve search accuracy.
        - Make it more specific when possible.

        Rules:
        1. Do not change the user's intent.
        2. Do not add unrelated information.
        3. Return only the improved query—no explanation or formatting.

        Original Query:
        {query}

        Expanded Query:
        """

        prompt = ChatPromptTemplate.from_template(prompt_template)
        model = ChatOpenAI(temperature=0.5, model="gpt-4o-mini")
        optimize_chain = prompt | model | StrOutputParser()
        return optimize_chain.invoke({"query": query})
    except Exception as e:
        return str(e)
    

def get_bm25_retriever(docs: List[Document], k: int = 10):
    if not docs:
        raise ValueError("❌ get_bm25_retriever received an empty `docs` list. Ensure documents are loaded before calling.")
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever


if __name__ == "__main__":
    query = "What are the main benefits of using wind propulsion technologies in maritime transport?"
    optimized_query = Query_Optimizer(query)
    print(f"Optimized Query: {optimized_query}")