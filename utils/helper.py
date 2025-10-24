# from unstructured.partition.pdf import partition_pdf
import os
import pypdf
import json
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from typing import List
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
        images = []
        print(">> Extracting Data")

        # Simple PDF text extraction using pypdf
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    texts.append(f"Page {page_num + 1}: {text.strip()}")

        print(f">> Extracted {len(texts)} text pages")
        print(">> Note: Table and image extraction temporarily disabled due to dependency issues")

        return tables, texts, images
    except Exception as e:
        print("Error is: ", str(e))
        return [], [], str(e)


def LoadAndExtractDataFromJSON(file_path):
    """
    Enhanced extraction of article content from JSON files with improved chunking and metadata.

    Optimized for article processing with better contextual preservation and semantic structure.

    Args:
        file_path: Path to the JSON file

    Returns:
        tuple: (tables, texts, images) where each contains structured content with metadata
    """
    try:
        tables = []
        texts = []
        images = []
        print(">> Extracting and processing article data from JSON")

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        file_name = os.path.basename(file_path)

        # Enhanced chunk processing with metadata preservation
        if 'chunks' in data:
            for i, chunk in enumerate(data['chunks']):
                chunk_text = chunk.get('text', '').strip()
                chunk_type = chunk.get('chunk_type', 'text')
                chunk_id = chunk.get('chunk_id', f"chunk_{i}")

                if not chunk_text:  # Skip empty chunks
                    continue

                # Create enhanced content structure
                enhanced_content = {
                    'content': chunk_text,
                    'chunk_id': chunk_id,
                    'chunk_type': chunk_type,
                    'source_file': file_name,
                    'metadata': chunk.get('grounding', [])  # Preserve positioning info
                }

                # Categorize content with better logic
                if chunk_type in ['table', 'Table']:
                    tables.append(enhanced_content)
                elif chunk_type in ['figure', 'image', 'Figure', 'Image']:
                    images.append(enhanced_content)
                else:
                    # For text chunks, add semantic context
                    enhanced_content['semantic_type'] = _classify_text_type(chunk_text)
                    texts.append(enhanced_content)

        # Enhanced markdown processing with section detection
        elif 'markdown' in data:
            markdown_content = data['markdown']
            sections = _split_markdown_into_sections(markdown_content, file_name)
            texts.extend(sections)

        print(f">> Successfully processed: {len(texts)} text sections, {len(tables)} tables, {len(images)} images")
        return tables, texts, images

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []


def _classify_text_type(text):
    """Classify text content for better semantic understanding"""
    text_lower = text.lower()

    if any(keyword in text_lower for keyword in ['abstract', 'summary', 'overview']):
        return 'abstract'
    elif any(keyword in text_lower for keyword in ['introduction', 'background']):
        return 'introduction'
    elif any(keyword in text_lower for keyword in ['conclusion', 'findings', 'results']):
        return 'conclusion'
    elif any(keyword in text_lower for keyword in ['method', 'approach', 'technique']):
        return 'methodology'
    elif len(text) < 200:
        return 'heading_or_caption'
    else:
        return 'content'


def _split_markdown_into_sections(markdown_content, source_file):
    """Split markdown content into logical sections for better retrieval"""
    import re

    sections = []

    # Split by headers (# ## ###)
    header_pattern = r'^(#{1,6})\s+(.+?)$'
    lines = markdown_content.split('\n')

    current_section = []
    current_header = None
    current_level = 0

    for line in lines:
        header_match = re.match(header_pattern, line)

        if header_match:
            # Save previous section if it has content
            if current_section and any(l.strip() for l in current_section):
                section_content = '\n'.join(current_section).strip()
                if section_content:
                    sections.append({
                        'content': section_content,
                        'chunk_id': f"section_{len(sections)}",
                        'chunk_type': 'text',
                        'source_file': source_file,
                        'header': current_header,
                        'header_level': current_level,
                        'semantic_type': _classify_text_type(section_content),
                        'metadata': []
                    })

            # Start new section
            current_header = header_match.group(2).strip()
            current_level = len(header_match.group(1))
            current_section = [line]
        else:
            current_section.append(line)

    # Add final section
    if current_section and any(l.strip() for l in current_section):
        section_content = '\n'.join(current_section).strip()
        if section_content:
            sections.append({
                'content': section_content,
                'chunk_id': f"section_{len(sections)}",
                'chunk_type': 'text',
                'source_file': source_file,
                'header': current_header,
                'header_level': current_level,
                'semantic_type': _classify_text_type(section_content),
                'metadata': []
            })

    return sections



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



def summarize_docs(docs, model_name="gpt-4o-mini"):
    """
    Summarize a list of Document objects section-wise.
    
    Args:
        docs: List of langchain Document objects (e.g., from MarkdownHeaderTextSplitter)
        model_name: LLM to use for summarization (default gpt-4o-mini)
    
    Returns:
        summaries: List of dicts with {metadata, summary}
    """
    # Define a simple summarization prompt
    prompt_template = """
    You are an expert summarizer.

    Task:
    - If the input is a **section** → produce a concise summary (3–5 bullet points).
    - If the input is a **full document** → produce a structured summary with short paragraphs (not exceeding 200 words total).

    Guidelines:
    - Focus only on the main ideas.
    - Remove redundancy.
    - Do not copy sentences directly; paraphrase concisely.
    - Output should be easy to skim.

    Content:
    {text}

    Summary:
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    summaries = []
    for doc in docs:
        summary = chain.run(text=doc.page_content)
        summaries.append({
            "metadata": doc.metadata,
            "summary": summary.strip()
        })
    
    return summaries


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