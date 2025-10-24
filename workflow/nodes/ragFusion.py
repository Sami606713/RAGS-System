from workflow.states.states import AgentState
from workflow.utils.helper import load_vector_store
from workflow.models.loadModel import load_model
from typing import List
from langchain_core.documents import Document
import numpy as np
from collections import defaultdict


def generate_query_variations(query: str, num_variations: int = 5) -> List[str]:
    """
    Generate multiple paraphrased versions of the query using LLM.
    """
    llm = load_model()

    prompt = f"""Generate {num_variations} different paraphrased versions of the following query.
Each version should ask the same thing but with different wording.
Return ONLY the paraphrased queries, one per line, without numbering or explanations.

Original query: {query}

Paraphrased queries:"""

    response = llm.invoke(prompt)
    variations = [line.strip() for line in response.content.strip().split('\n') if line.strip()]

    # Include the original query
    all_queries = [query] + variations[:num_variations-1]
    return all_queries


def reciprocal_rank_fusion(doc_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Apply Reciprocal Rank Fusion (RRF) to merge and re-rank documents.

    RRF formula: RRF(d) = sum over all lists of 1/(k + rank(d))
    where k is a constant (typically 60) and rank(d) is the rank of document d in a list.
    """
    # Track scores for each unique document
    doc_scores = defaultdict(float)
    doc_map = {}  # Map from doc content hash to actual document

    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list):
            # Use page_content as unique identifier
            doc_id = hash(doc.page_content)
            doc_map[doc_id] = doc

            # Add RRF score
            doc_scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by score (descending)
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Return documents in order of their fused score
    return [doc_map[doc_id] for doc_id, score in sorted_docs]


def rag_fusion(state: AgentState):
    """
    RAG Fusion: Generate multiple query variations, retrieve for each,
    then merge and re-rank using Reciprocal Rank Fusion.

    This replaces simple query expansion with a more robust approach.
    """
    try:
        print("RAG Fusion: Generating query variations...")

        # Get the refined question
        question = state['rewrite_question']

        # Generate 5 query variations
        query_variations = generate_query_variations(question, num_variations=5)
        print(f"RAG Fusion: Generated {len(query_variations)} query variations")

        # Retrieve documents for each query variation
        vector_store = load_vector_store()
        all_doc_lists = []

        for idx, query in enumerate(query_variations):
            print(f"RAG Fusion: Retrieving for variation {idx + 1}: {query[:50]}...")
            docs = vector_store.similarity_search(query, k=5)
            all_doc_lists.append(docs)

        # Apply Reciprocal Rank Fusion
        print("RAG Fusion: Applying Reciprocal Rank Fusion...")
        fused_docs = reciprocal_rank_fusion(all_doc_lists, k=60)

        # Take top K documents after fusion
        top_k = 10
        final_docs = fused_docs[:top_k]

        print(f"RAG Fusion: Returning {len(final_docs)} re-ranked documents")

        return {
            "context": final_docs
        }

    except Exception as e:
        print(f"RAG Fusion Error: {str(e)}")
        return {'error': str(e)}
