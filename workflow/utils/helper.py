from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
import re
from collections import defaultdict
from typing import List, Dict, Any
from langchain_core.documents import Document

from vectorStore.vectorStore import get_embeddings
from utils.sourceValidator import get_source_validator
def load_vector_store(vector_store_path: str = "final_index2"):
    """
    Load the FAISS vector store from the specified path with source validation.
    """
    try:
        embeddings = get_embeddings()
        print(f">> Loading vector store from {vector_store_path}...")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print(">> Vector store loaded successfully!")
        return TrustedSourceVectorStore(vector_store)
    except Exception as e:
        raise Exception(f"Error loading vector store: {str(e)}")


class TrustedSourceVectorStore:
    """Wrapper for FAISS vector store that filters results by trusted sources."""

    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.source_validator = get_source_validator()

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> List[Document]:
        """Perform similarity search and filter results by trusted sources."""
        try:
            # Get results from underlying vector store
            raw_results = self.vector_store.similarity_search(query, k=k*2, **kwargs)  # Get more to account for filtering

            # Filter by trusted sources
            trusted_results = []
            rejected_count = 0

            for doc in raw_results:
                if self.source_validator.validate_document_metadata(doc.metadata):
                    trusted_results.append(doc)
                else:
                    rejected_count += 1
                    source = doc.metadata.get('source', 'Unknown')
                    print(f">> Filtered out untrusted source in retrieval: {source}")

                # Stop when we have enough trusted results
                if len(trusted_results) >= k:
                    break

            if rejected_count > 0:
                print(f"WARNING: Filtered out {rejected_count} untrusted documents from search results")

            return trusted_results[:k]

        except Exception as e:
            print(f"Error in trusted similarity search: {e}")
            return []

    def as_retriever(self, **kwargs):
        """Return retriever interface with source filtering."""
        return TrustedSourceRetriever(self, **kwargs)

    def __getattr__(self, name):
        """Delegate other attributes to the underlying vector store."""
        return getattr(self.vector_store, name)


class TrustedSourceRetriever(BaseRetriever):
    """Retriever that filters results by trusted sources."""

    trusted_vector_store: Any = Field(description="The trusted vector store")
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Search parameters")

    def __init__(self, trusted_vector_store, search_kwargs=None, **kwargs):
        super().__init__(
            trusted_vector_store=trusted_vector_store,
            search_kwargs=search_kwargs or {},
            **kwargs
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """Retrieve documents for a query, filtered by trusted sources."""
        k = self.search_kwargs.get('k', 4)
        return self.trusted_vector_store.similarity_search(query, k=k)

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Alias for _get_relevant_documents method."""
        return self._get_relevant_documents(query)
    


def format_context(raw_context):
    page_chunks = defaultdict(list)   # group chunks by (source, page)
    references = set()
    seen_chunk_ids = set()

    for doc in raw_context:
        meta = doc.metadata

        if meta['type'] == 'chunk':
            chunk_id = meta.get("chunk_id")
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            pages = meta.get('pages', [])
            source = meta.get('source', 'Unknown')
            page_key = (source, tuple(pages))  # use tuple since list isn't hashable

            content = re.sub(r'\s+', ' ', doc.page_content.strip())
            chunk_summary = meta.get('chunk_summary', '')
            global_summary = meta.get('global_summary', '')

            page_chunks[page_key].append({
                "content": content,
                "chunk_summary": chunk_summary,
                "global_summary": global_summary
            })

            references.add(f"{source}, pp {pages if pages else 'N/A'}")

    # --- Format merged output ---
    text_content = []
    for (source, pages), chunks in page_chunks.items():
        page_str = pages if pages else 'N/A'
        merged_content = "\n".join(
            f"- {c['content']}\n  • Chunk Summary: {c['chunk_summary']}\n  • Global Summary: {c['global_summary']}"
            for c in chunks
        )
        formatted = f"""
**Source: {source}, Page {page_str}**
{merged_content}
"""
        text_content.append(formatted)

    final_output = f"""
================= 📑 CHUNKS =================
{''.join(text_content)}

================= 📚 REFERENCES =================
{chr(10).join(sorted(references))}
"""
    return final_output