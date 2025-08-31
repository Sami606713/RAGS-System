from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import TFIDFRetriever
import re
from collections import defaultdict

from vectorStore.vectorStore import get_embeddings
def load_vector_store(vector_store_path: str = "final_index2"):
    """
    Load the FAISS vector store from the specified path.
    """
    try:
        embeddings = get_embeddings()
        print(f">> Loading vector store from {vector_store_path}...")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        print("✅ Vector store loaded successfully!")
        return vector_store
    except Exception as e:
        raise Exception(f"Error loading vector store: {str(e)}")
    


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