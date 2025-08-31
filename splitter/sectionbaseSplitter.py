from langchain.text_splitter import TextSplitter
from typing import List, Dict
from langchain_core.documents import Document
import re

class section_base_splitter(TextSplitter):
    """
    Custom LangChain TextSplitter for Landing AI Markdown-style documents.
    Splits by headings (# ...) or capitalized labels ending with ':', preserves everything,
    and returns metadata including section titles.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def split_text(self, text: str) -> List[str]:
        """
        Split the document into chunks. Each chunk is text with section header included.
        """
        chunks = []

        # Capture everything before the first heading/label as 'Pre-section'
        pre_heading_match = re.match(r"^(.*?)(?=\n# |\n[A-Z][a-z]+ :|\Z)", text, flags=re.S)
        if pre_heading_match:
            intro_text = pre_heading_match.group(1).strip()
            if intro_text:
                chunks.append(intro_text)

        # Split by headings (# ...) or capitalized labels ending with ':'
        pattern = r"(^# .+?$|^[A-Z][A-Za-z0-9 &-]+ :)"
        matches = list(re.finditer(pattern, text, flags=re.M))

        for i, match in enumerate(matches):
            start_idx = match.start()  # include the heading/label itself
            if i + 1 < len(matches):
                end_idx = matches[i + 1].start()
            else:
                end_idx = len(text)

            chunk_text = text[start_idx:end_idx].strip()
            chunks.append(chunk_text)

        return chunks

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Accepts a list of dicts with 'page_content' and optional 'metadata',
        returns list of dicts with 'page_content' and 'metadata' including section_title.
        """
        output_docs = []

        for doc in documents:
            text = doc.page_content
            metadata = doc.metadata or {}

            # Split text into chunks
            section_chunks = self.split_text(text)

            for chunk in section_chunks:
                # Extract section title from chunk (first line if it's a heading/label)
                first_line = chunk.splitlines()[0].strip()
                if first_line.startswith("#"):
                    section_title = first_line.lstrip("#").strip()
                elif first_line.endswith(":"):
                    section_title = first_line.strip()
                else:
                    section_title = "Pre-section"

                # Add chunk with metadata
                chunk_metadata = metadata.copy()
                chunk_metadata["section_title"] = section_title

                output_docs.append(Document(page_content=chunk, metadata=chunk_metadata))

        return output_docs

if __name__ == "__main__":
    # ----------------------
    # Example Usage
    # ----------------------
    md_file_path = "doc2/Clean Energy Market Analysis in the US.extraction.md"

    with open(md_file_path, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Wrap into document format LangChain expects
    documents = [{"page_content": doc_text, "metadata": {"source": "landing_ai_file_1.md"}}]

    splitter = section_base_splitter()
    chunked_docs = splitter.split_documents(documents)

    for idx, doc in enumerate(chunked_docs, 1):
        print(f"--- Chunk {idx} ---")
        print("Section Title:", doc["metadata"]["section_title"])
        print(doc["page_content"])
        print("Source:", doc["metadata"]["source"])
        print()
