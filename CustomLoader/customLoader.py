import json
from typing import List
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document

class MarkdownAndChunksLoader(BaseLoader):
    """Custom loader for JSON files with markdown + chunks structure."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Document]:
        # Read JSON file
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        docs = []

        # Step 1: Add Markdown content as one Document
        if "markdown" in data:
            docs.append(Document(
                page_content=data["markdown"],
                metadata={"type": "markdown"}
            ))

        # Step 2: Add each chunk as separate Document
        for chunk in data.get("chunks", []):
            text = chunk.get("text", "")
            grounding = chunk.get("grounding", [])
            chunk_type = chunk.get("chunk_type", "")
            chunk_id = chunk.get("chunk_id", "")

            # Extract page numbers (list, in case multiple grounding objects exist)
            pages = [g.get("page") for g in grounding if "page" in g]

            docs.append(Document(
                page_content=text,
                metadata={
                    "type": "chunk",
                    "chunk_type": chunk_type,
                    "chunk_id": chunk_id,
                    "grounding": grounding,
                    "pages": pages,
                    "source": self.file_path
                }
            ))

        return docs
