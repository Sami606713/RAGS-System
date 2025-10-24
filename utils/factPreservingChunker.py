"""
Fact-Preserving Chunking Strategy

Addresses critical issue #7: Key facts split across chunks so reader misses unit or qualifier

Advanced chunking that:
- Preserves numeric facts with their units and qualifiers
- Maintains context for important statements
- Respects sentence and paragraph boundaries
- Optimizes for retrieval accuracy
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


@dataclass
class FactUnit:
    """A complete fact with its context"""
    numeric_value: str
    unit: str
    qualifier: str  # e.g., "approximately", "up to", "at least"
    full_context: str  # Complete sentence containing the fact
    start_pos: int
    end_pos: int


class FactPreservingChunker:
    """Advanced chunker that preserves important facts and their context"""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 300):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Patterns for identifying important facts
        self.fact_patterns = [
            # Currency with qualifiers: "approximately $123.45 per kg"
            r'((?:approximately|about|around|up to|at least|more than|less than|over|under)?\s*\$?[\d,]+\.?\d*\s*(?:USD|dollars?|cents?)?\s*(?:per|\/)\s*(?:kg|ton|lb|liter|gallon|kwh|unit))',

            # Technical specifications: "efficiency of 85.5%"
            r'((?:efficiency|performance|capacity|output|yield|rate)\s+(?:of|is|reaches|achieves)?\s*[\d,]+\.?\d*\s*%)',

            # Measurements with context: "temperature of 150°C to 200°C"
            r'((?:temperature|pressure|speed|weight|volume|length|height|width|depth|diameter)\s+(?:of|is|reaches|between|from)?\s*[\d,]+\.?\d*(?:°[CF]|psi|mph|kg|m|cm|mm|L|gal)?(?:\s*(?:to|-)\s*[\d,]+\.?\d*(?:°[CF]|psi|mph|kg|m|cm|mm|L|gal)?)?)',

            # Production/capacity figures: "produces 500 tons annually"
            r'((?:produces?|generates?|outputs?|capacity of|yields?)\s*[\d,]+\.?\d*\s*(?:tons?|kg|MW|kW|liters?|gallons?|units?)\s*(?:per|annually|yearly|daily|monthly)?)',

            # Time-based measurements: "takes 30 minutes to complete"
            r'((?:takes|requires|lasts|duration of)\s*[\d,]+\.?\d*\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?))',
        ]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents while preserving important facts"""

        chunked_documents = []

        for doc in documents:
            chunks = self._split_document_preserving_facts(doc)
            chunked_documents.extend(chunks)

        return chunked_documents

    def _split_document_preserving_facts(self, document: Document) -> List[Document]:
        """Split a single document while preserving facts"""

        content = document.page_content
        metadata = document.metadata.copy()

        # Step 1: Identify all important facts in the document
        facts = self._identify_facts(content)

        # Step 2: Create sentence boundaries
        sentences = self._get_sentence_boundaries(content)

        # Step 3: Create chunks that preserve facts
        chunks = self._create_fact_preserving_chunks(content, facts, sentences)

        # Step 4: Convert to Document objects
        chunk_documents = []
        for i, chunk_content in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_facts': self._count_facts_in_chunk(chunk_content, facts),
                'fact_preserved': True
            })

            chunk_doc = Document(
                page_content=chunk_content,
                metadata=chunk_metadata
            )
            chunk_documents.append(chunk_doc)

        return chunk_documents

    def _identify_facts(self, content: str) -> List[FactUnit]:
        """Identify important facts that must be preserved"""

        facts = []

        for pattern in self.fact_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                fact_text = match.group(1)

                # Extract the complete sentence containing this fact
                fact_sentence = self._extract_complete_sentence(content, match.start(), match.end())

                # Parse the fact components
                numeric_part = re.search(r'[\d,]+\.?\d*', fact_text)
                unit_part = re.search(r'(?:kg|ton|lb|liter|gallon|kwh|unit|%|°[CF]|psi|mph|m|cm|mm|L|gal|USD|dollars?|cents?)', fact_text, re.IGNORECASE)
                qualifier_part = re.search(r'(approximately|about|around|up to|at least|more than|less than|over|under)', fact_text, re.IGNORECASE)

                fact = FactUnit(
                    numeric_value=numeric_part.group(0) if numeric_part else "",
                    unit=unit_part.group(0) if unit_part else "",
                    qualifier=qualifier_part.group(0) if qualifier_part else "",
                    full_context=fact_sentence,
                    start_pos=self._find_sentence_start(content, match.start()),
                    end_pos=self._find_sentence_end(content, match.end())
                )

                facts.append(fact)

        return facts

    def _extract_complete_sentence(self, content: str, fact_start: int, fact_end: int) -> str:
        """Extract the complete sentence containing a fact"""

        # Find sentence boundaries around the fact
        sentence_start = self._find_sentence_start(content, fact_start)
        sentence_end = self._find_sentence_end(content, fact_end)

        return content[sentence_start:sentence_end].strip()

    def _find_sentence_start(self, content: str, position: int) -> int:
        """Find the start of the sentence containing the given position"""

        # Look backwards for sentence boundaries
        for i in range(position - 1, max(0, position - 200), -1):
            if content[i] in '.!?':
                # Check if this is actually a sentence end (not abbreviation)
                if i + 1 < len(content) and content[i + 1].isspace():
                    return i + 1
            elif content[i] in '\n\r' and i > 0:
                # Paragraph break
                return i + 1

        # If no sentence boundary found, use paragraph start
        for i in range(position - 1, max(0, position - 500), -1):
            if content[i] in '\n\r':
                return i + 1

        return 0

    def _find_sentence_end(self, content: str, position: int) -> int:
        """Find the end of the sentence containing the given position"""

        # Look forwards for sentence boundaries
        for i in range(position, min(len(content), position + 200)):
            if content[i] in '.!?':
                # Check if this is actually a sentence end
                if i + 1 < len(content) and (content[i + 1].isspace() or content[i + 1].isupper()):
                    return i + 1
            elif content[i] in '\n\r':
                # Paragraph break
                return i

        # If no sentence boundary found, use paragraph end
        for i in range(position, min(len(content), position + 500)):
            if content[i] in '\n\r':
                return i

        return min(len(content), position + 200)

    def _get_sentence_boundaries(self, content: str) -> List[Tuple[int, int]]:
        """Get all sentence boundaries in the content"""

        sentences = []
        sentence_pattern = r'[.!?]+\s+'

        start = 0
        for match in re.finditer(sentence_pattern, content):
            end = match.end()
            if end - start > 10:  # Minimum sentence length
                sentences.append((start, end))
            start = end

        # Add final sentence if exists
        if start < len(content) - 10:
            sentences.append((start, len(content)))

        return sentences

    def _create_fact_preserving_chunks(self, content: str, facts: List[FactUnit], sentences: List[Tuple[int, int]]) -> List[str]:
        """Create chunks that preserve important facts"""

        chunks = []
        current_chunk = ""
        current_chunk_start = 0

        # Sort facts by position
        facts_by_position = sorted(facts, key=lambda f: f.start_pos)

        i = 0
        while i < len(sentences):
            sentence_start, sentence_end = sentences[i]
            sentence_text = content[sentence_start:sentence_end]

            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence_text) > self.chunk_size and len(current_chunk) > self.min_chunk_size:

                # Before finalizing chunk, check if any facts would be split
                chunk_end = current_chunk_start + len(current_chunk)
                facts_in_range = [f for f in facts_by_position if f.start_pos < chunk_end and f.end_pos > chunk_end]

                if facts_in_range:
                    # Extend chunk to include complete fact context
                    max_fact_end = max(f.end_pos for f in facts_in_range)
                    additional_content = content[chunk_end:max_fact_end]
                    current_chunk += additional_content

                # Finalize current chunk
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_start = max(0, len(current_chunk) - self.chunk_overlap)
                current_chunk = current_chunk[overlap_start:] + sentence_text
                current_chunk_start = sentence_start - (len(current_chunk) - len(sentence_text))

            else:
                current_chunk += sentence_text

            i += 1

        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _count_facts_in_chunk(self, chunk_content: str, all_facts: List[FactUnit]) -> int:
        """Count how many important facts are preserved in this chunk"""

        count = 0
        for fact in all_facts:
            if fact.full_context in chunk_content:
                count += 1

        return count

    def validate_chunking_quality(self, original_documents: List[Document], chunked_documents: List[Document]) -> Dict:
        """Validate that chunking preserved important information"""

        validation_results = {
            'original_docs': len(original_documents),
            'final_chunks': len(chunked_documents),
            'facts_preserved': 0,
            'facts_lost': 0,
            'average_chunk_size': 0,
            'chunks_with_facts': 0,
            'quality_score': 0.0
        }

        # Count original facts
        original_facts = 0
        for doc in original_documents:
            facts = self._identify_facts(doc.page_content)
            original_facts += len(facts)

        # Count preserved facts
        preserved_facts = 0
        chunks_with_facts = 0
        total_chunk_size = 0

        for chunk_doc in chunked_documents:
            chunk_facts = self._identify_facts(chunk_doc.page_content)
            preserved_facts += len(chunk_facts)
            total_chunk_size += len(chunk_doc.page_content)

            if len(chunk_facts) > 0:
                chunks_with_facts += 1

        validation_results.update({
            'facts_preserved': preserved_facts,
            'facts_lost': max(0, original_facts - preserved_facts),
            'average_chunk_size': total_chunk_size // len(chunked_documents) if chunked_documents else 0,
            'chunks_with_facts': chunks_with_facts,
            'quality_score': preserved_facts / original_facts if original_facts > 0 else 1.0
        })

        return validation_results


def create_fact_preserving_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> FactPreservingChunker:
    """Factory function to create fact-preserving chunker"""
    return FactPreservingChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)