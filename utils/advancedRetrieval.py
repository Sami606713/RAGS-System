"""
Advanced Retrieval and Answer Generation System

Addresses critical issues:
- Hybrid retrieval with reranking
- Source prioritization and conflict resolution
- Answer template constraints and drift prevention
- Comprehensive citation system
- Explicit not-found handling
- Chunking optimization for fact preservation
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, Counter

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank

from utils.contentAnalyzer import ContentAnalyzer, IntentCategory, ContentQuality


class AnswerTemplate(Enum):
    """Structured answer templates to prevent drift"""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    NUMERICAL = "numerical"
    NOT_FOUND = "not_found"
    INSUFFICIENT = "insufficient_evidence"


@dataclass
class RetrievalResult:
    """Enhanced retrieval result with confidence and provenance"""
    document: Document
    relevance_score: float
    content_analysis: Any  # ContentAnalysis
    authority_score: float
    conflicts: List[str]
    citations: List[Dict]


@dataclass
class AnswerEvidence:
    """Structured evidence for answer generation"""
    primary_facts: List[Dict]
    supporting_evidence: List[Dict]
    conflicting_evidence: List[Dict]
    confidence_score: float
    citations: List[Dict]
    gaps_identified: List[str]


class AdvancedRetriever:
    """Hybrid retrieval system with reranking and quality control"""

    def __init__(self, vector_store, source_validator, content_analyzer: ContentAnalyzer):
        self.vector_store = vector_store
        self.source_validator = source_validator
        self.content_analyzer = content_analyzer
        self.reranker = None  # Initialize if Cohere API available

        # Authority scoring weights
        self.authority_weights = {
            'semantic_type': {
                'abstract': 0.9,
                'methodology': 0.8,
                'results': 0.8,
                'conclusion': 0.7,
                'introduction': 0.6,
                'content': 0.5
            },
            'content_quality': {
                ContentQuality.HIGH: 1.0,
                ContentQuality.MEDIUM: 0.7,
                ContentQuality.LOW: 0.3,
                ContentQuality.CORRUPTED: 0.0
            }
        }

    def hybrid_retrieve(self, query: str, intent: IntentCategory, k: int = 10) -> List[RetrievalResult]:
        """Advanced hybrid retrieval with intent-aware filtering"""

        # Step 1: Vector similarity search (broader recall)
        vector_results = self.vector_store.similarity_search(query, k=k*2)

        # Step 2: Create BM25 retriever for lexical matching
        if vector_results:
            bm25_retriever = BM25Retriever.from_documents(vector_results)
            bm25_retriever.k = k
            bm25_results = bm25_retriever.get_relevant_documents(query)
        else:
            bm25_results = []

        # Step 3: Combine and deduplicate results
        combined_results = self._combine_retrieval_results(vector_results, bm25_results)

        # Step 4: Intent-aware filtering
        filtered_results = self._filter_by_intent(combined_results, intent, query)

        # Step 5: Content analysis and quality scoring
        analyzed_results = []
        for doc in filtered_results:
            # Validate source trust
            if not self.source_validator.validate_document_metadata(doc.metadata):
                continue

            # Analyze content quality
            content_analysis = self.content_analyzer.analyze_content(
                doc.page_content,
                doc.metadata
            )

            # Skip corrupted content
            if content_analysis.quality_score == ContentQuality.CORRUPTED:
                continue

            # Calculate authority score
            authority_score = self._calculate_authority_score(doc, content_analysis)

            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(query, doc, content_analysis)

            # Identify conflicts with other results
            conflicts = self._identify_conflicts(doc, content_analysis, analyzed_results)

            # Extract citations
            citations = self._extract_citations(doc, content_analysis)

            result = RetrievalResult(
                document=doc,
                relevance_score=relevance_score,
                content_analysis=content_analysis,
                authority_score=authority_score,
                conflicts=conflicts,
                citations=citations
            )
            analyzed_results.append(result)

        # Step 6: Rerank results (if reranker available)
        if self.reranker and len(analyzed_results) > 1:
            analyzed_results = self._rerank_results(query, analyzed_results)

        # Step 7: Sort by combined score and return top k
        final_results = sorted(
            analyzed_results,
            key=lambda x: (x.authority_score * 0.4 + x.relevance_score * 0.6),
            reverse=True
        )

        return final_results[:k]

    def _combine_retrieval_results(self, vector_results: List[Document], bm25_results: List[Document]) -> List[Document]:
        """Combine and deduplicate retrieval results"""
        seen_content = set()
        combined = []

        # Add vector results first (semantic similarity priority)
        for doc in vector_results:
            content_hash = hash(doc.page_content[:200])  # First 200 chars for dedup
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)

        # Add unique BM25 results
        for doc in bm25_results:
            content_hash = hash(doc.page_content[:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                combined.append(doc)

        return combined

    def _filter_by_intent(self, documents: List[Document], intent: IntentCategory, query: str) -> List[Document]:
        """Filter documents based on query intent to prevent category errors"""

        if intent == IntentCategory.GENERAL:
            return documents

        # Intent-specific filtering
        filtered = []
        for doc in documents:
            content_lower = doc.page_content.lower()
            semantic_type = doc.metadata.get('semantic_type', 'content')

            # Price intent filtering (prevent production/consumer confusion)
            if intent == IntentCategory.PRICE_CONSUMER:
                if any(word in content_lower for word in ['retail', 'consumer', 'market price', 'selling price']):
                    filtered.append(doc)
                elif 'price' in content_lower and 'production' not in content_lower:
                    filtered.append(doc)

            elif intent == IntentCategory.PRICE_PRODUCTION:
                if any(word in content_lower for word in ['production cost', 'manufacturing', 'cost to produce']):
                    filtered.append(doc)

            elif intent == IntentCategory.TECHNICAL_SPEC:
                if any(word in content_lower for word in ['specification', 'technical', 'parameter']):
                    filtered.append(doc)
                elif semantic_type in ['methodology', 'results']:
                    filtered.append(doc)

            else:
                filtered.append(doc)

        return filtered if filtered else documents  # Fallback to all if none match

    def _calculate_authority_score(self, doc: Document, analysis: Any) -> float:
        """Calculate document authority score"""

        score = 0.5  # Base score

        # Semantic type weight
        semantic_type = doc.metadata.get('semantic_type', 'content')
        score += self.authority_weights['semantic_type'].get(semantic_type, 0.5) * 0.3

        # Content quality weight
        score += self.authority_weights['content_quality'].get(analysis.quality_score, 0.5) * 0.3

        # Confidence score weight
        score += analysis.confidence_score * 0.2

        # Numeric facts presence (authoritative data)
        if analysis.numeric_facts:
            avg_numeric_confidence = sum(fact.confidence for fact in analysis.numeric_facts) / len(analysis.numeric_facts)
            score += avg_numeric_confidence * 0.2

        return min(1.0, score)

    def _calculate_relevance_score(self, query: str, doc: Document, analysis: Any) -> float:
        """Calculate query-document relevance score"""

        base_score = 0.5
        query_lower = query.lower()
        content_lower = doc.page_content.lower()

        # Keyword overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words.intersection(content_words)) / len(query_words)
        base_score += overlap * 0.3

        # Intent matching bonus
        if analysis.intent_category != IntentCategory.GENERAL:
            base_score += 0.2

        # Header relevance (if content has structured headers)
        header = doc.metadata.get('header', '').lower()
        if header and any(word in header for word in query_words):
            base_score += 0.2

        return min(1.0, base_score)

    def _identify_conflicts(self, current_doc: Document, current_analysis: Any, existing_results: List[RetrievalResult]) -> List[str]:
        """Identify conflicts with existing retrieval results"""

        conflicts = []

        if not current_analysis.numeric_facts:
            return conflicts

        for existing in existing_results:
            if not existing.content_analysis.numeric_facts:
                continue

            # Check for conflicting numeric values with same units
            for current_fact in current_analysis.numeric_facts:
                for existing_fact in existing.content_analysis.numeric_facts:
                    if (current_fact.normalized_unit == existing_fact.normalized_unit and
                        abs(current_fact.value - existing_fact.value) / max(current_fact.value, existing_fact.value) > 0.2):

                        conflicts.append(f"Conflicting {current_fact.normalized_unit} values: {current_fact.value} vs {existing_fact.value}")

        return conflicts

    def _extract_citations(self, doc: Document, analysis: Any) -> List[Dict]:
        """Extract detailed citation information"""

        citations = []

        # Per-sentence citations from provenance spans
        for span in analysis.provenance_spans:
            citation = {
                'text': span['text'],
                'source_file': span['source_file'],
                'chunk_id': span['chunk_id'],
                'sentence_id': span['sentence_id'],
                'header': span.get('header', ''),
                'semantic_type': span.get('semantic_type', 'content'),
                'start_pos': span['start_pos'],
                'end_pos': span['end_pos']
            }
            citations.append(citation)

        return citations

    def _rerank_results(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank results using cross-encoder (if available)"""

        try:
            if self.reranker:
                documents = [result.document for result in results]
                reranked_docs = self.reranker.compress_documents(documents, query)

                # Map reranked docs back to results
                reranked_results = []
                for reranked_doc in reranked_docs:
                    for result in results:
                        if result.document.page_content == reranked_doc.page_content:
                            reranked_results.append(result)
                            break

                return reranked_results

        except Exception as e:
            print(f"Reranking failed: {e}")

        return results


class AnswerGenerator:
    """Structured answer generation with drift prevention and citation"""

    def __init__(self, llm):
        self.llm = llm
        self.content_analyzer = ContentAnalyzer()

    def generate_answer(self, query: str, retrieval_results: List[RetrievalResult], intent: IntentCategory) -> Dict:
        """Generate structured answer with comprehensive provenance"""

        # Step 1: Assess evidence quality
        evidence = self._assess_evidence(retrieval_results, intent)

        # Step 2: Determine answer template
        template = self._select_answer_template(evidence, intent)

        # Step 3: Handle insufficient evidence
        if template == AnswerTemplate.NOT_FOUND:
            return self._generate_not_found_response(query, evidence)

        # Step 4: Generate structured answer
        answer_data = self._generate_structured_answer(query, evidence, template, intent)

        return answer_data

    def _assess_evidence(self, results: List[RetrievalResult], intent: IntentCategory) -> AnswerEvidence:
        """Assess quality and sufficiency of evidence"""

        if not results:
            return AnswerEvidence([], [], [], 0.0, [], ["No relevant documents found"])

        # Categorize evidence by quality and authority
        high_authority = [r for r in results if r.authority_score >= 0.7]
        medium_authority = [r for r in results if 0.4 <= r.authority_score < 0.7]
        low_authority = [r for r in results if r.authority_score < 0.4]

        # Identify primary facts (high confidence numeric data)
        primary_facts = []
        for result in high_authority:
            for fact in result.content_analysis.numeric_facts:
                if fact.confidence >= 0.7:
                    primary_facts.append({
                        'fact': fact,
                        'source': result.document.metadata.get('source_file', 'Unknown'),
                        'authority': result.authority_score
                    })

        # Supporting evidence
        supporting_evidence = [
            {
                'content': result.document.page_content,
                'authority': result.authority_score,
                'citations': result.citations
            }
            for result in high_authority + medium_authority
        ]

        # Conflicting evidence
        conflicting_evidence = []
        all_conflicts = []
        for result in results:
            all_conflicts.extend(result.conflicts)

        if all_conflicts:
            conflicting_evidence = [
                {
                    'conflict': conflict,
                    'sources': [r.document.metadata.get('source_file', 'Unknown') for r in results if conflict in r.conflicts]
                }
                for conflict in set(all_conflicts)
            ]

        # Overall confidence
        if high_authority:
            confidence = sum(r.authority_score for r in high_authority) / len(high_authority)
        else:
            confidence = 0.3

        # All citations
        all_citations = []
        for result in results:
            all_citations.extend(result.citations)

        # Identify gaps
        gaps = []
        if not primary_facts:
            gaps.append("No high-confidence numeric data found")
        if len(high_authority) < 2:
            gaps.append("Insufficient authoritative sources")
        if conflicting_evidence:
            gaps.append("Conflicting information detected")

        return AnswerEvidence(
            primary_facts=primary_facts,
            supporting_evidence=supporting_evidence,
            conflicting_evidence=conflicting_evidence,
            confidence_score=confidence,
            citations=all_citations,
            gaps_identified=gaps
        )

    def _select_answer_template(self, evidence: AnswerEvidence, intent: IntentCategory) -> AnswerTemplate:
        """Select appropriate answer template based on evidence quality"""

        # Check for not-found conditions
        if evidence.confidence_score < 0.2 or not evidence.supporting_evidence:
            return AnswerTemplate.NOT_FOUND

        # Check for insufficient evidence
        if evidence.confidence_score < 0.5 or len(evidence.gaps_identified) > 2:
            return AnswerTemplate.INSUFFICIENT

        # Select template based on intent and evidence type
        if evidence.primary_facts and intent in [IntentCategory.PRICE_CONSUMER, IntentCategory.PRICE_PRODUCTION]:
            return AnswerTemplate.NUMERICAL

        if len(evidence.conflicting_evidence) > 0:
            return AnswerTemplate.COMPARATIVE

        return AnswerTemplate.FACTUAL

    def _generate_not_found_response(self, query: str, evidence: AnswerEvidence) -> Dict:
        """Generate explicit not-found response"""

        response = {
            'answer': f"I cannot provide a reliable answer to '{query}' based on the available documents.",
            'confidence': 0.0,
            'evidence_quality': 'insufficient',
            'gaps_identified': evidence.gaps_identified,
            'suggestions': [
                "The query may require information not present in the current document set",
                "Consider rephrasing the query or checking if relevant documents are properly uploaded",
                "Verify that the requested information exists in the source materials"
            ],
            'citations': [],
            'template_used': 'not_found'
        }

        return response

    def _generate_structured_answer(self, query: str, evidence: AnswerEvidence, template: AnswerTemplate, intent: IntentCategory) -> Dict:
        """Generate structured answer using appropriate template"""

        # Create constrained prompt based on template
        if template == AnswerTemplate.NUMERICAL:
            prompt = self._create_numerical_prompt(query, evidence, intent)
        elif template == AnswerTemplate.COMPARATIVE:
            prompt = self._create_comparative_prompt(query, evidence)
        else:
            prompt = self._create_factual_prompt(query, evidence)

        # Generate answer with strict constraints
        try:
            response = self.llm.invoke(prompt)
            answer_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return self._generate_error_response(query, str(e))

        # Post-process to ensure compliance
        processed_answer = self._post_process_answer(answer_text, evidence, template)

        return {
            'answer': processed_answer,
            'confidence': evidence.confidence_score,
            'evidence_quality': 'high' if evidence.confidence_score >= 0.7 else 'medium',
            'primary_facts': evidence.primary_facts,
            'conflicts_detected': evidence.conflicting_evidence,
            'citations': evidence.citations[:10],  # Limit citation count
            'template_used': template.value,
            'gaps_identified': evidence.gaps_identified
        }

    def _create_numerical_prompt(self, query: str, evidence: AnswerEvidence, intent: IntentCategory) -> str:
        """Create prompt for numerical answers with strict constraints"""

        facts_text = ""
        for fact_data in evidence.primary_facts[:3]:  # Limit to top 3 facts
            fact = fact_data['fact']
            facts_text += f"- {fact.value:.2f} {fact.normalized_unit} (confidence: {fact.confidence:.2f}, source: {fact_data['source']})\n"

        intent_instruction = ""
        if intent == IntentCategory.PRICE_CONSUMER:
            intent_instruction = "Focus specifically on CONSUMER/RETAIL prices. Do not include production or wholesale costs."
        elif intent == IntentCategory.PRICE_PRODUCTION:
            intent_instruction = "Focus specifically on PRODUCTION costs. Do not include consumer or retail prices."

        prompt = f"""Answer the following query using ONLY the provided factual data. Do not extrapolate or add information not explicitly stated.

Query: {query}

{intent_instruction}

High-confidence numerical facts:
{facts_text}

Constraints:
1. Use ONLY the numerical facts provided above
2. Include confidence levels and sources for all numbers
3. If multiple values exist, explain the range and sources
4. Do not make calculations or conversions beyond simple unit clarification
5. Maximum 3 sentences
6. Include proper citations

Answer:"""

        return prompt

    def _create_comparative_prompt(self, query: str, evidence: AnswerEvidence) -> str:
        """Create prompt for comparative answers addressing conflicts"""

        conflicts_text = ""
        for conflict in evidence.conflicting_evidence[:2]:
            conflicts_text += f"- {conflict['conflict']} (sources: {', '.join(conflict['sources'])})\n"

        prompt = f"""Answer the following query while addressing the conflicting information found in the sources.

Query: {query}

Conflicting information detected:
{conflicts_text}

Constraints:
1. Acknowledge the conflicts explicitly
2. Present different perspectives with their sources
3. Do not choose one source over another without clear authority reasoning
4. Maximum 4 sentences
5. Include citations for each conflicting claim

Answer:"""

        return prompt

    def _create_factual_prompt(self, query: str, evidence: AnswerEvidence) -> str:
        """Create prompt for factual answers"""

        evidence_text = ""
        for i, ev in enumerate(evidence.supporting_evidence[:3]):
            evidence_text += f"{i+1}. {ev['content'][:200]}... (authority: {ev['authority']:.2f})\n"

        prompt = f"""Answer the following query using ONLY the provided evidence. Be precise and factual.

Query: {query}

Evidence:
{evidence_text}

Constraints:
1. Use ONLY information explicitly stated in the evidence
2. Do not infer, extrapolate, or add external knowledge
3. Include source attribution for claims
4. Maximum 3 sentences
5. If evidence is partial, acknowledge limitations

Answer:"""

        return prompt

    def _post_process_answer(self, answer: str, evidence: AnswerEvidence, template: AnswerTemplate) -> str:
        """Post-process answer to ensure quality and compliance"""

        # Remove hallucination indicators
        hallucination_patterns = [
            r"according to.*general knowledge",
            r"it is widely known",
            r"typically",
            r"generally",
            r"usually"
        ]

        for pattern in hallucination_patterns:
            answer = re.sub(pattern, "", answer, flags=re.IGNORECASE)

        # Ensure numerical claims have citations
        if template == AnswerTemplate.NUMERICAL:
            numbers = re.findall(r'\$?[\d,]+\.?\d*', answer)
            if numbers and not any(word in answer.lower() for word in ['source:', 'according to', 'from']):
                answer += f" (Source citations required for verification)"

        # Limit length to prevent drift
        sentences = re.split(r'[.!?]+', answer)
        max_sentences = 4 if template == AnswerTemplate.COMPARATIVE else 3
        if len(sentences) > max_sentences:
            answer = '. '.join(sentences[:max_sentences]) + '.'

        return answer.strip()

    def _generate_error_response(self, query: str, error: str) -> Dict:
        """Generate error response"""
        return {
            'answer': f"Unable to process query '{query}' due to system error.",
            'confidence': 0.0,
            'evidence_quality': 'error',
            'error': error,
            'citations': [],
            'template_used': 'error'
        }


def create_advanced_retriever(vector_store, source_validator) -> AdvancedRetriever:
    """Factory function to create advanced retriever"""
    content_analyzer = ContentAnalyzer()
    return AdvancedRetriever(vector_store, source_validator, content_analyzer)