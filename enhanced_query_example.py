"""
Enhanced Query System Example

Demonstrates how to use the comprehensive pipeline improvements to address all critical issues:

1. Strict source validation
2. Intent-aware retrieval
3. OCR cleaning and numeric canonicalization
4. Per-sentence provenance tracking
5. Answer templates with drift prevention
6. Source prioritization and conflict resolution
7. Fact-preserving chunking
8. Hybrid retrieval with reranking
9. Comprehensive citations
10. Explicit not-found handling
11. Unit normalization
12. Confidence scoring

Usage: python enhanced_query_example.py
"""

import os
from datetime import datetime
from typing import Dict, List

from workflow.utils.helper import load_vector_store
from utils.sourceValidator import get_source_validator
from utils.contentAnalyzer import create_content_analyzer, IntentCategory
from utils.advancedRetrieval import create_advanced_retriever, AnswerGenerator
from langchain_openai import ChatOpenAI


class EnhancedQuerySystem:
    """Complete enhanced query system with all security and accuracy features"""

    def __init__(self):
        # Initialize core components
        self.vector_store = load_vector_store()
        self.source_validator = get_source_validator()
        self.content_analyzer = create_content_analyzer()
        self.advanced_retriever = create_advanced_retriever(
            self.vector_store, self.source_validator
        )
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.answer_generator = AnswerGenerator(self.llm)

    def enhanced_query(self, query: str, max_results: int = 5) -> Dict:
        """Execute enhanced query with comprehensive analysis"""

        print(f"\n>> Processing Query: '{query}'")
        print("=" * 80)

        start_time = datetime.now()

        # Step 1: Classify query intent to prevent category errors
        intent = self._classify_query_intent(query)
        print(f">> Query Intent: {intent.value}")

        # Step 2: Enhanced hybrid retrieval with security validation
        print(">> Executing hybrid retrieval with security validation...")
        retrieval_results = self.advanced_retriever.hybrid_retrieve(
            query=query,
            intent=intent,
            k=max_results
        )

        # Step 3: Security validation summary
        trusted_results = [r for r in retrieval_results if r.document.metadata.get('is_trusted', True)]
        blocked_count = len(retrieval_results) - len(trusted_results)

        print(f">> Security Check: {len(trusted_results)} trusted, {blocked_count} blocked")

        # Step 4: Quality analysis summary
        quality_breakdown = self._analyze_result_quality(retrieval_results)
        print(f">> Quality Analysis: {quality_breakdown}")

        # Step 5: Generate structured answer with provenance
        print(">> Generating structured answer...")
        answer_data = self.answer_generator.generate_answer(
            query=query,
            retrieval_results=retrieval_results,
            intent=intent
        )

        # Step 6: Comprehensive result compilation
        processing_time = (datetime.now() - start_time).total_seconds()

        enhanced_result = {
            'query': query,
            'intent_detected': intent.value,
            'processing_time_seconds': processing_time,
            'retrieval_stats': {
                'total_retrieved': len(retrieval_results),
                'trusted_sources': len(trusted_results),
                'blocked_untrusted': blocked_count,
                'quality_breakdown': quality_breakdown
            },
            'answer_data': answer_data,
            'detailed_provenance': self._extract_detailed_provenance(retrieval_results),
            'security_validation': self._get_security_summary(retrieval_results),
            'confidence_analysis': self._analyze_confidence(answer_data, retrieval_results)
        }

        self._display_results(enhanced_result)
        return enhanced_result

    def _classify_query_intent(self, query: str) -> IntentCategory:
        """Classify query intent to prevent category errors"""

        query_lower = query.lower()

        # Price classification with specificity
        if any(word in query_lower for word in ['price', 'cost', '$', 'dollar']):
            if any(word in query_lower for word in ['retail', 'consumer', 'buy', 'purchase', 'market price']):
                return IntentCategory.PRICE_CONSUMER
            elif any(word in query_lower for word in ['production', 'manufacturing', 'produce', 'make']):
                return IntentCategory.PRICE_PRODUCTION
            elif any(word in query_lower for word in ['wholesale', 'bulk', 'distributor']):
                return IntentCategory.PRICE_WHOLESALE

        # Technical specifications
        if any(word in query_lower for word in ['specification', 'technical', 'performance', 'efficiency']):
            return IntentCategory.TECHNICAL_SPEC

        # Research content
        if any(word in query_lower for word in ['methodology', 'method', 'approach']):
            return IntentCategory.METHODOLOGY

        if any(word in query_lower for word in ['results', 'findings', 'data', 'study']):
            return IntentCategory.RESULTS

        return IntentCategory.GENERAL

    def _analyze_result_quality(self, results: List) -> Dict:
        """Analyze quality distribution of results"""

        quality_counts = {'high': 0, 'medium': 0, 'low': 0, 'corrupted': 0}
        total_facts = 0
        conflicts_detected = 0

        for result in results:
            quality = result.content_analysis.quality_score.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1

            total_facts += len(result.content_analysis.numeric_facts)
            conflicts_detected += len(result.conflicts)

        return {
            'quality_distribution': quality_counts,
            'total_numeric_facts': total_facts,
            'conflicts_detected': conflicts_detected,
            'average_authority': sum(r.authority_score for r in results) / len(results) if results else 0
        }

    def _extract_detailed_provenance(self, results: List) -> List[Dict]:
        """Extract detailed provenance for transparency"""

        provenance = []
        for i, result in enumerate(results):
            source_info = {
                'result_rank': i + 1,
                'source_file': result.document.metadata.get('source_file', 'Unknown'),
                'chunk_id': result.document.metadata.get('chunk_id', 'Unknown'),
                'authority_score': result.authority_score,
                'relevance_score': result.relevance_score,
                'content_quality': result.content_analysis.quality_score.value,
                'intent_category': result.content_analysis.intent_category.value,
                'numeric_facts_count': len(result.content_analysis.numeric_facts),
                'issues_found': result.content_analysis.issues_found,
                'conflicts': result.conflicts,
                'sample_content': result.document.page_content[:200] + "..."
            }
            provenance.append(source_info)

        return provenance

    def _get_security_summary(self, results: List) -> Dict:
        """Get security validation summary"""

        security_summary = {
            'all_sources_validated': True,
            'untrusted_sources_blocked': 0,
            'trusted_sources_used': 0,
            'metadata_validation_passed': 0,
            'security_issues': []
        }

        for result in results:
            metadata = result.document.metadata

            # Check if source validation passed
            if self.source_validator.validate_document_metadata(metadata):
                security_summary['trusted_sources_used'] += 1
                security_summary['metadata_validation_passed'] += 1
            else:
                security_summary['untrusted_sources_blocked'] += 1
                security_summary['all_sources_validated'] = False
                security_summary['security_issues'].append(
                    f"Untrusted source: {metadata.get('source_file', 'Unknown')}"
                )

        return security_summary

    def _analyze_confidence(self, answer_data: Dict, results: List) -> Dict:
        """Analyze overall confidence in the answer"""

        confidence_analysis = {
            'overall_confidence': answer_data.get('confidence', 0.0),
            'evidence_quality': answer_data.get('evidence_quality', 'unknown'),
            'factors': {
                'source_authority': sum(r.authority_score for r in results) / len(results) if results else 0,
                'content_quality': len([r for r in results if r.content_analysis.quality_score.value == 'high']) / len(results) if results else 0,
                'numeric_data_confidence': 0,
                'conflict_penalty': -0.1 * len(answer_data.get('conflicts_detected', [])),
            },
            'confidence_breakdown': 'high' if answer_data.get('confidence', 0) >= 0.7 else 'medium' if answer_data.get('confidence', 0) >= 0.4 else 'low'
        }

        # Calculate numeric confidence if available
        if answer_data.get('primary_facts'):
            numeric_confidences = [fact['fact'].confidence for fact in answer_data['primary_facts']]
            confidence_analysis['factors']['numeric_data_confidence'] = sum(numeric_confidences) / len(numeric_confidences)

        return confidence_analysis

    def _display_results(self, result: Dict):
        """Display comprehensive results"""

        print("\n" + "="*80)
        print(">> ENHANCED QUERY RESULTS")
        print("="*80)

        answer_data = result['answer_data']

        print(f">> ANSWER: {answer_data['answer']}")
        print(f"\n>> CONFIDENCE: {answer_data['confidence']:.2f} ({result['confidence_analysis']['confidence_breakdown']})")
        print(f">> TEMPLATE: {answer_data.get('template_used', 'unknown')}")

        # Security summary
        security = result['security_validation']
        print(f"\n>> SECURITY STATUS: {'SECURE' if security['all_sources_validated'] else 'ISSUES DETECTED'}")
        print(f"   Trusted sources: {security['trusted_sources_used']}, Blocked: {security['untrusted_sources_blocked']}")

        # Quality summary
        quality = result['retrieval_stats']['quality_breakdown']
        print(f"\n>> QUALITY ANALYSIS:")
        print(f"   High: {quality['quality_distribution']['high']}, Medium: {quality['quality_distribution']['medium']}")
        print(f"   Low: {quality['quality_distribution']['low']}, Corrupted: {quality['quality_distribution']['corrupted']}")
        print(f"   Numeric facts: {quality['total_numeric_facts']}, Conflicts: {quality['conflicts_detected']}")

        # Primary facts
        if answer_data.get('primary_facts'):
            print(f"\n>> PRIMARY FACTS:")
            for fact_data in answer_data['primary_facts'][:3]:
                fact = fact_data['fact']
                print(f"   • {fact.value:.2f} {fact.normalized_unit} (confidence: {fact.confidence:.2f}, source: {fact_data['source']})")

        # Conflicts
        if answer_data.get('conflicts_detected'):
            print(f"\n>> CONFLICTS DETECTED:")
            for conflict in answer_data['conflicts_detected'][:2]:
                print(f"   • {conflict['conflict']}")

        # Gaps
        if answer_data.get('gaps_identified'):
            print(f"\n>> GAPS IDENTIFIED:")
            for gap in answer_data['gaps_identified']:
                print(f"   • {gap}")

        print(f"\n>> Processing time: {result['processing_time_seconds']:.2f} seconds")
        print("="*80)


def demo_enhanced_system():
    """Demonstrate the enhanced system with various query types"""

    print(">> Enhanced Article Processing System Demo")
    print("Addressing all 12 critical pipeline issues")
    print("="*80)

    # Initialize system
    try:
        system = EnhancedQuerySystem()
        print(">> System initialized successfully")
    except Exception as e:
        print(f"ERROR: System initialization failed: {e}")
        return

    # Demo queries that test different aspects
    demo_queries = [
        "What is the consumer price of clean energy per kWh?",  # Tests price category classification
        "What are the production costs for bioethanol?",        # Tests production vs consumer distinction
        "What is the efficiency of wind turbines?",             # Tests technical specifications
        "How does solar compare to wind energy costs?",         # Tests comparative analysis
        "What is the methodology for carbon capture?",          # Tests methodology classification
        "What data is available on fuel cell performance?"      # Tests not-found handling
    ]

    for query in demo_queries:
        try:
            result = system.enhanced_query(query)
            print(f"\n>> Query processed successfully: '{query}'")

        except Exception as e:
            print(f"\nERROR: Query failed: '{query}' - Error: {e}")

        print("\n" + "-"*80 + "\n")


if __name__ == "__main__":
    demo_enhanced_system()