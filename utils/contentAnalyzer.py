"""
Advanced Content Analysis and Quality Control Module

Addresses critical pipeline issues:
- Intent/taxonomy classification
- Numeric canonicalization and OCR cleaning
- Unit normalization
- Confidence scoring
- Content quality validation
"""

import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class IntentCategory(Enum):
    """Content intent classification"""
    PRICE_CONSUMER = "price_consumer"
    PRICE_PRODUCTION = "price_production"
    PRICE_WHOLESALE = "price_wholesale"
    TECHNICAL_SPEC = "technical_specification"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    ABSTRACT = "abstract"
    CONCLUSION = "conclusion"
    GENERAL = "general"


class ContentQuality(Enum):
    """Content quality levels"""
    HIGH = "high"          # Authoritative, well-structured
    MEDIUM = "medium"      # Good but may have minor issues
    LOW = "low"           # Poor quality, should be flagged
    CORRUPTED = "corrupted" # OCR artifacts, unusable


@dataclass
class NumericFact:
    """Structured numeric information with metadata"""
    value: float
    unit: str
    normalized_unit: str
    context: str
    confidence: float
    source_span: str
    provenance: Dict


@dataclass
class ContentAnalysis:
    """Complete content analysis result"""
    intent_category: IntentCategory
    quality_score: ContentQuality
    numeric_facts: List[NumericFact]
    cleaned_content: str
    confidence_score: float
    issues_found: List[str]
    provenance_spans: List[Dict]


class ContentAnalyzer:
    """Advanced content analysis and quality control"""

    def __init__(self):
        self.price_keywords = {
            'consumer': ['retail', 'consumer', 'market price', 'selling price', 'msrp', 'retail cost'],
            'production': ['production cost', 'manufacturing', 'cost to produce', 'production expense'],
            'wholesale': ['wholesale', 'bulk price', 'distributor', 'trade price']
        }

        self.unit_conversions = {
            # Weight conversions to kg
            'ton': 1000, 'tonnes': 1000, 'metric ton': 1000,
            'lb': 0.453592, 'lbs': 0.453592, 'pound': 0.453592, 'pounds': 0.453592,
            'g': 0.001, 'gram': 0.001, 'grams': 0.001,
            'kg': 1, 'kilogram': 1, 'kilograms': 1,

            # Volume conversions to liter
            'gal': 3.78541, 'gallon': 3.78541, 'gallons': 3.78541,
            'l': 1, 'liter': 1, 'liters': 1, 'litre': 1, 'litres': 1,
            'ml': 0.001, 'milliliter': 0.001, 'milliliters': 0.001,

            # Energy conversions to MJ
            'kwh': 3.6, 'kWh': 3.6, 'kilowatt-hour': 3.6,
            'btu': 0.001055, 'BTU': 0.001055,
            'mj': 1, 'MJ': 1, 'megajoule': 1
        }

    def analyze_content(self, content: str, metadata: Dict) -> ContentAnalysis:
        """Comprehensive content analysis"""

        # Clean OCR artifacts and normalize
        cleaned_content = self._clean_ocr_artifacts(content)

        # Classify intent/taxonomy
        intent = self._classify_intent(cleaned_content, metadata)

        # Assess content quality
        quality = self._assess_quality(cleaned_content)

        # Extract and normalize numeric facts
        numeric_facts = self._extract_numeric_facts(cleaned_content)

        # Calculate confidence score
        confidence = self._calculate_confidence(cleaned_content, metadata, numeric_facts)

        # Identify issues
        issues = self._identify_issues(content, cleaned_content, numeric_facts)

        # Create provenance spans
        provenance_spans = self._create_provenance_spans(cleaned_content, metadata)

        return ContentAnalysis(
            intent_category=intent,
            quality_score=quality,
            numeric_facts=numeric_facts,
            cleaned_content=cleaned_content,
            confidence_score=confidence,
            issues_found=issues,
            provenance_spans=provenance_spans
        )

    def _clean_ocr_artifacts(self, content: str) -> str:
        """Clean OCR artifacts and formatting issues"""

        # Fix spaced-out characters (e.g., '3 a n d 3and7' -> '3 and 3 and 7')
        content = re.sub(r'(\d)\s*a\s*n\s*d\s*(\d)', r'\1 and \2', content, flags=re.IGNORECASE)

        # Fix broken numeric patterns
        content = re.sub(r'(\d)\s+([a-z])\s+([a-z])\s+([a-z])\s*(\d)', r'\1\2\3\4\5', content)

        # Clean excessive whitespace
        content = re.sub(r'\s+', ' ', content)

        # Fix common OCR character substitutions
        ocr_fixes = {
            'O': '0',  # In numeric contexts
            'l': '1',  # In numeric contexts
            'I': '1',  # In numeric contexts
            'S': '5',  # In numeric contexts
        }

        # Apply OCR fixes in numeric contexts only
        for incorrect, correct in ocr_fixes.items():
            # Only replace in patterns that look like corrupted numbers
            pattern = r'(\$?\s*)' + re.escape(incorrect) + r'(\d+|\.\d+)'
            # Use lambda to avoid backreference issues with digit replacements
            content = re.sub(pattern, lambda m: m.group(1) + correct + m.group(2), content)

        # Fix broken currency patterns
        content = re.sub(r'\$\s*(\d)', r'$\1', content)
        content = re.sub(r'(\d)\s*USD', r'\1 USD', content)

        return content.strip()

    def _classify_intent(self, content: str, metadata: Dict) -> IntentCategory:
        """Classify content intent for better retrieval targeting"""

        content_lower = content.lower()
        semantic_type = metadata.get('semantic_type', 'content')

        # Check for price-related content with specific classification
        if any(word in content_lower for word in ['price', 'cost', '$', 'dollar', 'usd']):

            # Consumer/retail price indicators
            if any(word in content_lower for word in self.price_keywords['consumer']):
                return IntentCategory.PRICE_CONSUMER

            # Production cost indicators
            elif any(word in content_lower for word in self.price_keywords['production']):
                return IntentCategory.PRICE_PRODUCTION

            # Wholesale price indicators
            elif any(word in content_lower for word in self.price_keywords['wholesale']):
                return IntentCategory.PRICE_WHOLESALE

        # Technical specifications
        if any(word in content_lower for word in ['specification', 'technical', 'parameter', 'performance']):
            return IntentCategory.TECHNICAL_SPEC

        # Map semantic types to intent categories
        semantic_mapping = {
            'methodology': IntentCategory.METHODOLOGY,
            'abstract': IntentCategory.ABSTRACT,
            'conclusion': IntentCategory.CONCLUSION,
            'results': IntentCategory.RESULTS
        }

        return semantic_mapping.get(semantic_type, IntentCategory.GENERAL)

    def _assess_quality(self, content: str) -> ContentQuality:
        """Assess content quality and flag potential issues"""

        # Check for corruption indicators
        corruption_patterns = [
            r'[a-zA-Z]\s+[a-zA-Z]\s+[a-zA-Z]',  # Spaced out words
            r'\d\s*[a-zA-Z]\s*[a-zA-Z]\s*\d',   # Mixed digit-letter patterns
            r'[^\w\s\.\,\!\?\-\$\%\(\)]{3,}',   # Multiple special characters
        ]

        for pattern in corruption_patterns:
            if re.search(pattern, content):
                return ContentQuality.CORRUPTED

        # Quality indicators
        quality_score = 0

        # Positive indicators
        if len(content) > 100: quality_score += 1
        if re.search(r'\d+\.?\d*\s*[a-zA-Z]+', content): quality_score += 1  # Numbers with units
        if re.search(r'[A-Z][a-z]+\s[A-Z][a-z]+', content): quality_score += 1  # Proper capitalization
        if content.count('.') >= 2: quality_score += 1  # Multiple sentences

        # Negative indicators
        if len(content) < 50: quality_score -= 1
        if len(re.findall(r'[^\w\s]', content)) / len(content) > 0.1: quality_score -= 1  # Too many special chars

        if quality_score >= 3: return ContentQuality.HIGH
        elif quality_score >= 1: return ContentQuality.MEDIUM
        else: return ContentQuality.LOW

    def _extract_numeric_facts(self, content: str) -> List[NumericFact]:
        """Extract and normalize numeric facts with units"""

        facts = []

        # Enhanced numeric pattern matching
        patterns = [
            # Currency patterns: $123.45, $123 USD, 123 dollars
            r'(\$?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(USD|dollars?|cents?)?)',
            # Measurement patterns: 123.45 kg/ton/etc
            r'((\d+(?:\.\d+)?)\s*(kg|ton|tonnes?|lb|lbs?|pounds?|g|grams?|l|liters?|litres?|gal|gallons?|ml|kwh|btu|mj)(?:/[a-zA-Z]+)?)',
            # Percentage patterns
            r'((\d+(?:\.\d+)?)\s*%)',
            # Ratio patterns: 3:1, 3 to 1
            r'((\d+(?:\.\d+)?)\s*(?::|to)\s*(\d+(?:\.\d+)?))'
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                try:
                    full_match = match.group(1)
                    numeric_part = float(re.sub(r'[^\d\.]', '', match.group(2)))

                    # Extract unit
                    unit_match = re.search(r'[a-zA-Z%]+', full_match)
                    unit = unit_match.group(0) if unit_match else ''

                    # Normalize unit
                    normalized_unit = self._normalize_unit(unit)

                    # Calculate normalized value
                    normalized_value = self._convert_to_normalized_unit(numeric_part, unit)

                    # Extract context (surrounding text)
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end].strip()

                    # Calculate confidence based on context and formatting
                    confidence = self._calculate_numeric_confidence(full_match, context)

                    fact = NumericFact(
                        value=normalized_value,
                        unit=unit,
                        normalized_unit=normalized_unit,
                        context=context,
                        confidence=confidence,
                        source_span=full_match,
                        provenance={
                            'start_pos': match.start(),
                            'end_pos': match.end(),
                            'original_text': full_match
                        }
                    )
                    facts.append(fact)

                except (ValueError, AttributeError):
                    continue

        return facts

    def _normalize_unit(self, unit: str) -> str:
        """Normalize units to standard forms"""
        unit_lower = unit.lower()

        # Weight units -> kg
        if unit_lower in ['ton', 'tonnes', 'metric ton']:
            return 'kg'
        elif unit_lower in ['lb', 'lbs', 'pound', 'pounds']:
            return 'kg'
        elif unit_lower in ['g', 'gram', 'grams']:
            return 'kg'
        elif unit_lower in ['kg', 'kilogram', 'kilograms']:
            return 'kg'

        # Volume units -> L
        elif unit_lower in ['gal', 'gallon', 'gallons']:
            return 'L'
        elif unit_lower in ['l', 'liter', 'liters', 'litre', 'litres']:
            return 'L'
        elif unit_lower in ['ml', 'milliliter', 'milliliters']:
            return 'L'

        # Energy units -> MJ
        elif unit_lower in ['kwh', 'kilowatt-hour']:
            return 'MJ'
        elif unit_lower in ['btu']:
            return 'MJ'
        elif unit_lower in ['mj', 'megajoule']:
            return 'MJ'

        # Currency -> USD
        elif unit_lower in ['usd', 'dollars', 'dollar']:
            return 'USD'

        return unit

    def _convert_to_normalized_unit(self, value: float, unit: str) -> float:
        """Convert value to normalized unit"""
        unit_lower = unit.lower()
        conversion_factor = self.unit_conversions.get(unit_lower, 1.0)
        return value * conversion_factor

    def _calculate_numeric_confidence(self, numeric_text: str, context: str) -> float:
        """Calculate confidence score for numeric facts"""
        confidence = 0.5  # Base confidence

        # Positive indicators
        if '$' in numeric_text: confidence += 0.2
        if re.search(r'\d+\.\d{2}', numeric_text): confidence += 0.1  # Precise decimals
        if any(word in context.lower() for word in ['cost', 'price', 'value', 'amount']): confidence += 0.2
        if re.search(r'(approximately|about|around)', context.lower()): confidence -= 0.1

        # Negative indicators
        if re.search(r'[^\w\s\$\.\,]', numeric_text): confidence -= 0.2  # Special characters
        if len(context) < 10: confidence -= 0.1  # Little context

        return max(0.0, min(1.0, confidence))

    def _calculate_confidence(self, content: str, metadata: Dict, numeric_facts: List[NumericFact]) -> float:
        """Calculate overall content confidence score"""

        base_confidence = 0.5

        # Source quality indicators
        if metadata.get('semantic_type') in ['abstract', 'methodology', 'results']:
            base_confidence += 0.2

        # Content quality indicators
        if len(content) > 200: base_confidence += 0.1
        if len(numeric_facts) > 0: base_confidence += 0.1

        # Average numeric fact confidence
        if numeric_facts:
            avg_numeric_confidence = sum(fact.confidence for fact in numeric_facts) / len(numeric_facts)
            base_confidence = (base_confidence + avg_numeric_confidence) / 2

        return max(0.0, min(1.0, base_confidence))

    def _identify_issues(self, original: str, cleaned: str, numeric_facts: List[NumericFact]) -> List[str]:
        """Identify potential content issues"""
        issues = []

        if original != cleaned:
            issues.append("OCR_ARTIFACTS_DETECTED")

        if len(numeric_facts) == 0 and any(char.isdigit() for char in original):
            issues.append("NUMERIC_EXTRACTION_FAILED")

        # Check for conflicting numeric values
        if len(numeric_facts) > 1:
            normalized_values = [fact.value for fact in numeric_facts if fact.normalized_unit == numeric_facts[0].normalized_unit]
            if len(set(normalized_values)) > 1:
                issues.append("CONFLICTING_NUMERIC_VALUES")

        # Check for low confidence facts
        low_confidence_facts = [fact for fact in numeric_facts if fact.confidence < 0.3]
        if low_confidence_facts:
            issues.append("LOW_CONFIDENCE_NUMERIC_DATA")

        return issues

    def _create_provenance_spans(self, content: str, metadata: Dict) -> List[Dict]:
        """Create detailed provenance tracking for sentences"""

        sentences = re.split(r'[.!?]+', content)
        provenance_spans = []

        char_position = 0
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                span = {
                    'sentence_id': i,
                    'text': sentence.strip(),
                    'start_pos': char_position,
                    'end_pos': char_position + len(sentence),
                    'source_file': metadata.get('source_file', 'Unknown'),
                    'chunk_id': metadata.get('chunk_id', 'Unknown'),
                    'header': metadata.get('header', ''),
                    'semantic_type': metadata.get('semantic_type', 'content')
                }
                provenance_spans.append(span)
            char_position += len(sentence) + 1

        return provenance_spans


def create_content_analyzer() -> ContentAnalyzer:
    """Factory function to create content analyzer"""
    return ContentAnalyzer()