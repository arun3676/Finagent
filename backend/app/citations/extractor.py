"""
Claim Extractor

Extracts verifiable claims from generated responses.
Claims are statements that can be traced back to source documents.

Claim types:
- Numerical: Contains specific numbers/metrics
- Factual: States a fact about a company/event
- Comparative: Compares values or entities
- Temporal: References specific time periods

Usage:
    extractor = ClaimExtractor()
    claims = extractor.extract_claims(response_text)
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ClaimType(Enum):
    """Types of extractable claims."""
    NUMERICAL = "numerical"
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    TEMPORAL = "temporal"
    UNKNOWN = "unknown"


@dataclass
class ExtractedClaim:
    """A claim extracted from response text."""
    text: str
    claim_type: ClaimType
    position: int  # Character position in original text
    entities: List[str]  # Companies, tickers mentioned
    numbers: List[str]  # Numerical values in claim
    time_references: List[str]  # Dates, quarters, years


class ClaimExtractor:
    """
    Extracts verifiable claims from text.
    
    Uses pattern matching and NLP to identify
    statements that should be cited.
    """
    
    # Patterns for numerical claims
    NUMERICAL_PATTERNS = [
        r'\$[\d,.]+\s*(?:million|billion|M|B)?',
        r'[\d,.]+\s*(?:percent|%)',
        r'[\d,.]+x',
        r'revenue of \$?[\d,.]+',
        r'grew [\d,.]+%',
        r'increased [\d,.]+%',
        r'decreased [\d,.]+%'
    ]
    
    # Patterns for time references
    TIME_PATTERNS = [
        r'Q[1-4]\s*20\d{2}',
        r'FY\s*20\d{2}',
        r'20\d{2}',
        r'(?:first|second|third|fourth)\s+quarter',
        r'fiscal\s+year\s+20\d{2}'
    ]
    
    # Patterns for company references
    COMPANY_PATTERNS = [
        r'\b[A-Z]{1,5}\b',  # Ticker symbols
        r'(?:Apple|Microsoft|Google|Amazon|Tesla|Meta|Netflix|Nvidia)',
    ]
    
    def __init__(self, use_nlp: bool = False):
        """
        Initialize claim extractor.
        
        Args:
            use_nlp: Use NLP for advanced extraction (requires spacy)
        """
        self.use_nlp = use_nlp
        self._nlp = None
        
        if use_nlp:
            # TODO: Initialize spacy model
            pass
    
    def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract all verifiable claims from text.
        
        Args:
            text: Response text to analyze
            
        Returns:
            List of ExtractedClaim objects
        """
        claims = []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            if self._is_claimable(sentence):
                claim = self._create_claim(sentence, text)
                claims.append(claim)
        
        return claims
    
    def extract_numerical_claims(self, text: str) -> List[ExtractedClaim]:
        """
        Extract only numerical claims.
        
        Args:
            text: Response text
            
        Returns:
            List of numerical claims
        """
        all_claims = self.extract_claims(text)
        return [c for c in all_claims if c.claim_type == ClaimType.NUMERICAL]
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (could use nltk for better results)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_claimable(self, sentence: str) -> bool:
        """
        Check if a sentence contains a verifiable claim.
        
        Args:
            sentence: Sentence to check
            
        Returns:
            True if sentence contains a claim
        """
        # Check for numerical content
        for pattern in self.NUMERICAL_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        
        # Check for factual indicators
        factual_indicators = [
            "reported", "announced", "disclosed", "stated",
            "according to", "based on", "as of"
        ]
        for indicator in factual_indicators:
            if indicator in sentence.lower():
                return True
        
        return False
    
    def _create_claim(self, sentence: str, full_text: str) -> ExtractedClaim:
        """
        Create an ExtractedClaim from a sentence.
        
        Args:
            sentence: The claim sentence
            full_text: Full response text
            
        Returns:
            ExtractedClaim object
        """
        # Determine claim type
        claim_type = self._classify_claim(sentence)
        
        # Find position in original text
        position = full_text.find(sentence)
        
        # Extract entities
        entities = self._extract_entities(sentence)
        
        # Extract numbers
        numbers = self._extract_numbers(sentence)
        
        # Extract time references
        time_refs = self._extract_time_references(sentence)
        
        return ExtractedClaim(
            text=sentence,
            claim_type=claim_type,
            position=position,
            entities=entities,
            numbers=numbers,
            time_references=time_refs
        )
    
    def _classify_claim(self, sentence: str) -> ClaimType:
        """
        Classify the type of claim.
        
        Args:
            sentence: Claim sentence
            
        Returns:
            ClaimType enum value
        """
        # Check for numerical
        for pattern in self.NUMERICAL_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return ClaimType.NUMERICAL
        
        # Check for comparative
        comparative_words = ["compared to", "versus", "vs", "higher than", "lower than", "more than", "less than"]
        for word in comparative_words:
            if word in sentence.lower():
                return ClaimType.COMPARATIVE
        
        # Check for temporal
        for pattern in self.TIME_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                return ClaimType.TEMPORAL
        
        return ClaimType.FACTUAL
    
    def _extract_entities(self, sentence: str) -> List[str]:
        """
        Extract company/ticker entities from sentence.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            List of entity strings
        """
        entities = []
        
        # Find ticker symbols (uppercase 1-5 letters)
        tickers = re.findall(r'\b([A-Z]{1,5})\b', sentence)
        entities.extend(tickers)
        
        # Find company names
        companies = [
            "Apple", "Microsoft", "Google", "Alphabet", "Amazon",
            "Tesla", "Meta", "Netflix", "Nvidia", "AMD"
        ]
        for company in companies:
            if company.lower() in sentence.lower():
                entities.append(company)
        
        return list(set(entities))
    
    def _extract_numbers(self, sentence: str) -> List[str]:
        """
        Extract numerical values from sentence.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            List of number strings
        """
        numbers = []
        
        # Find dollar amounts
        dollars = re.findall(r'\$[\d,.]+\s*(?:million|billion|M|B)?', sentence)
        numbers.extend(dollars)
        
        # Find percentages
        percentages = re.findall(r'[\d,.]+\s*(?:percent|%)', sentence)
        numbers.extend(percentages)
        
        # Find plain numbers with context
        plain = re.findall(r'(?:^|\s)([\d,.]+)(?:\s|$)', sentence)
        numbers.extend(plain)
        
        return numbers
    
    def _extract_time_references(self, sentence: str) -> List[str]:
        """
        Extract time references from sentence.
        
        Args:
            sentence: Sentence to analyze
            
        Returns:
            List of time reference strings
        """
        time_refs = []
        
        for pattern in self.TIME_PATTERNS:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            time_refs.extend(matches)
        
        return time_refs
