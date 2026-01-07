"""
Validator Agent

Quality checks responses before returning to user.
Validates factual accuracy, citation quality, and completeness.

Validation checks:
1. Factual accuracy - Claims match sources
2. Citation coverage - Key claims are cited
3. Completeness - Query is fully answered
4. Numerical accuracy - Calculations are correct
5. Professional tone - Appropriate language

Usage:
    validator = Validator()
    result = await validator.validate(query, response, sources)
"""

from typing import List, Optional, Dict, Any, Tuple
import logging
import re
import json
from openai import AsyncOpenAI

from app.config import settings
from app.models import AgentState, RetrievedDocument, Citation
from app.agents.prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_TEMPLATE

logger = logging.getLogger(__name__)


class ValidationResult:
    """Result of validation check."""
    
    def __init__(
        self,
        is_valid: bool,
        score: float,
        feedback: str,
        issues: List[str] = None
    ):
        self.is_valid = is_valid
        self.score = score  # 0-100
        self.feedback = feedback
        self.issues = issues or []


class Validator:
    """
    Response validation agent.
    
    Performs quality checks on generated responses
    before returning to users.
    """
    
    # Validation thresholds
    MIN_VALID_SCORE = 70
    MIN_CITATION_COVERAGE = 0.5
    
    def __init__(self, model: str = None):
        """
        Initialize validator.
        
        Args:
            model: LLM model to use
        """
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def validate(
        self,
        query: str,
        response: str,
        citations: List[Citation],
        documents: List[RetrievedDocument]
    ) -> ValidationResult:
        """
        Validate a response.
        
        Args:
            query: Original query
            response: Generated response
            citations: Response citations
            documents: Source documents
            
        Returns:
            ValidationResult with score and feedback
        """
        logger.info(f"Validating response for query: '{query[:50]}...'")
        
        all_issues = []
        scores = {}
        
        # 1. Check factual accuracy (CRITICAL for hallucination prevention)
        factual_score, factual_issues = await self.check_factual_accuracy(
            response, citations, documents
        )
        scores["factual_accuracy"] = factual_score
        all_issues.extend(factual_issues)
        
        # 2. Check citation coverage
        citation_score, citation_issues = self.check_citation_coverage(
            response, citations
        )
        scores["citation_coverage"] = citation_score
        all_issues.extend(citation_issues)
        
        # 3. Check completeness
        completeness_score, completeness_issues = await self.check_completeness(
            query, response
        )
        scores["completeness"] = completeness_score
        all_issues.extend(completeness_issues)
        
        # 4. Check numerical accuracy (CRITICAL for financial data)
        numerical_score, numerical_issues = self.check_numerical_accuracy(
            response, documents
        )
        scores["numerical_accuracy"] = numerical_score
        all_issues.extend(numerical_issues)
        
        # 5. Aggregate scores
        overall_score = self._aggregate_scores(scores)
        is_valid = overall_score >= self.MIN_VALID_SCORE and len(all_issues) == 0
        
        feedback = self._build_feedback(scores, all_issues, is_valid)
        
        logger.info(f"Validation result: score={overall_score:.1f}, valid={is_valid}, issues={len(all_issues)}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=overall_score,
            feedback=feedback,
            issues=all_issues
        )
    
    async def validate_for_state(self, state: AgentState) -> AgentState:
        """
        Validate response and update agent state.
        
        LangGraph-compatible interface.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with validation results
        """
        result = await self.validate(
            state.original_query,
            state.draft_response,
            state.citations,
            state.retrieved_docs
        )
        
        state.is_valid = result.is_valid
        state.validation_feedback = result.feedback
        
        return state
    
    async def check_factual_accuracy(
        self,
        response: str,
        citations: List[Citation],
        documents: List[RetrievedDocument]
    ) -> Tuple[float, List[str]]:
        """
        Check if claims in response match sources.
        
        CRITICAL: This is the primary defense against hallucinations.
        
        Args:
            response: Generated response
            citations: Response citations
            documents: Source documents
            
        Returns:
            Tuple of (score, issues)
        """
        issues = []
        
        if not citations:
            issues.append("No citations provided for verification")
            return 0.0, issues
        
        # Extract claims from response (sentences with factual content)
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        factual_sentences = [
            s for s in sentences
            if any(keyword in s.lower() for keyword in 
                   ['revenue', 'income', 'profit', 'loss', 'margin', 'growth', 
                    'increased', 'decreased', 'reported', '$', '%'])
        ]
        
        if not factual_sentences:
            return 100.0, []  # No factual claims to verify
        
        # Build citation map
        citation_map = {c.citation_id: c for c in citations}
        
        # Check each claim has supporting evidence
        verified_claims = 0
        for sentence in factual_sentences:
            # Find citation markers in sentence
            citation_refs = re.findall(r'\[(\d+)\]', sentence)
            
            if not citation_refs:
                issues.append(f"Uncited claim: '{sentence[:80]}...'")
                continue
            
            # Verify claim against cited sources
            claim_verified = False
            for ref in citation_refs:
                citation = citation_map.get(ref)
                if citation:
                    # Check similarity between claim and source
                    similarity = self._compute_claim_similarity(
                        sentence, citation.source_text
                    )
                    if similarity > 0.8:  # High threshold for factual accuracy
                        claim_verified = True
                        break
            
            if claim_verified:
                verified_claims += 1
            else:
                issues.append(f"Claim not supported by citation: '{sentence[:80]}...'")
        
        accuracy = (verified_claims / len(factual_sentences)) * 100 if factual_sentences else 100.0
        
        return accuracy, issues
    
    def check_citation_coverage(
        self,
        response: str,
        citations: List[Citation]
    ) -> Tuple[float, List[str]]:
        """
        Check if key claims are properly cited.
        
        Args:
            response: Generated response
            citations: Response citations
            
        Returns:
            Tuple of (score, issues)
        """
        issues = []
        
        # Count sentences with numerical claims
        import re
        sentences = response.split('.')
        numerical_sentences = [
            s for s in sentences 
            if re.search(r'\$?\d+(?:\.\d+)?(?:\s*(?:million|billion|%|percent))?', s)
        ]
        
        # Check if numerical claims have citations
        cited_sentences = [
            s for s in numerical_sentences
            if re.search(r'\[\d+\]', s)
        ]
        
        if numerical_sentences:
            coverage = len(cited_sentences) / len(numerical_sentences)
        else:
            coverage = 1.0  # No numerical claims to cite
        
        if coverage < self.MIN_CITATION_COVERAGE:
            issues.append(
                f"Only {coverage:.0%} of numerical claims are cited. "
                f"Expected at least {self.MIN_CITATION_COVERAGE:.0%}."
            )
        
        return coverage * 100, issues
    
    async def check_completeness(
        self,
        query: str,
        response: str
    ) -> Tuple[float, List[str]]:
        """
        Check if response fully answers the query.
        
        Args:
            query: Original query
            response: Generated response
            
        Returns:
            Tuple of (score, issues)
        """
        issues = []
        
        # Extract key entities/topics from query
        query_lower = query.lower()
        
        # Check for company mentions
        companies_in_query = re.findall(r'\b[A-Z]{2,5}\b', query)
        if companies_in_query:
            for company in companies_in_query:
                if company not in response:
                    issues.append(f"Query mentions {company} but response doesn't address it")
        
        # Check for specific metrics requested
        metrics = ['revenue', 'profit', 'margin', 'growth', 'ebitda', 'earnings', 'loss']
        requested_metrics = [m for m in metrics if m in query_lower]
        
        if requested_metrics:
            missing_metrics = [m for m in requested_metrics if m not in response.lower()]
            if missing_metrics:
                issues.append(f"Query asks about {', '.join(missing_metrics)} but not addressed")
        
        # Check response length (very short responses likely incomplete)
        if len(response.split()) < 30:
            issues.append("Response is too brief to fully answer the query")
        
        # Score based on issues
        if not issues:
            return 100.0, []
        elif len(issues) == 1:
            return 70.0, issues
        else:
            return 50.0, issues
    
    def check_numerical_accuracy(
        self,
        response: str,
        documents: List[RetrievedDocument]
    ) -> Tuple[float, List[str]]:
        """
        Verify numerical values in response match sources.
        
        Args:
            response: Generated response
            documents: Source documents
            
        Returns:
            Tuple of (score, issues)
        """
        import re
        
        issues = []
        
        # Extract numbers from response
        response_numbers = set(re.findall(r'\$?([\d,]+(?:\.\d+)?)', response))
        
        # Extract numbers from source documents
        source_text = " ".join(d.chunk.content for d in documents)
        source_numbers = set(re.findall(r'\$?([\d,]+(?:\.\d+)?)', source_text))
        
        # Check if response numbers appear in sources
        unsupported = response_numbers - source_numbers
        
        # Filter out common numbers (years, small integers)
        unsupported = {
            n for n in unsupported 
            if not (n.isdigit() and (int(n) < 100 or 1900 < int(n) < 2100))
        }
        
        if unsupported:
            issues.append(
                f"Numbers not found in sources: {', '.join(list(unsupported)[:5])}"
            )
        
        if response_numbers:
            accuracy = 1 - (len(unsupported) / len(response_numbers))
        else:
            accuracy = 1.0
        
        return accuracy * 100, issues
    
    def _compute_claim_similarity(self, claim: str, source: str) -> float:
        """
        Compute similarity between claim and source text.
        
        Simple token overlap for now. Could use embeddings for better accuracy.
        
        Args:
            claim: Claim to verify
            source: Source text
            
        Returns:
            Similarity score (0-1)
        """
        claim_tokens = set(claim.lower().split())
        source_tokens = set(source.lower().split())
        
        if not claim_tokens:
            return 0.0
        
        overlap = len(claim_tokens & source_tokens)
        similarity = overlap / len(claim_tokens)
        
        return similarity
    
    def _build_feedback(self, scores: Dict[str, float], issues: List[str], is_valid: bool) -> str:
        """
        Build human-readable feedback message.
        
        Args:
            scores: Individual check scores
            issues: List of issues found
            is_valid: Overall validation result
            
        Returns:
            Feedback message
        """
        if is_valid:
            return "Response passed all validation checks."
        
        feedback_parts = ["Validation failed:"]
        
        for check, score in scores.items():
            if score < 70:
                feedback_parts.append(f"- {check.replace('_', ' ').title()}: {score:.1f}%")
        
        if issues:
            feedback_parts.append("\nIssues found:")
            for i, issue in enumerate(issues[:5], 1):  # Top 5 issues
                feedback_parts.append(f"{i}. {issue}")
        
        return "\n".join(feedback_parts)
    
    def _aggregate_scores(
        self,
        scores: Dict[str, float],
        weights: Dict[str, float] = None
    ) -> float:
        """
        Aggregate individual check scores.
        
        Args:
            scores: Individual check scores
            weights: Optional weights for each check
            
        Returns:
            Weighted average score
        """
        default_weights = {
            "factual_accuracy": 0.40,  # Highest weight - critical for hallucination prevention
            "numerical_accuracy": 0.30,  # High weight - financial data must be exact
            "citation_coverage": 0.20,
            "completeness": 0.10
        }
        weights = weights or default_weights
        
        total = sum(
            scores.get(check, 0) * weight
            for check, weight in weights.items()
        )
        
        return total
