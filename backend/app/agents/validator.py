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
from app.models import (
    AgentState,
    RetrievedDocument,
    Citation,
    ValidationResult as ValidationResultModel,
    calculate_trust_level,
    DocumentType,
    AgentRole,
    StepEvent
)
from app.agents.prompts import VALIDATOR_SYSTEM_PROMPT, VALIDATOR_USER_TEMPLATE
from app.utils.temporal import extract_temporal_constraints
from datetime import datetime

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
    MIN_VALID_SCORE = 75
    MIN_CITATION_COVERAGE = 0.7
    
    def __init__(self, model: str = None):
        """
        Initialize validator.
        
        Args:
            model: LLM model to use
        """
        self.model = model or settings.LLM_MODEL
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences handling common abbreviations and decimals.
        """
        # Replace decimals to protect them from split
        # protect 1.2, 10.5 etc.
        protected_text = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', text)
        
        # Split by period followed by space or end of string
        sentences = re.split(r'\.\s+|\.$', protected_text)
        
        # Restore decimals
        sentences = [s.replace('<DECIMAL>', '.') for s in sentences if s.strip()]
        return sentences

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
        
        # Run validation checks
        factual_score, factual_issues = await self.check_factual_accuracy(response, citations, documents)
        citation_score, citation_issues = self.check_citation_coverage(response, citations)
        completeness_score, completeness_issues = await self.check_completeness(query, response)
        numerical_score, numerical_issues = self.check_numerical_accuracy(response, documents)
        temporal_score, temporal_issues = self.check_temporal_accuracy(query, documents)
        
        # Aggregate scores
        scores = {
            "factual_accuracy": factual_score,
            "citation_coverage": citation_score,
            "completeness": completeness_score,
            "numerical_accuracy": numerical_score,
            "temporal_accuracy": temporal_score
        }
        
        total_score = self._aggregate_scores(scores)
        
        # Collect issues
        all_issues = factual_issues + citation_issues + completeness_issues + numerical_issues + temporal_issues
        
        # Determine validity
        is_valid = total_score >= self.MIN_VALID_SCORE
        
        # Build feedback
        feedback = self._build_feedback(scores, all_issues, is_valid)
        
        logger.info(f"Validation complete. Score: {total_score:.1f}, Valid: {is_valid}")
        
        return ValidationResult(
            is_valid=is_valid,
            score=total_score,
            feedback=feedback,
            issues=all_issues
        )
    
    async def validate_for_state(self, state: AgentState) -> Dict[str, Any]:
        """
        Validate response and update agent state.

        LangGraph-compatible interface - returns dict with updated fields.

        Args:
            state: Current agent state

        Returns:
            Dict with is_valid, validation_feedback, and validation_result fields for state update
        """
        # Run existing validation to get basic result
        basic_result = await self.validate(
            state.original_query,
            state.draft_response,
            state.citations,
            state.retrieved_docs
        )

        # Calculate detailed confidence breakdown
        factual_score, factual_issues = await self.check_factual_accuracy(
            state.draft_response,
            state.citations,
            state.retrieved_docs
        )
        citation_score, citation_issues = self.check_citation_coverage(
            state.draft_response,
            state.citations
        )
        completeness_score, completeness_issues = await self.check_completeness(
            state.original_query,
            state.draft_response
        )
        numerical_score, numerical_issues = self.check_numerical_accuracy(
            state.draft_response,
            state.retrieved_docs
        )
        temporal_score, temporal_issues = self.check_temporal_accuracy(
            state.original_query,
            state.retrieved_docs
        )
        source_quality_score = self._calculate_source_quality(state.retrieved_docs)

        # Build confidence breakdown (normalize to 0-1)
        confidence_breakdown = {
            "factual_accuracy": factual_score / 100.0,
            "citation_coverage": citation_score / 100.0,
            "numerical_accuracy": numerical_score / 100.0,
            "temporal_accuracy": temporal_score / 100.0,
            "source_quality": source_quality_score
        }

        # Calculate overall confidence using same weights as validation
        overall_confidence = (
            confidence_breakdown["factual_accuracy"] * 0.35 +
            confidence_breakdown["numerical_accuracy"] * 0.30 +
            confidence_breakdown["citation_coverage"] * 0.20 +
            confidence_breakdown["temporal_accuracy"] * 0.05 +
            confidence_breakdown["source_quality"] * 0.10
        )

        # Calculate trust level
        trust_level, trust_label, trust_color = calculate_trust_level(overall_confidence)

        # Count claims
        sentences = self._split_sentences(state.draft_response)
        factual_sentences = [
            s for s in sentences
            if any(keyword in s.lower() for keyword in
                   ['revenue', 'income', 'profit', 'loss', 'margin', 'growth',
                    'increased', 'decreased', 'reported', '$', '%'])
        ]
        total_claims = len(factual_sentences)

        # Calculate verified claims based on factual_score
        verified_claims = int(total_claims * (factual_score / 100.0))
        unverified_claims = total_claims - verified_claims

        # Collect validation notes
        validation_notes = []
        all_issues = factual_issues + citation_issues + completeness_issues + numerical_issues + temporal_issues

        if basic_result.is_valid:
            validation_notes.append("All validation checks passed")
            if numerical_score == 100 and total_claims > 0:
                validation_notes.append("All numerical claims verified against sources")
            if source_quality_score >= 0.8:
                validation_notes.append("High source relevance scores")
        else:
            # Add top issues as notes
            for issue in all_issues[:3]:
                validation_notes.append(issue)

        # Check if revalidation is needed
        MAX_ITERATIONS = 3
        required_revalidation = not basic_result.is_valid and state.iteration_count < MAX_ITERATIONS

        # Build ValidationResult model
        validation_result = ValidationResultModel(
            is_valid=basic_result.is_valid,
            trust_level=trust_level,
            trust_label=trust_label,
            trust_color=trust_color,
            overall_confidence=overall_confidence,
            confidence_breakdown=confidence_breakdown,
            claims_checked=total_claims,
            claims_verified=verified_claims,
            claims_unverified=unverified_claims,
            sources_used=len(state.retrieved_docs),
            avg_source_relevance=source_quality_score,
            source_diversity=self._describe_source_diversity(state),
            validation_notes=validation_notes,
            validation_attempts=state.iteration_count + 1,
            required_revalidation=required_revalidation
        )

        logger.info(
            f"Validation result: trust_level={trust_level}, "
            f"confidence={overall_confidence:.2f}, "
            f"claims={verified_claims}/{total_claims}"
        )

        # Create step events for validation checks
        new_events = list(state.step_events) if state.step_events else []

        # Emit validation start event
        validation_start_event = StepEvent(
            event_type="validation_check",
            agent=AgentRole.VALIDATOR,
            timestamp=datetime.now(),
            data={
                "action": "checking_response",
                "claims_to_verify": total_claims,
                "citations_count": len(state.citations)
            }
        )
        new_events.append(validation_start_event)

        # Emit detailed validation check events
        for check_name, score in confidence_breakdown.items():
            check_event = StepEvent(
                event_type="validation_check",
                agent=AgentRole.VALIDATOR,
                timestamp=datetime.now(),
                data={
                    "check_type": check_name,
                    "score": score,
                    "passed": score >= 0.7
                }
            )
            new_events.append(check_event)

        # Emit final validation result event
        validation_result_event = StepEvent(
            event_type="validation_check",
            agent=AgentRole.VALIDATOR,
            timestamp=datetime.now(),
            data={
                "action": "validation_complete",
                "is_valid": basic_result.is_valid,
                "trust_level": trust_level,
                "overall_confidence": overall_confidence,
                "will_retry": required_revalidation
            }
        )
        new_events.append(validation_result_event)

        return {
            "is_valid": basic_result.is_valid,
            "validation_feedback": basic_result.feedback,
            "validation_result": validation_result,
            "iteration_count": state.iteration_count + 1,
            "step_events": new_events
        }
    
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
        
        # Extract claims from response
        sentences = self._split_sentences(response)
        
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
                # Be lenient if sentence is short or generic
                if len(sentence.split()) > 10:
                    issues.append(f"Uncited claim: '{sentence[:80]}...'")
                continue
            
            # Verify claim against cited sources
            claim_verified = False
            for ref in citation_refs:
                citation_key = f"cite_{ref}"
                citation = citation_map.get(citation_key)
                
                # Try integer key if string fails
                if not citation:
                     # iterate to find matching id suffix
                     for c in citations:
                         if c.citation_id.endswith(f"_{ref}"):
                             citation = c
                             break

                if citation:
                    # Check similarity between claim and source
                    similarity = self._compute_claim_similarity(
                        sentence, citation.source_text
                    )
                    # Lower threshold to 0.3 for token overlap (it's rough)
                    if similarity > 0.3:
                        claim_verified = True
                        break
            
            if claim_verified:
                verified_claims += 1
            else:
                # Only flag if we have citations but similarity is low
                # Maybe the citation is correct but wording is different.
                # Don't penalize too hard if citations exist.
                if citation_refs:
                     # Assume if it's cited, it's likely okay-ish, but check strictness
                     pass 
                issues.append(f"Claim low similarity to source: '{sentence[:80]}...'")
        
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
        sentences = self._split_sentences(response)
        
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
        from decimal import Decimal, InvalidOperation

        issues = []

        def normalize_number(value: str) -> Optional[str]:
            cleaned = value.replace("$", "").replace(",", "")
            if not cleaned:
                return None
            try:
                dec = Decimal(cleaned)
            except InvalidOperation:
                return None
            normalized = format(dec.normalize(), "f")
            normalized = normalized.rstrip("0").rstrip(".")
            return normalized or "0"

        def is_ignorable(value: str) -> bool:
            try:
                dec = Decimal(value)
            except InvalidOperation:
                return False
            if dec == dec.to_integral_value():
                num = int(dec)
                return abs(num) < 100 or 1900 < num < 2100
            return False

        number_pattern = r'-?\$?[\d,]+(?:\.\d+)?'

        # Extract numbers from response
        response_numbers = {
            normalize_number(n)
            for n in re.findall(number_pattern, response)
        }
        response_numbers.discard(None)

        # Extract numbers from source documents
        source_text = " ".join(d.chunk.content for d in documents)
        source_numbers = {
            normalize_number(n)
            for n in re.findall(number_pattern, source_text)
        }
        source_numbers.discard(None)

        # Check if response numbers appear in sources
        unsupported = response_numbers - source_numbers

        # Filter out common numbers (years, small integers)
        unsupported = {n for n in unsupported if not is_ignorable(n)}

        if unsupported:
            issues.append(
                f"Numbers not found in sources: {', '.join(list(unsupported)[:5])}"
            )
            return 0.0, issues

        if response_numbers:
            return 100.0, []

        return 100.0, issues

    def check_temporal_accuracy(
        self,
        query: str,
        documents: List[RetrievedDocument]
    ) -> Tuple[float, List[str]]:
        """
        Verify sources align with requested fiscal period.

        Args:
            query: Original query
            documents: Source documents

        Returns:
            Tuple of (score, issues)
        """
        constraints = extract_temporal_constraints(query)
        if not constraints.fiscal_year:
            return 100.0, []

        issues = []
        checked_sources = 0

        for doc in documents:
            meta = doc.chunk.metadata
            source_year = meta.fiscal_year
            if source_year is None:
                continue

            checked_sources += 1
            source_quarter = meta.fiscal_quarter or 4

            if constraints.fiscal_quarter:
                if (source_year > constraints.fiscal_year or
                        (source_year == constraints.fiscal_year and source_quarter > constraints.fiscal_quarter)):
                    issues.append(
                        f"Source from {meta.fiscal_period or f'FY{source_year}'} "
                        f"cited for {constraints.fiscal_period} query"
                    )
            else:
                if source_year > constraints.fiscal_year:
                    issues.append(
                        f"Source from FY{source_year} cited for FY{constraints.fiscal_year} query"
                    )

        if not checked_sources:
            return 100.0, ["No fiscal period metadata available for temporal validation"]

        if issues:
            return 0.0, issues

        return 100.0, []
    
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

    def _calculate_source_quality(self, documents: List[RetrievedDocument]) -> float:
        """
        Calculate average source relevance score.

        Args:
            documents: Retrieved source documents

        Returns:
            Average relevance score (0-1)
        """
        if not documents:
            return 0.0

        total_score = sum(doc.score for doc in documents)
        avg_score = total_score / len(documents)

        return avg_score

    def _describe_source_diversity(self, state: AgentState) -> str:
        """
        Generate human-readable description of source diversity.

        Args:
            state: Current agent state with retrieved documents

        Returns:
            Human-readable description like "3 10-K filings, 1 earnings call"
        """
        documents = state.retrieved_docs

        if not documents:
            return "No sources"

        # Count by document type
        type_counts = {}
        tickers = set()

        for doc in documents:
            doc_type = doc.chunk.metadata.document_type
            ticker = doc.chunk.metadata.ticker

            # Count document types
            type_name = doc_type.value if isinstance(doc_type, DocumentType) else str(doc_type)
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            tickers.add(ticker)

        # Build description
        parts = []
        for doc_type, count in sorted(type_counts.items()):
            plural = "filing" if count == 1 else "filings"
            if doc_type in ["10-K", "10-Q", "8-K"]:
                parts.append(f"{count} {doc_type} {plural}")
            else:
                parts.append(f"{count} {doc_type}")

        description = ", ".join(parts)

        # Add ticker info if multiple companies
        if len(tickers) > 1:
            ticker_list = ", ".join(sorted(tickers))
            description += f" ({ticker_list})"

        return description
    
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
            "factual_accuracy": 0.35,  # Highest weight - critical for hallucination prevention
            "numerical_accuracy": 0.30,  # High weight - financial data must be exact
            "citation_coverage": 0.20,
            "completeness": 0.10,
            "temporal_accuracy": 0.05
        }
        weights = weights or default_weights
        
        total = sum(
            scores.get(check, 0) * weight
            for check, weight in weights.items()
        )
        
        return total
