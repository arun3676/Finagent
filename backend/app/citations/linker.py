"""
Citation Linker

Links extracted claims to source document chunks.
Uses semantic similarity and exact matching to find supporting evidence.

Linking strategies:
1. Exact match: Numbers/quotes appear verbatim in source
2. Semantic match: Claim meaning matches source content
3. Entity match: Same entities discussed in source

Usage:
    linker = CitationLinker(embedding_service)
    citations = await linker.link_claims(claims, documents)
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from app.models import Citation, RetrievedDocument
from app.citations.extractor import ExtractedClaim, ClaimType
from app.retrieval.embeddings import EmbeddingService
from app.citations.utils import (
    extract_context,
    generate_preview_text,
    format_source_metadata,
    determine_validation_method
)


@dataclass
class LinkScore:
    """Score for a claim-document link."""
    chunk_id: str
    exact_match_score: float
    semantic_score: float
    entity_overlap_score: float
    combined_score: float


class CitationLinker:
    """
    Links claims to source documents.
    
    Uses multiple matching strategies to find
    the best supporting evidence for each claim.
    """
    
    # Minimum score to consider a link valid
    MIN_LINK_SCORE = 0.5
    
    def __init__(
        self,
        embedding_service: EmbeddingService = None,
        use_semantic: bool = True
    ):
        """
        Initialize citation linker.
        
        Args:
            embedding_service: Service for semantic matching
            use_semantic: Enable semantic similarity matching
        """
        self.embedding_service = embedding_service
        self.use_semantic = use_semantic and embedding_service is not None
    
    async def link_claims(
        self,
        claims: List[ExtractedClaim],
        documents: List[RetrievedDocument]
    ) -> List[Citation]:
        """
        Link claims to source documents.
        
        Args:
            claims: Extracted claims to link
            documents: Source documents to search
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        for i, claim in enumerate(claims):
            best_link = await self._find_best_link(claim, documents)
            
            if best_link and best_link.combined_score >= self.MIN_LINK_SCORE:
                # Find the document for this chunk
                doc = next(
                    (d for d in documents if d.chunk.chunk_id == best_link.chunk_id),
                    None
                )
                
                if doc:
                    meta = doc.chunk.metadata

                    # Get source text
                    source_text = doc.chunk.content[:500]

                    # Extract context with highlighting
                    source_context, highlight_start, highlight_end = extract_context(
                        full_text=doc.chunk.content,
                        target_text=claim.text,
                        context_sentences=2
                    )

                    # Generate preview text
                    preview = generate_preview_text(source_text, max_length=50)

                    # Format full source metadata
                    source_meta = format_source_metadata(doc.chunk)

                    # Determine validation method
                    validation_method = determine_validation_method(
                        claim=claim.text,
                        source_text=doc.chunk.content,
                        confidence=best_link.combined_score
                    )

                    citation = Citation(
                        citation_id=f"cite_{i+1}",
                        citation_number=i+1,
                        claim=claim.text,
                        source_chunk_id=best_link.chunk_id,
                        source_document_id=doc.chunk.document_id,
                        source_text=source_text,
                        source_context=source_context,
                        highlight_start=highlight_start,
                        highlight_end=highlight_end,
                        source_metadata=source_meta,
                        confidence=best_link.combined_score,
                        validation_method=validation_method,
                        preview_text=preview,
                        # Legacy fields for backward compatibility
                        page_reference=f"{meta.document_type.value}, {doc.chunk.section}",
                        source_url=meta.source_url,
                        metadata={
                            "ticker": meta.ticker,
                            "filing_type": meta.document_type.value,
                            "section": doc.chunk.section,
                            "source_url": meta.source_url,
                        }
                    )
                    citations.append(citation)
        
        return citations
    
    async def _find_best_link(
        self,
        claim: ExtractedClaim,
        documents: List[RetrievedDocument]
    ) -> Optional[LinkScore]:
        """
        Find the best document link for a claim.
        
        Args:
            claim: Claim to link
            documents: Documents to search
            
        Returns:
            Best LinkScore or None
        """
        scores = []
        
        for doc in documents:
            score = await self._score_link(claim, doc)
            scores.append(score)
        
        if not scores:
            return None
        
        # Return highest scoring link
        return max(scores, key=lambda x: x.combined_score)
    
    async def _score_link(
        self,
        claim: ExtractedClaim,
        document: RetrievedDocument
    ) -> LinkScore:
        """
        Score the link between a claim and document.
        
        Args:
            claim: Claim to link
            document: Document to check
            
        Returns:
            LinkScore object
        """
        content = document.chunk.content.lower()
        claim_text = claim.text.lower()
        
        # Exact match scoring
        exact_score = self._score_exact_match(claim, content)
        
        # Semantic scoring
        semantic_score = 0.0
        if self.use_semantic:
            semantic_score = await self._score_semantic(claim.text, document.chunk.content)
        
        # Entity overlap scoring
        entity_score = self._score_entity_overlap(claim, content)
        
        # Combine scores (weighted average)
        combined = (
            0.4 * exact_score +
            0.4 * semantic_score +
            0.2 * entity_score
        )
        
        return LinkScore(
            chunk_id=document.chunk.chunk_id,
            exact_match_score=exact_score,
            semantic_score=semantic_score,
            entity_overlap_score=entity_score,
            combined_score=combined
        )
    
    def _score_exact_match(
        self,
        claim: ExtractedClaim,
        content: str
    ) -> float:
        """
        Score based on exact matches of numbers/entities.
        
        Args:
            claim: Claim with extracted numbers
            content: Document content (lowercase)
            
        Returns:
            Score from 0 to 1
        """
        if not claim.numbers:
            return 0.0
        
        matches = 0
        for number in claim.numbers:
            # Normalize number format
            normalized = number.lower().replace(',', '').replace('$', '')
            if normalized in content.replace(',', ''):
                matches += 1
        
        return matches / len(claim.numbers) if claim.numbers else 0.0
    
    async def _score_semantic(
        self,
        claim_text: str,
        doc_content: str
    ) -> float:
        """
        Score based on semantic similarity.
        
        Args:
            claim_text: Claim text
            doc_content: Document content
            
        Returns:
            Cosine similarity score
        """
        if not self.embedding_service:
            return 0.0
        
        try:
            # Get embeddings
            claim_emb = await self.embedding_service.embed_text(claim_text)
            doc_emb = await self.embedding_service.embed_text(doc_content[:1000])
            
            # Calculate cosine similarity
            return self._cosine_similarity(claim_emb, doc_emb)
        except Exception:
            return 0.0
    
    def _score_entity_overlap(
        self,
        claim: ExtractedClaim,
        content: str
    ) -> float:
        """
        Score based on entity overlap.
        
        Args:
            claim: Claim with extracted entities
            content: Document content (lowercase)
            
        Returns:
            Score from 0 to 1
        """
        if not claim.entities:
            return 0.0
        
        matches = sum(1 for e in claim.entities if e.lower() in content)
        return matches / len(claim.entities)
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0 to 1)
        """
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def validate_citation(
        self,
        citation: Citation,
        document: RetrievedDocument
    ) -> bool:
        """
        Validate that a citation is properly supported.
        
        Args:
            citation: Citation to validate
            document: Source document
            
        Returns:
            True if citation is valid
        """
        # Check that source text is from the document
        if citation.source_text not in document.chunk.content:
            return False
        
        # Check confidence threshold
        if citation.confidence < self.MIN_LINK_SCORE:
            return False
        
        return True
