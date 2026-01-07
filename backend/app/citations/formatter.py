"""
Citation Formatter

Formats citations for various output formats.
Supports inline citations, footnotes, and bibliography styles.

Output formats:
- Inline: [1], [2], etc. in text
- Footnote: Superscript numbers with footnotes
- Bibliography: Full source list at end
- Academic: Author-date style

Usage:
    formatter = CitationFormatter()
    formatted = formatter.format_response(response, citations, style="inline")
"""

from typing import List, Dict, Any, Optional
from enum import Enum

from app.models import Citation, DocumentMetadata


class CitationStyle(Enum):
    """Supported citation styles."""
    INLINE = "inline"
    FOOTNOTE = "footnote"
    BIBLIOGRAPHY = "bibliography"
    ACADEMIC = "academic"


class CitationFormatter:
    """
    Formats citations for output.
    
    Handles various citation styles and generates
    properly formatted source lists.
    """
    
    def __init__(self, style: CitationStyle = CitationStyle.INLINE):
        """
        Initialize formatter with default style.
        
        Args:
            style: Default citation style
        """
        self.default_style = style
    
    def format_response(
        self,
        response: str,
        citations: List[Citation],
        sources: List[DocumentMetadata],
        style: CitationStyle = None
    ) -> Dict[str, str]:
        """
        Format a response with citations.
        
        Args:
            response: Response text (may have [N] markers)
            citations: List of citations
            sources: Source document metadata
            style: Citation style to use
            
        Returns:
            Dictionary with formatted response and sources
        """
        style = style or self.default_style
        
        if style == CitationStyle.INLINE:
            return self._format_inline(response, citations, sources)
        elif style == CitationStyle.FOOTNOTE:
            return self._format_footnote(response, citations, sources)
        elif style == CitationStyle.BIBLIOGRAPHY:
            return self._format_bibliography(response, citations, sources)
        elif style == CitationStyle.ACADEMIC:
            return self._format_academic(response, citations, sources)
        else:
            return self._format_inline(response, citations, sources)
    
    def _format_inline(
        self,
        response: str,
        citations: List[Citation],
        sources: List[DocumentMetadata]
    ) -> Dict[str, str]:
        """
        Format with inline [N] citations.
        
        Args:
            response: Response text
            citations: Citations list
            sources: Source metadata
            
        Returns:
            Formatted output
        """
        # Response already has [N] markers, just format sources
        source_list = self._format_source_list(sources)
        
        return {
            "response": response,
            "sources": source_list,
            "citation_count": len(citations)
        }
    
    def _format_footnote(
        self,
        response: str,
        citations: List[Citation],
        sources: List[DocumentMetadata]
    ) -> Dict[str, str]:
        """
        Format with footnote-style citations.
        
        Args:
            response: Response text
            citations: Citations list
            sources: Source metadata
            
        Returns:
            Formatted output with footnotes
        """
        import re
        
        # Convert [N] to superscript style
        formatted_response = re.sub(
            r'\[(\d+)\]',
            r'<sup>\1</sup>',
            response
        )
        
        # Create footnotes section
        footnotes = []
        for i, citation in enumerate(citations, 1):
            footnotes.append(f"{i}. {citation.page_reference}: \"{citation.source_text[:100]}...\"")
        
        return {
            "response": formatted_response,
            "footnotes": "\n".join(footnotes),
            "sources": self._format_source_list(sources)
        }
    
    def _format_bibliography(
        self,
        response: str,
        citations: List[Citation],
        sources: List[DocumentMetadata]
    ) -> Dict[str, str]:
        """
        Format with bibliography at end.
        
        Args:
            response: Response text
            citations: Citations list
            sources: Source metadata
            
        Returns:
            Formatted output with bibliography
        """
        # Create detailed bibliography
        bibliography = []
        for i, source in enumerate(sources, 1):
            entry = (
                f"[{i}] {source.company_name} ({source.ticker}). "
                f"{source.document_type.value}. "
                f"Filed {source.filing_date.strftime('%B %d, %Y')}. "
                f"Retrieved from {source.source_url}"
            )
            bibliography.append(entry)
        
        return {
            "response": response,
            "bibliography": "\n\n".join(bibliography)
        }
    
    def _format_academic(
        self,
        response: str,
        citations: List[Citation],
        sources: List[DocumentMetadata]
    ) -> Dict[str, str]:
        """
        Format with academic author-date style.
        
        Args:
            response: Response text
            citations: Citations list
            sources: Source metadata
            
        Returns:
            Formatted output in academic style
        """
        import re
        
        # Create mapping of citation numbers to academic format
        citation_map = {}
        for i, source in enumerate(sources, 1):
            year = source.filing_date.year
            academic_cite = f"({source.ticker}, {year})"
            citation_map[str(i)] = academic_cite
        
        # Replace [N] with academic citations
        def replace_citation(match):
            num = match.group(1)
            return citation_map.get(num, match.group(0))
        
        formatted_response = re.sub(r'\[(\d+)\]', replace_citation, response)
        
        # Create references section
        references = []
        for source in sources:
            ref = (
                f"{source.ticker}. ({source.filing_date.year}). "
                f"{source.document_type.value}. "
                f"{source.company_name}. "
                f"U.S. Securities and Exchange Commission."
            )
            references.append(ref)
        
        return {
            "response": formatted_response,
            "references": "\n\n".join(sorted(set(references)))
        }
    
    def _format_source_list(
        self,
        sources: List[DocumentMetadata]
    ) -> str:
        """
        Format a simple source list.
        
        Args:
            sources: Source metadata list
            
        Returns:
            Formatted source list string
        """
        lines = []
        for i, source in enumerate(sources, 1):
            line = (
                f"[{i}] {source.company_name} ({source.ticker}) - "
                f"{source.document_type.value}, "
                f"{source.filing_date.strftime('%Y-%m-%d')}"
            )
            lines.append(line)
        
        return "\n".join(lines)
    
    def format_single_citation(
        self,
        citation: Citation,
        style: CitationStyle = None
    ) -> str:
        """
        Format a single citation.
        
        Args:
            citation: Citation to format
            style: Citation style
            
        Returns:
            Formatted citation string
        """
        style = style or self.default_style
        
        if style == CitationStyle.INLINE:
            return f"[{citation.citation_id.split('_')[1]}]"
        elif style == CitationStyle.FOOTNOTE:
            return f"<sup>{citation.citation_id.split('_')[1]}</sup>"
        else:
            return f"[{citation.citation_id}]"
    
    def generate_citation_summary(
        self,
        citations: List[Citation]
    ) -> Dict[str, Any]:
        """
        Generate a summary of citations.
        
        Args:
            citations: List of citations
            
        Returns:
            Summary statistics
        """
        if not citations:
            return {"count": 0}
        
        return {
            "count": len(citations),
            "avg_confidence": sum(c.confidence for c in citations) / len(citations),
            "high_confidence": len([c for c in citations if c.confidence >= 0.8]),
            "low_confidence": len([c for c in citations if c.confidence < 0.5])
        }
