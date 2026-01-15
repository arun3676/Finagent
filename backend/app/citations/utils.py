"""
Citation Utilities

Helper functions for enhanced citation generation:
- Context extraction (surrounding sentences)
- Highlight position calculation
- Preview text generation
- Source metadata formatting
"""

import re
from typing import Tuple, Dict, Any, Optional
from app.models import DocumentChunk, DocumentMetadata


def extract_context(
    full_text: str,
    target_text: str,
    context_sentences: int = 2
) -> Tuple[str, int, int]:
    """
    Extract surrounding context for a piece of text.

    Args:
        full_text: Full document text
        target_text: Text to find and extract context for
        context_sentences: Number of sentences to include before/after

    Returns:
        Tuple of (context_with_target, highlight_start, highlight_end)
        where positions are relative to the context string
    """
    # Find target text in full text
    target_pos = full_text.find(target_text)

    if target_pos == -1:
        # Fallback: try case-insensitive or partial match
        target_lower = target_text.lower()
        full_lower = full_text.lower()
        target_pos = full_lower.find(target_lower)

        if target_pos == -1:
            # Can't find it, return truncated version
            return full_text[:500], 0, min(len(target_text), 500)

    # Split into sentences
    sentences = split_into_sentences(full_text)

    # Find which sentence contains the target
    char_count = 0
    target_sentence_idx = -1

    for idx, sentence in enumerate(sentences):
        sentence_start = char_count
        sentence_end = char_count + len(sentence)

        if sentence_start <= target_pos < sentence_end:
            target_sentence_idx = idx
            break

        char_count = sentence_end

    if target_sentence_idx == -1:
        # Fallback
        context_start = max(0, target_pos - 200)
        context_end = min(len(full_text), target_pos + len(target_text) + 200)
        context = full_text[context_start:context_end]
        highlight_start = target_pos - context_start
        highlight_end = highlight_start + len(target_text)
        return context, highlight_start, highlight_end

    # Extract context sentences
    start_idx = max(0, target_sentence_idx - context_sentences)
    end_idx = min(len(sentences), target_sentence_idx + context_sentences + 1)

    context_sentences_list = sentences[start_idx:end_idx]
    context = " ".join(context_sentences_list)

    # Calculate highlight positions relative to context
    context_before_target = " ".join(sentences[start_idx:target_sentence_idx])
    if context_before_target:
        context_before_target += " "  # Add space separator

    # Find target within the context
    highlight_start = context.find(target_text)
    if highlight_start == -1:
        # Try to find it approximately
        highlight_start = len(context_before_target)

    highlight_end = highlight_start + len(target_text)

    return context, highlight_start, highlight_end


def split_into_sentences(text: str) -> list:
    """
    Split text into sentences, handling common abbreviations.

    Args:
        text: Text to split

    Returns:
        List of sentences
    """
    # Protect common abbreviations and decimals
    protected = text
    protected = re.sub(r'(\d)\.(\d)', r'\1<DECIMAL>\2', protected)
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Inc|Corp|Ltd|etc|vs|e\.g|i\.e)\.', r'\1<PERIOD>', protected)

    # Split by period followed by space and capital letter, or period at end
    sentences = re.split(r'\.(?:\s+(?=[A-Z])|$)', protected)

    # Restore protected characters and clean up
    sentences = [
        s.replace('<DECIMAL>', '.').replace('<PERIOD>', '.').strip()
        for s in sentences
        if s.strip()
    ]

    # Add back the periods (except for last sentence if it didn't end with period)
    result = []
    for i, sent in enumerate(sentences):
        if sent and not sent.endswith('.'):
            sent = sent + '.'
        result.append(sent)

    return result


def generate_preview_text(source_text: str, max_length: int = 50) -> str:
    """
    Generate a short preview of the source text.

    Args:
        source_text: Source text to preview
        max_length: Maximum length of preview

    Returns:
        Preview text with ellipsis if truncated
    """
    # Clean whitespace
    cleaned = " ".join(source_text.split())

    if len(cleaned) <= max_length:
        return cleaned

    # Truncate at word boundary
    truncated = cleaned[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length // 2:
        truncated = truncated[:last_space]

    return truncated + "..."


def find_highlight_positions(
    context: str,
    target_text: str
) -> Tuple[int, int]:
    """
    Find the start and end positions of target text within context.

    Args:
        context: Context text
        target_text: Text to highlight

    Returns:
        Tuple of (start_pos, end_pos)
    """
    start_pos = context.find(target_text)

    if start_pos == -1:
        # Try case-insensitive
        start_pos = context.lower().find(target_text.lower())

    if start_pos == -1:
        # Can't find exact match, return approximate position
        return 0, min(len(target_text), len(context))

    end_pos = start_pos + len(target_text)
    return start_pos, end_pos


def format_source_metadata(
    chunk: DocumentChunk
) -> Dict[str, Any]:
    """
    Format document chunk metadata for citation display.

    Args:
        chunk: Document chunk

    Returns:
        Formatted metadata dict
    """
    meta = chunk.metadata

    return {
        "ticker": meta.ticker,
        "company_name": meta.company_name,
        "document_type": meta.document_type.value if hasattr(meta.document_type, 'value') else str(meta.document_type),
        "filing_date": meta.filing_date.isoformat() if hasattr(meta.filing_date, 'isoformat') else str(meta.filing_date),
        "fiscal_year": meta.fiscal_year,
        "fiscal_quarter": meta.fiscal_quarter,
        "fiscal_period": meta.fiscal_period,
        "period_end_date": meta.period_end_date.isoformat() if hasattr(meta.period_end_date, 'isoformat') and meta.period_end_date else None,
        "section": chunk.section,
        "page_number": chunk.page_number,
        "source_url": meta.source_url,
        "accession_number": meta.accession_number,
    }


def determine_validation_method(
    claim: str,
    source_text: str,
    confidence: float
) -> str:
    """
    Determine the validation method used for this citation.

    Args:
        claim: Claim text
        source_text: Source text
        confidence: Confidence score

    Returns:
        Validation method: "exact_match", "semantic_similarity", or "llm_verified"
    """
    # Check for exact match
    if claim.lower() in source_text.lower():
        return "exact_match"

    # Check for high confidence (likely LLM verified)
    if confidence >= 0.9:
        return "llm_verified"

    # Default to semantic similarity
    return "semantic_similarity"
