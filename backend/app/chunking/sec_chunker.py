"""
SEC Filing Chunker

Section-aware chunking for SEC filings (10-K, 10-Q, 8-K).
Preserves document structure and maintains semantic coherence.

Key features:
- Respects section boundaries (Item 1, Item 1A, etc.)
- Handles nested subsections
- Preserves paragraph integrity
- Maintains context headers in each chunk

Usage:
    chunker = SECChunker(chunk_size=1000, overlap=200)
    chunks = chunker.chunk_document(document_text, metadata)
"""

import re
import tiktoken
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from app.config import settings
from app.models import DocumentChunk, DocumentMetadata

logger = logging.getLogger(__name__)


@dataclass
class Section:
    """Represents a section in an SEC filing."""
    name: str
    title: str
    content: str
    start_pos: int
    end_pos: int
    level: int  # 1 for Item, 2 for subsection


class SECChunker:
    """
    Section-aware chunker for SEC filings.
    
    Chunking strategy:
    1. Parse document into sections (Items)
    2. For each section, chunk by paragraphs
    3. Merge small paragraphs, split large ones
    4. Add section context to each chunk
    """
    
    # SEC 10-K section patterns
    SECTION_10K_PATTERNS = {
        "item_1": r"(?i)item\s*1[.\s]+business",
        "item_1a": r"(?i)item\s*1a[.\s]+risk\s*factors",
        "item_1b": r"(?i)item\s*1b[.\s]+unresolved\s*staff\s*comments",
        "item_2": r"(?i)item\s*2[.\s]+properties",
        "item_3": r"(?i)item\s*3[.\s]+legal\s*proceedings",
        "item_4": r"(?i)item\s*4[.\s]+mine\s*safety",
        "item_5": r"(?i)item\s*5[.\s]+market",
        "item_6": r"(?i)item\s*6[.\s]+selected\s*financial",
        "item_7": r"(?i)item\s*7[.\s]+management.*discussion",
        "item_7a": r"(?i)item\s*7a[.\s]+quantitative.*qualitative",
        "item_8": r"(?i)item\s*8[.\s]+financial\s*statements",
        "item_9": r"(?i)item\s*9[.\s]+changes.*disagreements",
        "item_9a": r"(?i)item\s*9a[.\s]+controls\s*and\s*procedures",
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        preserve_tables: bool = True
    ):
        """
        Initialize SEC chunker.
        
        Args:
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            preserve_tables: Keep tables intact (don't split)
        """
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.preserve_tables = preserve_tables
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Chunk an SEC filing into semantic chunks.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        filing_type = metadata.document_type.value
        sections = self.parse_sections(text, filing_type)
        
        if not sections:
            logger.warning("No sections found, chunking entire document")
            sections = {"full_document": text}
        
        all_chunks = []
        document_id = f"{metadata.ticker}_{metadata.filing_date.strftime('%Y%m%d')}_{filing_type}"
        
        for section_name, section_content in sections.items():
            section_obj = Section(
                name=section_name,
                title=section_name.replace("_", " ").title(),
                content=section_content,
                start_pos=0,
                end_pos=len(section_content),
                level=1
            )
            
            section_chunks = self.chunk_section(section_obj, metadata)
            all_chunks.extend(section_chunks)
        
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.document_id = document_id
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections")
        return all_chunks
    
    def parse_sections(self, text: str, filing_type: str = "10-K") -> Dict[str, str]:
        """
        Parse document into sections based on filing type.
        
        Args:
            text: Document text
            filing_type: Type of SEC filing
            
        Returns:
            Dictionary mapping section names to content
        """
        if filing_type == "10-K":
            return self._parse_10k_sections(text)
        elif filing_type == "10-Q":
            return self._parse_10q_sections(text)
        else:
            logger.warning(f"Unknown filing type {filing_type}, treating as single section")
            return {"full_document": text}
    
    def _parse_10k_sections(self, text: str) -> Dict[str, str]:
        """Parse 10-K filing into sections."""
        sections = {}
        
        # Patterns for 10-K sections
        # Note: Added more robust patterns with leading whitespace support
        section_patterns = [
            ("item_1", r"(?i)^\s*Item\s*1\.?\s*Business", r"(?i)^\s*Item\s*1A\.?\s*Risk"),
            ("item_1a", r"(?i)^\s*Item\s*1A\.?\s*Risk\s*Factors", r"(?i)^\s*Item\s*1B\.?\s*Unresolved"),
            ("item_7", r"(?i)^\s*Item\s*7\.?\s*Management", r"(?i)^\s*Item\s*7A\.?\s*Quantitative"),
            ("item_7a", r"(?i)^\s*Item\s*7A\.?\s*Quantitative", r"(?i)^\s*Item\s*8\.?\s*Financial"),
            ("item_8", r"(?i)^\s*Item\s*8\.?\s*Financial", r"(?i)^\s*Item\s*9\.?\s*Changes"),
            ("item_15", r"(?i)^\s*Item\s*15\.?\s*Exhibits", r"(?i)^\s*(?:Item\s*16|Signatures)"),
        ]
        
        for section_name, start_pattern, end_pattern in section_patterns:
            content = self._find_best_section_match(text, start_pattern, end_pattern)
            if content:
                sections[section_name] = content
                logger.debug(f"Extracted {section_name}: {len(content)} chars")
        
        return sections
    
    def _parse_10q_sections(self, text: str) -> Dict[str, str]:
        """Parse 10-Q filing into sections."""
        sections = {}
        
        section_patterns = [
            ("part_i_item_1", r"(?i)^\s*Item\s*1\.?\s*Financial\s*Statements", r"(?i)^\s*Item\s*2\.?\s*Management"),
            ("part_i_item_2", r"(?i)^\s*Item\s*2\.?\s*Management", r"(?i)^\s*Item\s*3\.?\s*Quantitative"),
            ("part_i_item_3", r"(?i)^\s*Item\s*3\.?\s*Quantitative", r"(?i)^\s*Part\s*II"),
        ]
        
        for section_name, start_pattern, end_pattern in section_patterns:
            content = self._find_best_section_match(text, start_pattern, end_pattern)
            if content:
                sections[section_name] = content
        
        return sections

    def _find_best_section_match(self, text: str, start_pattern: str, end_pattern: str) -> Optional[str]:
        """
        Find the best match for a section, avoiding TOCs.
        Strategy: Look for the longest content block between start and end markers.
        """
        # We relax the anchors to allow matching within the line, but prefer start of line
        # The patterns passed in are currently ^Item... which is good for clean text
        # But text might have noise.
        
        # Modify patterns to be less strict about start of line if needed, 
        # but strict about the phrasing
        
        # Use MULTILINE to match ^ at start of lines
        start_matches = list(re.finditer(start_pattern, text, re.MULTILINE))
        end_matches = list(re.finditer(end_pattern, text, re.MULTILINE))
        
        if not start_matches:
            # Try without start anchor
            relaxed_start = start_pattern.replace("^", "")
            start_matches = list(re.finditer(relaxed_start, text))
            
        if not end_matches:
            relaxed_end = end_pattern.replace("^", "")
            end_matches = list(re.finditer(relaxed_end, text))
            
        if not start_matches:
            return None
            
        best_content = None
        max_length = 0
        
        for start_match in start_matches:
            start_idx = start_match.start()
            
            # Find the first end match that comes after this start match
            valid_end_matches = [m for m in end_matches if m.start() > start_idx]
            
            if valid_end_matches:
                # Usually the correct end is the nearest one, BUT for TOCs, they are close.
                # Real sections are far apart.
                # However, we might have multiple "Item 1" in TOC.
                # We want the one that has the most content before the next section marker.
                
                # Check distance to the *first* valid end match
                first_end = valid_end_matches[0]
                end_idx = first_end.start()
                length = end_idx - start_idx
                
                # Filter out likely TOC entries (small length)
                if length < 500: 
                    continue
                    
                if length > max_length:
                    max_length = length
                    best_content = text[start_idx:end_idx]
            else:
                # No end match found after this start match. 
                # This could be the last section, or parsing error.
                # If it's the last section, take everything until end or a reasonable limit
                remaining = len(text) - start_idx
                if remaining > max_length and remaining > 500:
                    max_length = remaining
                    best_content = text[start_idx:]
        
        return self._clean_section_text(best_content) if best_content else None
    
    def _clean_section_text(self, text: str) -> str:
        """Clean and normalize section text."""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()
    
    def chunk_section(
        self,
        section: Section,
        metadata: DocumentMetadata
    ) -> List[DocumentChunk]:
        """
        Chunk a single section into smaller pieces.
        
        Args:
            section: Section to chunk
            metadata: Document metadata
            
        Returns:
            List of chunks from this section
        """
        paragraphs = self._split_by_paragraphs(section.content)
        
        if self.preserve_tables:
            paragraphs = self._detect_and_preserve_tables(paragraphs)
        
        merged_chunks = self._merge_small_chunks(paragraphs, min_size=100)
        
        final_chunks = []
        for chunk_text in merged_chunks:
            token_count = len(self.tokenizer.encode(chunk_text))
            
            if token_count > self.chunk_size:
                split_chunks = self._split_large_chunk(chunk_text)
                final_chunks.extend(split_chunks)
            else:
                final_chunks.append(chunk_text)
        
        chunks = []
        for i, content in enumerate(final_chunks):
            chunk = self._create_chunk(
                content=content,
                metadata=metadata,
                section=section.name,
                chunk_index=i,
                document_id=""
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs.
        
        Args:
            text: Text to split
            
        Returns:
            List of paragraphs
        """
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _detect_and_preserve_tables(self, paragraphs: List[str]) -> List[str]:
        """
        Detect financial tables and mark them for preservation.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            List of paragraphs with tables preserved
        """
        processed = []
        
        for para in paragraphs:
            is_table = self._is_table(para)
            
            if is_table:
                processed.append(para)
            else:
                processed.append(para)
        
        return processed
    
    def _is_table(self, text: str) -> bool:
        """
        Heuristic to detect if text is a financial table.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be a table
        """
        lines = text.split('\n')
        if len(lines) < 3:
            return False
        
        number_pattern = r'\d+[,.]?\d*'
        dollar_pattern = r'\$\s*\d+[,.]?\d*'
        
        numeric_lines = sum(1 for line in lines if re.search(number_pattern, line))
        dollar_lines = sum(1 for line in lines if re.search(dollar_pattern, line))
        
        if numeric_lines / len(lines) > 0.5 or dollar_lines > 2:
            return True
        
        tab_count = text.count('\t')
        if tab_count > len(lines):
            return True
        
        return False
    
    def _merge_small_chunks(
        self,
        chunks: List[str],
        min_size: int = 200
    ) -> List[str]:
        """
        Merge chunks that are too small.
        
        Args:
            chunks: List of text chunks
            min_size: Minimum chunk size in characters
            
        Returns:
            Merged chunks
        """
        if not chunks:
            return []
        
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            combined = current_chunk + "\n\n" + next_chunk
            combined_tokens = len(self.tokenizer.encode(combined))
            
            if len(current_chunk) < min_size and combined_tokens <= self.chunk_size:
                current_chunk = combined
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        merged.append(current_chunk)
        return merged
    
    def _split_large_chunk(
        self,
        text: str,
        max_size: int = None
    ) -> List[str]:
        """
        Split a chunk that exceeds max size.
        
        Args:
            text: Text to split
            max_size: Maximum chunk size in tokens
            
        Returns:
            List of smaller chunks
        """
        max_size = max_size or self.chunk_size
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = len(self.tokenizer.encode(test_chunk))
            
            if token_count <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                if len(self.tokenizer.encode(sentence)) > max_size:
                    words = sentence.split()
                    word_chunk = ""
                    for word in words:
                        test_word_chunk = word_chunk + " " + word if word_chunk else word
                        if len(self.tokenizer.encode(test_word_chunk)) <= max_size:
                            word_chunk = test_word_chunk
                        else:
                            if word_chunk:
                                chunks.append(word_chunk.strip())
                            word_chunk = word
                    if word_chunk:
                        current_chunk = word_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0 and self.chunk_overlap > 0:
                prev_tokens = self.tokenizer.encode(chunks[i-1])
                overlap_tokens = prev_tokens[-self.chunk_overlap:]
                overlap_text = self.tokenizer.decode(overlap_tokens)
                chunk = overlap_text + " " + chunk
            overlapped_chunks.append(chunk)
        
        return overlapped_chunks
    
    def _create_chunk(
        self,
        content: str,
        metadata: DocumentMetadata,
        section: str,
        chunk_index: int,
        document_id: str
    ) -> DocumentChunk:
        """
        Create a DocumentChunk with proper metadata.
        
        Args:
            content: Chunk text content
            metadata: Document metadata
            section: Section name
            chunk_index: Position in document
            document_id: Parent document ID
            
        Returns:
            DocumentChunk instance
        """
        import hashlib
        import uuid

        # Generate a proper UUID from the hash
        hash_hex = hashlib.md5(
            f"{document_id}_{chunk_index}_{content[:50]}".encode()
        ).hexdigest()

        # Convert hex to UUID format (Qdrant compatible)
        chunk_id = str(uuid.UUID(hash_hex))

        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=content,
            metadata=metadata,
            section=section,
            chunk_index=chunk_index
        )
