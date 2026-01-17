"use client";

import { Citation } from "@/types/chat";
import { CitationMarker } from "./citation-marker";

interface ResponseWithCitationsProps {
  answer: string;
  citations: Citation[];
  onCitationClick: (citation: Citation) => void;
  activeCitationId: string | null;
}

export function ResponseWithCitations({
  answer,
  citations,
  onCitationClick,
  activeCitationId,
}: ResponseWithCitationsProps) {
  // Parse answer text and replace [1], [2], etc. with CitationMarker components
  const parseAnswerWithCitations = () => {
    if (!answer || citations.length === 0) {
      return <span>{answer}</span>;
    }

    // Regular expression to match citation markers like [1], [2], etc.
    const citationRegex = /\[(\d+)\]/g;
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let match;

    while ((match = citationRegex.exec(answer)) !== null) {
      const citationNumber = parseInt(match[1], 10);
      const citation = citations.find((c) => c.citation_number === citationNumber);

      // Add text before the citation marker
      if (match.index > lastIndex) {
        parts.push(
          <span key={`text-${lastIndex}`}>
            {answer.slice(lastIndex, match.index)}
          </span>
        );
      }

      // Add citation marker or fallback text
      if (citation) {
        parts.push(
          <CitationMarker
            key={`citation-${citation.citation_id}-${match.index}`}
            number={citationNumber}
            citation={citation}
            isActive={citation.citation_id === activeCitationId}
            onClick={() => onCitationClick(citation)}
          />
        );
      } else {
        // Fallback if citation not found
        parts.push(
          <sup key={`missing-${citationNumber}-${match.index}`} className="text-muted-foreground">
            [{citationNumber}]
          </sup>
        );
      }

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text after last citation
    if (lastIndex < answer.length) {
      parts.push(
        <span key={`text-${lastIndex}`}>{answer.slice(lastIndex)}</span>
      );
    }

    return <>{parts}</>;
  };

  return (
    <div className="prose prose-sm max-w-none dark:prose-invert">
      {parseAnswerWithCitations()}
    </div>
  );
}
