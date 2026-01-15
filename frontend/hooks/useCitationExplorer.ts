import { useState } from "react";
import { Citation } from "@/types/chat";

export function useCitationExplorer(citations: Citation[]) {
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const openCitation = (citation: Citation) => {
    setActiveCitation(citation);
    setIsPanelOpen(true);
  };

  const closeCitation = () => {
    setIsPanelOpen(false);
    // Delay clearing active so animation completes
    setTimeout(() => setActiveCitation(null), 300);
  };

  const getCitationByNumber = (number: number): Citation | undefined => {
    return citations.find((c) => c.citation_number === number);
  };

  return {
    activeCitation,
    isPanelOpen,
    openCitation,
    closeCitation,
    getCitationByNumber,
  };
}
