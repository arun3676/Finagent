import { render, screen, fireEvent, waitFor } from "@/lib/test-utils";
import { EnhancedCitationBadge } from "../enhanced-citation-badge";
import { mockEnhancedCitation } from "@/lib/mock-data";

describe("EnhancedCitationBadge", () => {
  it("renders citation badge with index", () => {
    render(<EnhancedCitationBadge citation={mockEnhancedCitation} index={1} />);
    
    expect(screen.getByText("[1]")).toBeInTheDocument();
  });

  it("shows correct confidence color for high confidence", () => {
    const highConfCitation = { ...mockEnhancedCitation, confidence_score: 0.95 };
    render(<EnhancedCitationBadge citation={highConfCitation} index={1} />);
    
    const badge = screen.getByText("[1]").closest("span");
    expect(badge).toHaveClass("text-green-600");
  });

  it("shows correct confidence color for medium confidence", () => {
    const mediumConfCitation = { ...mockEnhancedCitation, confidence_score: 0.75 };
    render(<EnhancedCitationBadge citation={mediumConfCitation} index={1} />);
    
    const badge = screen.getByText("[1]").closest("span");
    expect(badge).toHaveClass("text-yellow-600");
  });

  it("shows correct confidence color for low confidence", () => {
    const lowConfCitation = { ...mockEnhancedCitation, confidence_score: 0.5 };
    render(<EnhancedCitationBadge citation={lowConfCitation} index={1} />);
    
    const badge = screen.getByText("[1]").closest("span");
    expect(badge).toHaveClass("text-red-600");
  });

  it("opens dialog on click", async () => {
    render(<EnhancedCitationBadge citation={mockEnhancedCitation} index={1} />);
    
    fireEvent.click(screen.getByText("[1]"));
    
    await waitFor(() => {
      expect(screen.getByText("Source Document")).toBeInTheDocument();
    });
  });

  it("displays source document information in dialog", async () => {
    render(<EnhancedCitationBadge citation={mockEnhancedCitation} index={1} />);
    
    fireEvent.click(screen.getByText("[1]"));
    
    await waitFor(() => {
      expect(screen.getByText(/AAPL/)).toBeInTheDocument();
      expect(screen.getByText(/10K/)).toBeInTheDocument();
    });
  });

  it("copies source text to clipboard", async () => {
    Object.assign(navigator, {
      clipboard: {
        writeText: jest.fn(),
      },
    });

    render(<EnhancedCitationBadge citation={mockEnhancedCitation} index={1} />);
    
    fireEvent.click(screen.getByText("[1]"));
    
    await waitFor(() => {
      const copyButton = screen.getByText("Copy");
      fireEvent.click(copyButton);
    });

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
      mockEnhancedCitation.source_chunk.text
    );
  });
});
