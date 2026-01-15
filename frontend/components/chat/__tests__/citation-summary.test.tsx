import { render, screen } from "@/lib/test-utils";
import { CitationSummary } from "../citation-summary";
import { mockEnhancedCitation } from "@/lib/mock-data";

describe("CitationSummary", () => {
  const mockCitations = [
    { ...mockEnhancedCitation, id: "1", confidence_score: 0.95 },
    { ...mockEnhancedCitation, id: "2", confidence_score: 0.85 },
    { ...mockEnhancedCitation, id: "3", confidence_score: 0.65 },
  ];

  it("displays total citation count", () => {
    render(<CitationSummary citations={mockCitations} coverage={90} />);
    
    expect(screen.getByText("3 Citations")).toBeInTheDocument();
  });

  it("displays coverage percentage", () => {
    render(<CitationSummary citations={mockCitations} coverage={90} />);
    
    expect(screen.getByText("90%")).toBeInTheDocument();
  });

  it("shows high confidence badge", () => {
    render(<CitationSummary citations={mockCitations} coverage={90} />);
    
    expect(screen.getByText("1 high")).toBeInTheDocument();
  });

  it("shows medium confidence badge", () => {
    render(<CitationSummary citations={mockCitations} coverage={90} />);
    
    expect(screen.getByText("1 medium")).toBeInTheDocument();
  });

  it("shows low confidence badge", () => {
    render(<CitationSummary citations={mockCitations} coverage={90} />);
    
    expect(screen.getByText("1 low")).toBeInTheDocument();
  });

  it("applies correct color for high coverage", () => {
    render(<CitationSummary citations={mockCitations} coverage={95} />);
    
    const coverageText = screen.getByText("95%");
    expect(coverageText).toHaveClass("text-green-500");
  });

  it("applies correct color for medium coverage", () => {
    render(<CitationSummary citations={mockCitations} coverage={75} />);
    
    const coverageText = screen.getByText("75%");
    expect(coverageText).toHaveClass("text-yellow-500");
  });

  it("applies correct color for low coverage", () => {
    render(<CitationSummary citations={mockCitations} coverage={50} />);
    
    const coverageText = screen.getByText("50%");
    expect(coverageText).toHaveClass("text-red-500");
  });
});
