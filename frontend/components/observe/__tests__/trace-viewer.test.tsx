import { render, screen, fireEvent } from "@/lib/test-utils";
import { TraceViewer } from "../trace-viewer";
import { mockTraceEvents } from "@/lib/mock-data";

describe("TraceViewer", () => {
  it("renders all trace events", () => {
    render(<TraceViewer events={mockTraceEvents} />);
    
    expect(screen.getByText("Query Classification")).toBeInTheDocument();
    expect(screen.getByText("Document Retrieval")).toBeInTheDocument();
    expect(screen.getByText("Response Generation")).toBeInTheDocument();
  });

  it("displays event durations", () => {
    render(<TraceViewer events={mockTraceEvents} />);
    
    expect(screen.getByText("245ms")).toBeInTheDocument();
    expect(screen.getByText("890ms")).toBeInTheDocument();
    expect(screen.getByText("1240ms")).toBeInTheDocument();
  });

  it("shows metadata when available", () => {
    render(<TraceViewer events={mockTraceEvents} />);
    
    expect(screen.getByText("5 chunks")).toBeInTheDocument();
    expect(screen.getByText("456 tokens")).toBeInTheDocument();
  });

  it("expands event on click", () => {
    render(<TraceViewer events={mockTraceEvents} />);
    
    const event = screen.getByText("Query Classification");
    fireEvent.click(event);
    
    expect(screen.getByText("Input")).toBeInTheDocument();
    expect(screen.getByText("Output")).toBeInTheDocument();
  });

  it("displays loading state", () => {
    render(<TraceViewer events={[]} isLoading />);
    
    expect(screen.getByText("Processing...")).toBeInTheDocument();
  });

  it("shows empty state when no events", () => {
    render(<TraceViewer events={[]} />);
    
    expect(screen.getByText("No trace events to display")).toBeInTheDocument();
  });

  it("applies correct color for event types", () => {
    render(<TraceViewer events={mockTraceEvents} />);
    
    const routerEvent = screen.getByText("Query Classification").closest("div");
    expect(routerEvent).toHaveClass("text-purple-500");
  });
});
