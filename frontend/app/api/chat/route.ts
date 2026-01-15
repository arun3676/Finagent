import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";
export const maxDuration = 60;

interface ChatMessage {
  role: string;
  content: string;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { messages, ticker_context } = body as {
      messages: ChatMessage[];
      ticker_context?: string[];
    };

    // Get the last user message as the query
    const lastUserMessage = messages.filter((m) => m.role === "user").pop();
    
    if (!lastUserMessage) {
      return NextResponse.json(
        { error: "No user message found" },
        { status: 400 }
      );
    }

    // Use port 8000 by default; no /api/v1 prefix
    const backendUrl =
      process.env.NEXT_PUBLIC_FINAGENT_API_URL ||
      process.env.NEXT_PUBLIC_API_URL ||
      "http://localhost:8000";

    // Build request to FastAPI backend
    const queryRequest = {
      query: lastUserMessage.content,
      filters: ticker_context ? { tickers: ticker_context } : undefined,
      options: {
        include_reasoning: true,
        include_citations: true,
        max_sources: 5,
      },
    };

    // Try streaming endpoint first, fall back to regular query
    const streamingUrl = `${backendUrl}/query/stream`;
    const regularUrl = `${backendUrl}/query`;

    // First, check if streaming endpoint exists
    let response: Response;
    
    try {
      response = await fetch(streamingUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(queryRequest),
      });

      if (response.ok && response.headers.get("content-type")?.includes("text/event-stream")) {
        // Return streaming response
        return new Response(response.body, {
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }
    } catch {
      // Streaming endpoint doesn't exist, fall back
    }

    // Fall back to regular endpoint
    response = await fetch(regularUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(queryRequest),
    });

    if (!response.ok) {
      const error = await response.text();
      console.error("Backend error:", error);
      return NextResponse.json(
        { error: "Backend request failed" },
        { status: response.status }
      );
    }

    const data = await response.json();

    // Convert backend response to chat format
    // This simulates streaming by returning the full response
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      start(controller) {
        // Send the answer as a stream
        controller.enqueue(encoder.encode(data.answer || data.response || "No response"));
        controller.close();
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
      },
    });
  } catch (error) {
    console.error("API route error:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
