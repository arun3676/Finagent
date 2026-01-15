"use client";

import { useEffect, useRef } from "react";
import { Message } from "@/types/chat";
import { MessageItem } from "./message-item";
import { ScrollArea } from "@/components/ui/scroll-area";

interface MessageListProps {
  messages: Message[];
  showReasoning?: boolean;
}

export function MessageList({ messages, showReasoning = true }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <div className="text-center space-y-4 max-w-md">
          <div className="text-4xl">📊</div>
          <h2 className="text-xl font-semibold">Financial Research Assistant</h2>
          <p className="text-muted-foreground">
            Ask questions about SEC filings, earnings calls, or company financials.
            I&apos;ll provide answers with citations to the source documents.
          </p>
          <div className="text-sm text-muted-foreground space-y-1">
            <p>Try asking:</p>
            <ul className="space-y-1">
              <li>&ldquo;What are Apple&apos;s main risk factors?&rdquo;</li>
              <li>&ldquo;Compare Microsoft and Google revenue growth&rdquo;</li>
              <li>&ldquo;What guidance did NVIDIA give for Q4?&rdquo;</li>
            </ul>
          </div>
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className="flex-1">
      <div className="space-y-4 p-4">
        {messages.map((message) => (
          <MessageItem
            key={message.id}
            message={message}
            showReasoning={showReasoning}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
