"use client";

import { useState, useRef, KeyboardEvent } from "react";
import { Send, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { ResponseLengthSelector } from "./response-length-selector";
import { cn } from "@/lib/utils";
import type { ResponseLength } from "@/types/api";

interface ChatInputProps {
  onSubmit: (message: string, responseLength?: ResponseLength) => void;
  isLoading?: boolean;
  placeholder?: string;
  disabled?: boolean;
}

export function ChatInput({
  onSubmit,
  isLoading = false,
  placeholder = "Ask about company financials, SEC filings, or earnings...",
  disabled = false,
}: ChatInputProps) {
  const [input, setInput] = useState("");
  const [responseLength, setResponseLength] = useState<ResponseLength>("normal");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    if (input.trim() && !isLoading && !disabled) {
      onSubmit(input.trim(), responseLength);
      setInput("");
      // Reset textarea height
      if (textareaRef.current) {
        textareaRef.current.style.height = "auto";
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  // Auto-resize textarea
  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = e.target;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    setInput(textarea.value);
  };

  return (
    <div className="bg-background p-4">
      <div className="relative flex items-end gap-2 max-w-4xl mx-auto">
        <div className="relative flex-1">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={isLoading || disabled}
            className={cn(
              "min-h-[52px] max-h-[200px] resize-none pr-12",
              "rounded-xl border-2 focus:border-primary",
              "text-base placeholder:text-muted-foreground"
            )}
            rows={1}
          />
          <Button
            size="icon"
            className="absolute right-2 bottom-2 h-8 w-8 rounded-lg"
            onClick={handleSubmit}
            disabled={!input.trim() || isLoading || disabled}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
      </div>
      
      {/* Response Length Selector */}
      <div className="flex items-center justify-between max-w-4xl mx-auto mt-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Response length:</span>
          <ResponseLengthSelector
            value={responseLength}
            onChange={setResponseLength}
            disabled={isLoading || disabled}
          />
        </div>
        <p className="text-xs text-muted-foreground">
          FinAgent can make mistakes. Verify important financial data.
        </p>
      </div>
    </div>
  );
}
