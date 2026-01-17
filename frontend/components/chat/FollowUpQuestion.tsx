"use client";

import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { FollowUpQuestion, FollowUpResponse } from "@/types/followup";
import { FollowUpResponseDisplay } from "./FollowUpResponse";

const categoryIcons: Record<FollowUpQuestion["category"], string> = {
  temporal: "📈",
  deeper: "🔍",
  comparative: "⚖️",
  related: "🔗",
};

interface FollowUpQuestionProps {
  question: FollowUpQuestion;
  isLoading: boolean;
  isDisabled: boolean;
  response: FollowUpResponse | null;
  onClick: () => void;
}

export function FollowUpQuestionItem({
  question,
  isLoading,
  isDisabled,
  response,
  onClick,
}: FollowUpQuestionProps) {
  const responseRef = useRef<HTMLDivElement | null>(null);
  const hasResponse = useRef(false);

  useEffect(() => {
    if (response && !hasResponse.current) {
      responseRef.current?.focus();
    }
    hasResponse.current = Boolean(response);
  }, [response]);

  const speedIndicator = question.requiresNewRetrieval ? "📊" : "⚡";
  const isButtonDisabled = isDisabled || isLoading;

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={onClick}
        disabled={isButtonDisabled}
        aria-disabled={isButtonDisabled}
        aria-busy={isLoading}
        className={cn(
          "w-full p-3 text-left hover:bg-blue-50 dark:hover:bg-blue-900/20",
          "transition-colors duration-150 flex items-center gap-3 rounded-md",
          isLoading && "animate-pulse bg-slate-200 dark:bg-slate-700",
          isButtonDisabled && "cursor-not-allowed opacity-60"
        )}
      >
        <span className="text-lg" aria-hidden>
          {categoryIcons[question.category]}
        </span>
        <span className="flex-1 text-sm text-slate-900 dark:text-slate-100">
          {question.text}
        </span>
        {isLoading ? (
          <span className="flex items-center gap-2 text-xs text-slate-500">
            <Loader2 className="h-4 w-4 animate-spin" />
            <span className="sr-only" role="status" aria-live="polite">
              Loading follow-up response
            </span>
          </span>
        ) : (
          <span className="text-xs text-slate-500" aria-hidden>
            {speedIndicator}
          </span>
        )}
      </button>
      <div
        ref={responseRef}
        tabIndex={-1}
        aria-live="polite"
        className={cn(
          "transition-all duration-200 ease-out",
          response ? "max-h-96 opacity-100" : "max-h-0 opacity-0 overflow-hidden"
        )}
      >
        {response && <FollowUpResponseDisplay response={response} />}
      </div>
    </div>
  );
}
