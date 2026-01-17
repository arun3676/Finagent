"use client";

import { useId, useState } from "react";
import { ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { executeFollowUp } from "@/lib/api/followup";
import type { FollowUpQuestion, FollowUpResponse } from "@/types/followup";
import { FollowUpQuestionItem } from "./FollowUpQuestion";

interface FollowUpPanelProps {
  questions: FollowUpQuestion[];
  parentQueryId: string;
  onFollowUpResponse: (response: FollowUpResponse) => void;
}

export function FollowUpPanel({
  questions,
  parentQueryId,
  onFollowUpResponse,
}: FollowUpPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [responses, setResponses] = useState<Record<string, FollowUpResponse>>(
    {}
  );
  const panelId = useId();

  if (!questions || questions.length === 0) {
    return null;
  }

  const handleQuestionClick = async (question: FollowUpQuestion) => {
    if (loadingId) {
      return;
    }

    const existingResponse = responses[question.id];
    if (existingResponse && !existingResponse.error) {
      return;
    }

    setLoadingId(question.id);

    try {
      const response = await executeFollowUp({
        questionId: question.id,
        questionText: question.text,
        parentQueryId,
      });

      setResponses((prev) => ({
        ...prev,
        [question.id]: response,
      }));
      onFollowUpResponse(response);
    } catch {
      setResponses((prev) => ({
        ...prev,
        [question.id]: {
          question: question.text,
          answer: "Sorry, couldn't load this answer. Please try again.",
          citations: [],
          executionTimeMs: 0,
          error: true,
        },
      }));
    } finally {
      setLoadingId(null);
    }
  };

  return (
    <div
      className={cn(
        isExpanded &&
          "border border-slate-200 dark:border-slate-700 rounded-lg overflow-hidden"
      )}
    >
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-controls={panelId}
        onClick={() => setIsExpanded((prev) => !prev)}
        className={cn(
          "flex items-center justify-between p-3 bg-slate-50 dark:bg-slate-800",
          "cursor-pointer hover:bg-slate-100 dark:hover:bg-slate-700",
          "transition-colors duration-200",
          isExpanded ? "rounded-none" : "rounded-lg"
        )}
      >
        <span className="flex items-center gap-2 text-sm font-medium">
          <span aria-hidden>💡</span>
          Explore further
        </span>
        <ChevronRight
          className={cn(
            "h-4 w-4 transition-transform duration-200",
            isExpanded ? "rotate-90" : "rotate-0"
          )}
          aria-hidden
        />
      </button>

      <div
        id={panelId}
        className={cn(
          "transition-all duration-300 ease-in-out",
          isExpanded ? "max-h-96 opacity-100" : "max-h-0 opacity-0 overflow-hidden"
        )}
      >
        <div className="border-t border-slate-200 dark:border-slate-700 p-3 space-y-3">
          {questions.map((question) => {
            const response = responses[question.id] || null;
            const isLoading = loadingId === question.id;
            const isDisabled = Boolean(loadingId && loadingId !== question.id);

            return (
              <FollowUpQuestionItem
                key={question.id}
                question={question}
                isLoading={isLoading}
                isDisabled={isDisabled}
                response={response}
                onClick={() => handleQuestionClick(question)}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
