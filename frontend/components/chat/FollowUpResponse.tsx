"use client";

import type { ReactNode } from "react";
import type { FollowUpResponse } from "@/types/followup";

interface FollowUpResponseProps {
  response: FollowUpResponse;
}

function formatExecutionTime(executionTimeMs: number): string | null {
  if (!executionTimeMs) {
    return null;
  }
  const seconds = executionTimeMs / 1000;
  return `Answered in ${seconds.toFixed(1)}s`;
}

function renderAnswerWithCitations(answer: string) {
  if (!answer) {
    return null;
  }

  const citationRegex = /\\[(\\d+)\\]/g;
  const parts: ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;

  while ((match = citationRegex.exec(answer)) !== null) {
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {answer.slice(lastIndex, match.index)}
        </span>
      );
    }

    parts.push(
      <sup
        key={`citation-${match.index}`}
        className="text-xs text-slate-500 dark:text-slate-400"
      >
        [{match[1]}]
      </sup>
    );

    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < answer.length) {
    parts.push(<span key={`text-${lastIndex}`}>{answer.slice(lastIndex)}</span>);
  }

  return parts;
}

export function FollowUpResponseDisplay({ response }: FollowUpResponseProps) {
  const executionLabel = formatExecutionTime(response.executionTimeMs);
  const answerContent = renderAnswerWithCitations(response.answer);

  return (
    <div className="mt-2 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md text-sm">
      <p className={response.error ? "text-rose-600 dark:text-rose-400" : ""}>
        {answerContent || response.answer}
      </p>
      {executionLabel && (
        <div className="text-xs text-slate-400 mt-1">{executionLabel}</div>
      )}
    </div>
  );
}
