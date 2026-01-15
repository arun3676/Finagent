"use client";

import { useState } from "react";
import { Lightbulb } from "lucide-react";
import { KeyFindingsProps } from "@/types";

export function KeyFindings({ findings }: KeyFindingsProps) {
  const [showAll, setShowAll] = useState(false);
  const MAX_VISIBLE = 5;

  if (!findings || findings.length === 0) {
    return null;
  }

  const visibleFindings = showAll ? findings : findings.slice(0, MAX_VISIBLE);
  const hasMore = findings.length > MAX_VISIBLE;

  return (
    <div className="bg-muted/30 rounded-lg p-4 border border-muted">
      <div className="flex items-center gap-2 mb-3">
        <Lightbulb className="h-5 w-5 text-yellow-600 dark:text-yellow-500" />
        <h3 className="text-sm font-semibold">Key Findings</h3>
      </div>

      <ul className="space-y-2">
        {visibleFindings.map((finding, index) => (
          <li
            key={index}
            className="text-sm text-foreground flex items-start gap-2 animate-in slide-in-from-left-2"
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <span className="text-muted-foreground mt-1 flex-shrink-0">•</span>
            <span>{finding}</span>
          </li>
        ))}
      </ul>

      {hasMore && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="text-xs text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 mt-3 font-medium"
        >
          {showAll ? 'Show less' : `Show ${findings.length - MAX_VISIBLE} more`}
        </button>
      )}
    </div>
  );
}
