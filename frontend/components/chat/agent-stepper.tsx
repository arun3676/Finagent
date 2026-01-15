"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  Circle,
  CheckCircle,
  XCircle,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { AgentStepperProps, AgentStep, StepStatus } from "@/types";

export function AgentStepper({
  steps,
  isExpanded: controlledIsExpanded,
  onToggleExpand,
  complexity,
}: AgentStepperProps) {
  const completedCount = steps.filter((s) => s.status === "completed").length;
  const totalCount = steps.length;
  const hasActiveStep = steps.some((s) => s.status === "active");
  const isComplete = completedCount === totalCount && !hasActiveStep;

  // Auto-expand for COMPLEX queries
  const shouldAutoExpand = complexity === "COMPLEX";

  return (
    <div className="border rounded-lg overflow-hidden bg-card shadow-sm">
      {/* Header */}
      <button
        onClick={onToggleExpand}
        className={cn(
          "w-full flex items-center justify-between px-4 py-3",
          "hover:bg-muted/50 transition-colors",
          "text-left"
        )}
      >
        <div className="flex items-center gap-3">
          {controlledIsExpanded ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}

          {hasActiveStep && !isComplete && (
            <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
          )}

          {isComplete && (
            <CheckCircle className="h-4 w-4 text-green-500" />
          )}

          <span className="text-sm font-medium">
            {hasActiveStep && !isComplete
              ? "Processing Query..."
              : isComplete
              ? "Query Complete"
              : "Agents Ready"}
          </span>

          <span className="text-xs text-muted-foreground">
            [{completedCount}/{totalCount} agents]
          </span>

          {complexity && (
            <span
              className={cn(
                "text-xs px-2 py-0.5 rounded-full font-medium",
                complexity === "SIMPLE" &&
                  "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
                complexity === "MODERATE" &&
                  "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400",
                complexity === "COMPLEX" &&
                  "bg-orange-100 text-orange-700 dark:bg-orange-900/30 dark:text-orange-400"
              )}
            >
              {complexity}
            </span>
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {controlledIsExpanded && (
        <div className="border-t bg-muted/20">
          <div className="px-4 py-3 space-y-2">
            {steps.map((step, index) => (
              <StepItem key={step.agent} step={step} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

interface StepItemProps {
  step: AgentStep;
}

function StepItem({ step }: StepItemProps) {
  const hasSubItems = step.subItems && step.subItems.length > 0;

  return (
    <div
      className={cn(
        "transition-all duration-300 ease-in-out",
        step.status === "active" && "animate-in slide-in-from-left-2"
      )}
    >
      <div className="flex items-start gap-3">
        {/* Status Icon */}
        <div className="flex-shrink-0 mt-0.5">
          <StepIcon status={step.status} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-2">
            <span
              className={cn(
                "text-sm font-medium",
                step.status === "pending" && "text-muted-foreground",
                step.status === "active" && "text-blue-600 dark:text-blue-400",
                step.status === "completed" && "text-foreground",
                step.status === "error" && "text-destructive"
              )}
            >
              {step.displayName}
            </span>

            {step.duration !== undefined && (
              <span className="text-xs text-muted-foreground">
                {step.duration}ms
              </span>
            )}
          </div>

          {/* Detail */}
          {step.detail && (
            <div className="text-xs text-muted-foreground mt-0.5">
              {step.detail}
            </div>
          )}

          {/* Sub-items */}
          {hasSubItems && (
            <div className="mt-2 space-y-1">
              {step.subItems!.map((subItem, idx) => (
                <div
                  key={idx}
                  className="text-xs text-muted-foreground pl-4 border-l-2 border-muted py-0.5 animate-in slide-in-from-left-1"
                >
                  {subItem}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface StepIconProps {
  status: StepStatus;
}

function StepIcon({ status }: StepIconProps) {
  switch (status) {
    case "pending":
      return (
        <Circle className="h-4 w-4 text-muted-foreground/50" strokeWidth={2} />
      );

    case "active":
      return (
        <div className="relative">
          <Circle
            className="h-4 w-4 text-blue-500 animate-pulse"
            strokeWidth={2}
            fill="currentColor"
          />
          <div className="absolute inset-0 rounded-full bg-blue-500/20 animate-ping" />
        </div>
      );

    case "completed":
      return (
        <CheckCircle
          className="h-4 w-4 text-green-500 animate-in zoom-in-50"
          strokeWidth={2}
        />
      );

    case "error":
      return (
        <XCircle
          className="h-4 w-4 text-destructive"
          strokeWidth={2}
        />
      );

    default:
      return null;
  }
}
