"use client";

import { useState } from "react";
import { Zap, AlignLeft, FileText, ChevronDown } from "lucide-react";
import type { ResponseLength } from "@/types/api";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const RESPONSE_LENGTH_OPTIONS: Array<{
  value: ResponseLength;
  label: string;
  description: string;
  Icon: typeof Zap;
}> = [
  {
    value: "short",
    label: "Short",
    description: "2-3 lines • Quick facts",
    Icon: Zap,
  },
  {
    value: "normal",
    label: "Normal",
    description: "5-6 lines • Balanced",
    Icon: AlignLeft,
  },
  {
    value: "detailed",
    label: "Detailed",
    description: "8-15 sentences • Full context",
    Icon: FileText,
  },
];

interface ResponseLengthSelectorProps {
  value: ResponseLength;
  onChange: (value: ResponseLength) => void;
  disabled?: boolean;
}

export function ResponseLengthSelector({
  value,
  onChange,
  disabled = false,
}: ResponseLengthSelectorProps) {
  // isCollapsed: when true, only show the selected option
  const [isCollapsed, setIsCollapsed] = useState(false);

  const handleOptionClick = (optionValue: ResponseLength) => {
    if (disabled) return;

    if (isCollapsed) {
      // If collapsed and clicking the selected option, expand to show all
      setIsCollapsed(false);
    } else {
      // If expanded, select the option and collapse
      onChange(optionValue);
      setIsCollapsed(true);
    }
  };

  const selectedOption = RESPONSE_LENGTH_OPTIONS.find(opt => opt.value === value);

  return (
    <TooltipProvider delayDuration={200}>
      <div
        className={cn(
          "inline-flex items-center gap-0.5 rounded-full bg-muted/70 p-1",
          "transition-all duration-300 ease-in-out"
        )}
      >
        {RESPONSE_LENGTH_OPTIONS.map((option) => {
          const isSelected = option.value === value;
          const isVisible = !isCollapsed || isSelected;

          return (
            <div
              key={option.value}
              className={cn(
                "transition-all duration-300 ease-in-out overflow-hidden",
                isVisible ? "opacity-100 max-w-[100px]" : "opacity-0 max-w-0"
              )}
            >
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => handleOptionClick(option.value)}
                    disabled={disabled}
                    className={cn(
                      "flex items-center gap-1 h-6 rounded-full px-2.5 text-[11px] font-medium whitespace-nowrap",
                      "transition-all duration-200 ease-in-out",
                      !disabled && "hover:bg-black/10",
                      isSelected
                        ? "bg-black text-white shadow-sm border border-black"
                        : "text-neutral-500",
                      disabled && "cursor-not-allowed",
                      disabled && !isSelected && "opacity-30"
                    )}
                  >
                    <option.Icon className="h-3 w-3 flex-shrink-0" aria-hidden="true" />
                    <span>{option.label}</span>
                    {isSelected && isCollapsed && (
                      <ChevronDown className="h-3 w-3 ml-0.5 flex-shrink-0" />
                    )}
                  </button>
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs">
                  <div className="font-medium text-foreground">{option.label}</div>
                  <div className="text-muted-foreground">{option.description}</div>
                  {isSelected && isCollapsed && (
                    <div className="text-primary mt-1">Click to change</div>
                  )}
                </TooltipContent>
              </Tooltip>
            </div>
          );
        })}
      </div>
    </TooltipProvider>
  );
}
