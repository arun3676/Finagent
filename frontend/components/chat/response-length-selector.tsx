"use client";

import { useState } from "react";
import { Zap, AlignLeft, FileText } from "lucide-react";
import type { ResponseLength } from "@/types/api";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
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
  const [isExpanded, setIsExpanded] = useState(true);

  const handleClick = (optionValue: ResponseLength) => {
    if (isExpanded) {
      // If expanded and clicking any button, select it and collapse
      onChange(optionValue);
      setIsExpanded(false);
    } else {
      // If collapsed and clicking the selected button, expand to show all
      setIsExpanded(true);
    }
  };

  const visibleOptions = isExpanded 
    ? RESPONSE_LENGTH_OPTIONS 
    : RESPONSE_LENGTH_OPTIONS.filter(opt => opt.value === value);

  return (
    <TooltipProvider delayDuration={200}>
      <Tabs
        value={value}
        onValueChange={(nextValue) => {
          onChange(nextValue as ResponseLength);
        }}
        className="w-auto"
      >
        <TabsList className="h-8 rounded-full bg-muted/70 p-1">
          {visibleOptions.map((option) => (
            <Tooltip key={option.value}>
              <TooltipTrigger asChild>
                <TabsTrigger
                  value={option.value}
                  onClick={() => handleClick(option.value)}
                  className={cn(
                    "h-6 gap-1 rounded-full px-2 text-[11px] text-neutral-500",
                    !disabled && "hover:bg-black/10",
                    "data-[state=active]:bg-black data-[state=active]:text-white data-[state=active]:shadow-sm",
                    "data-[state=active]:border data-[state=active]:border-black",
                    disabled && "cursor-not-allowed",
                    disabled && value !== option.value && "opacity-30"
                  )}
                >
                  <option.Icon className="h-3 w-3" aria-hidden="true" />
                  <span>{option.label}</span>
                </TabsTrigger>
              </TooltipTrigger>
              <TooltipContent side="top" className="text-xs">
                <div className="font-medium text-foreground">{option.label}</div>
                <div className="text-muted-foreground">{option.description}</div>
              </TooltipContent>
            </Tooltip>
          ))}
        </TabsList>
      </Tabs>
    </TooltipProvider>
  );
}
