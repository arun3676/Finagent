"use client";

import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

interface ConfidenceBarProps {
  label: string;
  value: number; // 0-1
  showPercentage?: boolean;
}

export function ConfidenceBar({
  label,
  value,
  showPercentage = true,
}: ConfidenceBarProps) {
  const [animatedValue, setAnimatedValue] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedValue(value);
    }, 100);
    return () => clearTimeout(timer);
  }, [value]);

  const percentage = Math.round(value * 100);
  const barColor =
    value >= 0.85
      ? "bg-green-500"
      : value >= 0.65
      ? "bg-yellow-500"
      : "bg-red-500";

  return (
    <div className="flex items-center gap-3">
      <span className="text-xs text-muted-foreground min-w-[140px]">
        {label}
      </span>
      <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full transition-all duration-1000 ease-out",
            barColor
          )}
          style={{ width: `${animatedValue * 100}%` }}
        />
      </div>
      {showPercentage && (
        <span className="text-xs font-medium min-w-[40px] text-right">
          {percentage}%
        </span>
      )}
    </div>
  );
}
