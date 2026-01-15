"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Shield, AlertTriangle, AlertCircle } from "lucide-react";

interface TrustBadgeProps {
  trustLevel: "high" | "medium" | "low";
  trustLabel: string;
  trustColor: string;
}

export function TrustBadge({ trustLevel, trustLabel, trustColor }: TrustBadgeProps) {
  const config = {
    high: {
      icon: Shield,
      className: "bg-green-100 text-green-800 border-green-300 dark:bg-green-900/20 dark:text-green-400",
    },
    medium: {
      icon: AlertTriangle,
      className: "bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/20 dark:text-amber-400",
    },
    low: {
      icon: AlertCircle,
      className: "bg-red-100 text-red-800 border-red-300 dark:bg-red-900/20 dark:text-red-400",
    },
  };

  const { icon: Icon, className } = config[trustLevel];

  return (
    <Badge variant="outline" className={cn("gap-1.5 font-medium", className)}>
      <Icon className="h-3.5 w-3.5" />
      {trustLabel}
    </Badge>
  );
}
