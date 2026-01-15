"use client";

import { useEffect, useState } from "react";
import { CheckCircle, XCircle, Loader2, FileText, Database, Brain, Search } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { IngestProgress } from "@/types/company";
import { cn } from "@/lib/utils";

interface IngestProgressCardProps {
  progress: IngestProgress;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

const STEP_ICONS: Record<string, React.ElementType> = {
  fetching: Search,
  parsing: FileText,
  chunking: FileText,
  embedding: Brain,
  indexing: Database,
  completed: CheckCircle,
  failed: XCircle,
};

const STEP_LABELS: Record<string, string> = {
  fetching: "Fetching SEC filings...",
  parsing: "Parsing documents...",
  chunking: "Creating chunks...",
  embedding: "Generating embeddings...",
  indexing: "Indexing to vector store...",
  completed: "Indexing complete!",
  failed: "Indexing failed",
};

export function IngestProgressCard({ progress, onComplete, onError }: IngestProgressCardProps) {
  const [animatedProgress, setAnimatedProgress] = useState(0);

  // Animate progress bar
  useEffect(() => {
    const timer = setTimeout(() => {
      setAnimatedProgress(progress.progress_percent);
    }, 100);
    return () => clearTimeout(timer);
  }, [progress.progress_percent]);

  // Callbacks
  useEffect(() => {
    if (progress.status === "completed" && onComplete) {
      onComplete();
    }
    if (progress.status === "failed" && onError) {
      onError(progress.error_message || "Unknown error");
    }
  }, [progress.status, progress.error_message, onComplete, onError]);

  const Icon = STEP_ICONS[progress.status] || Loader2;
  const isActive = !["completed", "failed"].includes(progress.status);
  const isFailed = progress.status === "failed";
  const isComplete = progress.status === "completed";

  return (
    <div
      className={cn(
        "border rounded-lg p-4 space-y-4",
        isComplete && "border-green-500/50 bg-green-500/5",
        isFailed && "border-red-500/50 bg-red-500/5"
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              "p-2 rounded-lg",
              isActive && "bg-blue-500/10 text-blue-500",
              isComplete && "bg-green-500/10 text-green-500",
              isFailed && "bg-red-500/10 text-red-500"
            )}
          >
            <Icon className={cn("h-5 w-5", isActive && "animate-spin")} />
          </div>
          <div>
            <h4 className="font-semibold">{progress.ticker}</h4>
            <p className="text-sm text-muted-foreground">
              {progress.current_step || STEP_LABELS[progress.status]}
            </p>
          </div>
        </div>
        <Badge
          variant={isComplete ? "default" : isFailed ? "destructive" : "secondary"}
        >
          {progress.status}
        </Badge>
      </div>

      {/* Progress Bar */}
      {isActive && (
        <div className="space-y-2">
          <Progress value={animatedProgress} className="h-2" />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{Math.round(animatedProgress)}% complete</span>
            <span>
              {progress.documents_processed}/{progress.documents_total} docs
            </span>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div className="space-y-1">
          <p className="text-2xl font-bold">{progress.documents_processed}</p>
          <p className="text-xs text-muted-foreground">Documents</p>
        </div>
        <div className="space-y-1">
          <p className="text-2xl font-bold">{progress.chunks_created}</p>
          <p className="text-xs text-muted-foreground">Chunks</p>
        </div>
        <div className="space-y-1">
          <p className="text-2xl font-bold">{Math.round(animatedProgress)}%</p>
          <p className="text-xs text-muted-foreground">Progress</p>
        </div>
      </div>

      {/* Error Message */}
      {isFailed && progress.error_message && (
        <div className="bg-red-500/10 border border-red-500/20 rounded p-3">
          <p className="text-sm text-red-600 dark:text-red-400">
            {progress.error_message}
          </p>
        </div>
      )}
    </div>
  );
}
