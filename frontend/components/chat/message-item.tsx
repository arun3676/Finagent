"use client";

import { useState, useEffect } from "react";
import { User, Bot, ChevronDown, ChevronRight, Copy, Check } from "lucide-react";
import { Message, MessageAgentStep } from "@/types/chat";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { CitationBadge } from "./citation-badge";
import { StreamingText } from "./streaming-text";
import { ResponseWithCitations } from "./response-with-citations";
import { CitationPanel } from "./citation-panel";
import { AnalystNotebook } from "./analyst-notebook";
import { TrustScoreFooter } from "./trust-score-footer";
import { FollowUpPanel } from "./FollowUpPanel";
import { useCitationExplorer } from "@/hooks/useCitationExplorer";

interface MessageItemProps {
  message: Message;
  showReasoning?: boolean;
}

export function MessageItem({ message, showReasoning = true }: MessageItemProps) {
  const [showSteps, setShowSteps] = useState(false);
  const [copied, setCopied] = useState(false);
  const [isNotebookExpanded, setIsNotebookExpanded] = useState(false);
  const [isTrustExpanded, setIsTrustExpanded] = useState(false);

  const isUser = message.role === "user";
  const isAssistant = message.role === "assistant";
  const citationThreshold = 0.7;
  const visibleCitations = (message.citations || []).filter(
    (citation) => (citation.confidence ?? 0) >= citationThreshold
  );

  // Citation explorer
  const {
    activeCitation,
    isPanelOpen,
    openCitation,
    closeCitation,
  } = useCitationExplorer(message.citations || []);

  // Auto-expand notebook for COMPLEX queries
  useEffect(() => {
    if (message.metadata?.query_complexity === 'complex' && message.analyst_notebook) {
      setIsNotebookExpanded(true);
    }
  }, [message.metadata?.query_complexity, message.analyst_notebook]);

  // Auto-expand trust score for low trust level
  useEffect(() => {
    if (message.validation?.trust_level === 'low') {
      setIsTrustExpanded(true);
    }
  }, [message.validation?.trust_level]);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Handle citation clicks from AnalystNotebook (receives citationId string)
  const handleNotebookCitationClick = (citationId: string) => {
    const citation = message.citations?.find((c) => c.citation_id === citationId);
    if (citation) {
      openCitation(citation);
    }
  };

  return (
    <div
      className={cn(
        "flex gap-4 p-4 rounded-lg",
        isUser ? "bg-muted/50" : "bg-background"
      )}
    >
      {/* Avatar */}
      <div
        className={cn(
          "flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center",
          isUser ? "bg-primary text-primary-foreground" : "bg-muted"
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>

      {/* Content */}
      <div className="flex-1 space-y-3 overflow-hidden">
        {/* Message Text */}
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {message.is_streaming ? (
            <StreamingText text={message.content} />
          ) : isAssistant && message.citations && message.citations.length > 0 ? (
            <ResponseWithCitations
              answer={message.content}
              citations={message.citations}
              onCitationClick={openCitation}
              activeCitationId={activeCitation?.citation_id || null}
            />
          ) : (
            <p className="whitespace-pre-wrap">{message.content}</p>
          )}
        </div>

        {/* Analyst's Notebook */}
        {isAssistant && message.analyst_notebook && (
          <AnalystNotebook
            notebook={message.analyst_notebook}
            onCitationClick={handleNotebookCitationClick}
            isExpanded={isNotebookExpanded}
            onToggle={() => setIsNotebookExpanded(!isNotebookExpanded)}
          />
        )}

        {/* Citations */}
        {isAssistant && visibleCitations.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {message.citations?.map((citation, idx) => {
              if ((citation.confidence ?? 0) < citationThreshold) {
                return null;
              }
              return (
                <CitationBadge
                  key={`${citation.citation_id || "citation"}-${idx}`}
                  citation={citation}
                  index={idx + 1}
                />
              );
            })}
          </div>
        )}

        {/* Follow-up Questions */}
        {isAssistant &&
          message.followUpQuestions &&
          message.followUpQuestions.length > 0 &&
          message.queryId && (
            <FollowUpPanel
              questions={message.followUpQuestions}
              parentQueryId={message.queryId}
              onFollowUpResponse={(response) => {
                console.log("Follow-up answered:", response);
              }}
            />
          )}

        {/* Agent Steps (Reasoning) */}
        {isAssistant && showReasoning && message.agent_steps && message.agent_steps.length > 0 && (
          <div className="border rounded-lg overflow-hidden">
            <button
              className="w-full flex items-center gap-2 p-2 text-sm text-muted-foreground hover:bg-muted/50 transition-colors"
              onClick={() => setShowSteps(!showSteps)}
            >
              {showSteps ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
              Agent Reasoning ({message.agent_steps.length} steps)
              {message.metadata?.total_duration_ms && (
                <span className="ml-auto">
                  {(message.metadata.total_duration_ms / 1000).toFixed(2)}s
                </span>
              )}
            </button>
            
            {showSteps && (
              <div className="border-t divide-y">
                {message.agent_steps.map((step, idx) => (
                  <AgentStepItem key={idx} step={step} index={idx + 1} />
                ))}
              </div>
            )}
          </div>
        )}

        {/* Trust Score Footer */}
        {isAssistant && message.validation && !message.is_streaming && (
          <TrustScoreFooter
            validation={message.validation}
            isExpanded={isTrustExpanded}
            onToggle={() => setIsTrustExpanded(!isTrustExpanded)}
          />
        )}

        {/* Actions */}
        {isAssistant && !message.is_streaming && (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={handleCopy}
            >
              {copied ? (
                <Check className="h-3 w-3 mr-1" />
              ) : (
                <Copy className="h-3 w-3 mr-1" />
              )}
              {copied ? "Copied" : "Copy"}
            </Button>
            {message.metadata?.query_complexity && (
              <Badge variant="outline" className="text-xs">
                {message.metadata.query_complexity}
              </Badge>
            )}
          </div>
        )}
      </div>

      {/* Citation Panel */}
      {isAssistant && (
        <CitationPanel
          citation={activeCitation}
          isOpen={isPanelOpen}
          onClose={closeCitation}
        />
      )}
    </div>
  );
}

// Agent Step Sub-component
function AgentStepItem({ step, index }: { step: MessageAgentStep; index: number }) {
  const statusColors = {
    pending: "text-muted-foreground",
    running: "text-blue-500",
    completed: "text-green-500",
    failed: "text-red-500",
  };

  return (
    <div className="p-3 text-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-muted-foreground">
            {index}.
          </span>
          <span className="font-medium">{step.step_name}</span>
          <span className={cn("text-xs", statusColors[step.status])}>
            ({step.status})
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          {step.duration_ms}ms
        </span>
      </div>
      <p className="text-muted-foreground mt-1 ml-6">{step.description}</p>
    </div>
  );
}
