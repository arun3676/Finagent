"use client";

import { useCallback, useState, useEffect, useRef } from "react";
import { useChatStore } from "@/lib/stores/chat-store";
import { useUIStore } from "@/lib/stores/ui-store";
import { useAgentStepper } from "@/lib/hooks/useAgentStepper";
import { MessageList } from "./message-list";
import { ChatInput } from "./chat-input";
import { ComplexityBadge } from "./complexity-badge";
import { AgentStepper } from "./agent-stepper";
import { Message, ComplexityInfo, Citation } from "@/types/chat";
import { ValidationResult } from "@/types/validation";
import { AnalystNotebook } from "@/types/analyst";
import { QueryCitation, ResponseLength } from "@/types/api";
import { sendStreamingQuery } from "@/lib/api/query";
import { getFollowUpQuestions } from "@/lib/api/followup";
import type { FollowUpQuestion } from "@/types/followup";

/**
 * Convert QueryCitation (from API) to Citation (for display)
 */
function mapQueryCitationToDisplayCitation(qc: QueryCitation, index: number): Citation {
  // Extract source_url from various possible locations
  const sourceUrl = qc.page_reference || (qc.metadata as any)?.source_url || '';

  return {
    citation_id: qc.citation_id,
    citation_number: index + 1,
    claim: qc.claim,
    source_chunk_id: qc.source_chunk_id,
    source_text: qc.source_text,
    source_context: qc.source_text, // Use source_text as context fallback
    highlight_start: 0,
    highlight_end: qc.source_text?.length || 0,
    source_metadata: {
      ticker: qc.metadata?.ticker || '',
      company_name: qc.metadata?.ticker || '', // Fallback to ticker
      document_type: qc.metadata?.filing_type || '',
      filing_date: '',
      section: qc.metadata?.section || '',
      page_number: qc.metadata?.page || 0,
      source_url: sourceUrl,
    },
    confidence: qc.confidence,
    validation_method: 'semantic_similarity',
    preview_text: qc.source_text?.slice(0, 50) || '',
    // Legacy fields
    page_reference: qc.page_reference,
    source_url: sourceUrl,
    metadata: qc.metadata,
  };
}

export function ChatInterface() {
  const {
    activeConversationId,
    conversations,
    isStreaming,
    createConversation,
    addMessage,
    updateMessage,
    appendToMessage,
    setStreaming,
  } = useChatStore();

  const { showAgentReasoning } = useUIStore();

  // Complexity state
  const [complexity, setComplexity] = useState<ComplexityInfo | null>(null);

  // Validation state (using ref to avoid closure issues)
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const validationRef = useRef<ValidationResult | null>(null);

  // Analyst notebook state (using ref to avoid closure issues)
  const analystNotebookRef = useRef<AnalystNotebook | null>(null);

  // Agent stepper state
  const { steps, handleSSEEvent, reset: resetStepper, getCompletedCount, getTotalCount } = useAgentStepper();
  const [isStepperExpanded, setIsStepperExpanded] = useState(false);

  // Auto-expand stepper for COMPLEX queries
  useEffect(() => {
    if (complexity?.level === 'COMPLEX') {
      setIsStepperExpanded(true);
    }
  }, [complexity]);

  // Get active conversation messages
  const activeConversation = conversations.find(
    (c) => c.id === activeConversationId
  );
  const messages = activeConversation?.messages || [];

  // Function to fetch follow-up questions with retry logic
  const fetchFollowUpQuestions = useCallback(async (
    convId: string,
    messageId: string,
    queryId: string,
    retryCount = 0
  ) => {
    const maxRetries = 3;
    const retryDelay = 2000; // 2 seconds

    try {
      const result = await getFollowUpQuestions(queryId);
      
      if (result.follow_up_questions && result.follow_up_questions.length > 0) {
        // Map backend response to frontend types
        const followUpQuestions: FollowUpQuestion[] = result.follow_up_questions.map((q: any) => ({
          id: q.id,
          text: q.text,
          category: q.category,
          relevantChunkIds: q.relevant_chunk_ids || [],
          requiresNewRetrieval: q.requires_new_retrieval || false,
        }));

        updateMessage(convId, messageId, {
          followUpQuestions,
        });
      } else if (
        (result.status === "pending" || result.status === "not_found") &&
        retryCount < maxRetries
      ) {
        // Follow-ups are still being generated, retry after delay
        setTimeout(() => {
          fetchFollowUpQuestions(convId, messageId, queryId, retryCount + 1);
        }, retryDelay);
      }
    } catch (error) {
      console.error("Failed to fetch follow-up questions:", error);
      // Don't retry on error, just log it
    }
  }, [updateMessage]);

  const handleSubmit = useCallback(
    async (content: string, responseLength?: ResponseLength) => {
      // Ensure we have an active conversation
      let convId = activeConversationId;
      if (!convId) {
        convId = createConversation();
      }

      // Reset complexity state, stepper, and validation
      setComplexity(null);
      setValidation(null);
      validationRef.current = null;
      analystNotebookRef.current = null;
      resetStepper();
      setIsStepperExpanded(false);

      // Add user message
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: new Date(),
      };
      addMessage(convId, userMessage);

      // Create placeholder assistant message
      const assistantMessageId = crypto.randomUUID();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        timestamp: new Date(),
        is_streaming: true,
      };
      addMessage(convId, assistantMessage);

      // Start streaming
      setStreaming(true);

      try {
        await sendStreamingQuery(
          {
            query: content,
            response_length: responseLength || "normal",
            options: {
              include_reasoning: true,
              include_citations: true,
            },
          },
          (chunk) => {
            appendToMessage(convId, assistantMessageId, chunk);
          },
          (response) => {
            // Use refs to get the latest values (avoid stale closure)
            // Map QueryCitation[] to Citation[] for display
            const displayCitations = (response.citations || []).map(
              (qc, idx) => mapQueryCitationToDisplayCitation(qc, idx)
            );
            updateMessage(convId, assistantMessageId, {
              is_streaming: false,
              citations: displayCitations,
              analyst_notebook: response.analyst_notebook || analystNotebookRef.current || undefined,
              validation: validationRef.current || undefined,
              queryId: response.query_id,
              metadata: {
                query_time_ms: response.metadata.query_time_ms,
                model_used: response.metadata.model_used,
                sources_consulted: response.metadata.sources_consulted,
                confidence: response.metadata.confidence,
              },
            });

            // Fetch follow-up questions after response is complete
            if (response.query_id) {
              fetchFollowUpQuestions(convId, assistantMessageId, response.query_id);
            }
          },
          (error) => {
            updateMessage(convId, assistantMessageId, {
              content: `Error: ${error}`,
              is_streaming: false,
              validation: validationRef.current || undefined, // Include any validation received before error
            });
          },
          (complexityData) => {
            setComplexity(complexityData);
          },
          (stepEvent) => {
            handleSSEEvent(stepEvent);
          },
          (validationData) => {
            validationRef.current = validationData;
            setValidation(validationData);
          },
          (notebookData) => {
            analystNotebookRef.current = notebookData;
          }
        );
      } catch (error) {
        console.error("Chat error:", error);
        updateMessage(convId, assistantMessageId, {
          content: "Sorry, there was an error processing your request. Please try again.",
          is_streaming: false,
        });
      } finally {
        setStreaming(false);
      }
    },
    [
      activeConversationId,
      messages,
      createConversation,
      addMessage,
      updateMessage,
      appendToMessage,
      setStreaming,
      resetStepper,
      handleSSEEvent,
      fetchFollowUpQuestions,
    ]
  );

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} showReasoning={showAgentReasoning} />
      <div className="border-t bg-background">
        <div className="max-w-4xl mx-auto px-4 pt-2 pb-1 space-y-2">
          {/* Show stepper when query is processing */}
          {(isStreaming || getCompletedCount() > 0) && (
            <AgentStepper
              steps={steps}
              isExpanded={isStepperExpanded}
              onToggleExpand={() => setIsStepperExpanded(!isStepperExpanded)}
              complexity={complexity?.level}
            />
          )}
          <ComplexityBadge complexity={complexity} isLoading={isStreaming} />
        </div>
        <ChatInput onSubmit={handleSubmit} isLoading={isStreaming} />
      </div>
    </div>
  );
}
