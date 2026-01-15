import { RemoteRunnable } from "@langchain/core/runnables/remote";
import { EventHandlerFields, exposeEndpoints, streamRunnableUI } from "@/utils/server";
import "server-only";
import { StreamEvent } from "@langchain/core/tracers/log_stream";
import { createStreamableUI, createStreamableValue } from "ai/rsc";
import { AIMessage } from "@/app/ai/message";
import { ChartContainer, ChartLoading } from "@/components/prebuilt/chart-container";
import React from "react";
import { LineItemsTable, LineItemsTableLoading } from "@/components/prebuilt/line-items-table";
import { WebSearchResults, WebSearchResultsLoading } from "@/components/web-search-results";
import {
  InsiderTransactionsTable,
  InsiderTransactionsTableLoading
} from "@/components/prebuilt/insider-transactions-table";

const API_URL = process.env.NEXT_PUBLIC_FINAGENT_API_URL 
  ? `${process.env.NEXT_PUBLIC_FINAGENT_API_URL}/chat`
  : "http://localhost:8000/chat";

type ToolComponent = {
  loading: (props?: any) => React.JSX.Element;
  final: (props?: any) => React.JSX.Element;
};

type ToolComponentMap = {
  [tool: string]: ToolComponent;
};

const TOOL_COMPONENT_MAP: ToolComponentMap = {
  "get-prices": {
    loading: (props?: any) => <ChartLoading {...props} />,
    final: (props?: any) => <ChartContainer {...props} />,
  },
  "search-line-items": {
    loading: (props?: any) => <LineItemsTableLoading/>,
    final: (props?: any) => <LineItemsTable {...props} />,
  },
  "search-web": {
    loading: (props?: any) => <WebSearchResultsLoading/>,
    final: (props?: any) => <WebSearchResults {...props} />,
  },
  "insider-transactions": {
    loading: (props?: any) => <InsiderTransactionsTableLoading/>,
    final: (props?: any) => <InsiderTransactionsTable {...props} />,
  }
};

async function agent(inputs: {
  input: string;
  chat_history: [role: string, content: string][];
  file?: {
    base64: string;
    extension: string;
  };
}) {
  "use server";
  const remoteRunnable = new RemoteRunnable({
    url: API_URL,
    options: {
      timeout: 300000, // 5 minutes to handle long ingestion processes
    },
  });

  let selectedToolComponent: ToolComponent | null = null;
  let selectedToolUI: ReturnType<typeof createStreamableUI> | null = null;
  let hasAssistantOutput = false;

  const ensureTextStream = (runId: string, fields: EventHandlerFields) => {
    if (!fields.callbacks[runId]) {
      const textStream = createStreamableValue();
      fields.ui.append(<AIMessage value={textStream.value} />);
      fields.callbacks[runId] = textStream;
    }
    return fields.callbacks[runId];
  };

  const extractTextFromChatEnd = (event: StreamEvent): string | null => {
    const output: any = event.data?.output;
    const generations = output?.generations;
    if (Array.isArray(generations) && generations[0]?.[0]) {
      const gen = generations[0][0];
      if (typeof gen?.text === "string") return gen.text;
      if (typeof gen?.message?.content === "string") return gen.message.content;
    }
    if (typeof output?.text === "string") return output.text;
    if (typeof output?.content === "string") return output.content;
    return null;
  };

  /**
   * Handles 'on_chain_start' and 'on_chain_end' for specific nodes
   */
  const handleChainEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    // Handle Retriever Node
    if (event.name === "retriever") {
      if (event.event === "on_chain_start") {
        const ui = createStreamableUI(<WebSearchResultsLoading />);
        fields.ui.append(ui.value);
        fields.callbacks["retriever"] = ui;
      } else if (event.event === "on_chain_end") {
        const ui = fields.callbacks["retriever"];
        if (ui && "done" in ui) {
          // Extract docs from AgentState
          // event.data.output is AgentState which has retrieved_docs
          const output = event.data.output;
          // output might be the AgentState object directly or wrapped
          const docs = output?.retrieved_docs || [];
          
          // Map backend docs to frontend component props
          const searchResults = docs.map((d: any) => ({
             title: d.chunk.metadata.company_name + " - " + d.chunk.metadata.document_type,
             content: d.chunk.content,
             url: d.chunk.metadata.source_url,
             score: d.score
          }));
          
          ui.done(<WebSearchResults results={searchResults} />);
        }
      }
    }
  };

  const handleChatModelStreamEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    if (event.event === "on_chat_model_start") {
      ensureTextStream(event.run_id, fields);
      return;
    }
    if (event.event === "on_chat_model_end") {
      const text = extractTextFromChatEnd(event);
      if (text) {
        const textStream = ensureTextStream(event.run_id, fields);
        if (textStream && "append" in textStream) {
          textStream.append(text);
          hasAssistantOutput = true;
        }
      }
      const textStream = fields.callbacks[event.run_id];
      if (textStream && "done" in textStream) {
        textStream.done();
      }
      return;
    }
    if (
      event.event !== "on_chat_model_stream" ||
      !event.data.chunk ||
      typeof event.data.chunk !== "object"
    ) {
      return;
    }
      
    // Filter out potential noise
    if (!event.data.chunk.content) return;

    const textStream = ensureTextStream(event.run_id, fields);
    if (textStream && "append" in textStream) {
      textStream.append(event.data.chunk.content);
      hasAssistantOutput = true;
    }
  };

  const handleSynthesizerEndEvent = (
    event: StreamEvent,
    fields: EventHandlerFields,
  ) => {
    if (event.event !== "on_chain_end" || event.name !== "synthesizer") return;
    if (hasAssistantOutput) return;

    const output: any = event.data?.output;
    const responseText =
      output?.draft_response ||
      output?.answer ||
      output?.final_answer ||
      null;
    if (!responseText) return;

    const textStream = ensureTextStream(event.run_id, fields);
    if (textStream && "append" in textStream) {
      textStream.append(responseText);
      hasAssistantOutput = true;
    }
    if (textStream && "done" in textStream) {
      textStream.done();
    }
  };

  return streamRunnableUI(
    remoteRunnable,
    {
      input: inputs.input,
      chat_history: inputs.chat_history,
      file: inputs.file,
    },
    {
      eventHandlers: [
        handleChainEvent,
        handleChatModelStreamEvent,
        handleSynthesizerEndEvent,
      ],
    },
  );
}

export const EndpointsContext = exposeEndpoints({ agent });
