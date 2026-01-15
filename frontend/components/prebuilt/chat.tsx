"use client";

import React, { useState, useRef, useEffect } from "react";
import { Input } from "../ui/input";
import { Button } from "../ui/button";
import { Card } from "../ui/card";
import { EndpointsContext } from "@/app/ai/agent";
import { useActions } from "@/utils/client";
import { LocalContext } from "@/app/shared";
import { HumanMessageText } from "./message";
import { Send, Paperclip } from "lucide-react";
import { cn } from "../../utils/utils";

function convertFileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64String = reader.result as string;
      resolve(base64String.split(",")[1]); // Remove the data URL prefix
    };
    reader.onerror = (error) => {
      reject(error);
    };
    reader.readAsDataURL(file);
  });
}

function FileUploadMessage({ file }: { file: File }) {
  return (
    <div className="flex w-full max-w-fit ml-auto bg-muted p-2 rounded-md mb-2">
      <p className="text-sm text-muted-foreground flex items-center gap-2">
        <Paperclip className="h-4 w-4" />
        File uploaded: {file.name}
      </p>
    </div>
  );
}

export default function Chat() {
  const actions = useActions<typeof EndpointsContext>();

  const [elements, setElements] = useState<React.JSX.Element[]>([]);
  const [history, setHistory] = useState<[role: string, content: string][]>([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState<File>();
  const scrollRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [elements]);

  async function onSubmit(inputVal: string) {
    const newElements = [...elements];
    let base64File: string | undefined = undefined;
    let fileExtension = selectedFile?.type.split("/")[1];
    if (selectedFile) {
      base64File = await convertFileToBase64(selectedFile);
    }
    const element = await actions.agent({
      input: inputVal,
      chat_history: history,
      file:
        base64File && fileExtension
          ? {
              base64: base64File,
              extension: fileExtension,
            }
          : undefined,
    });

    newElements.push(
      <div className="flex flex-col w-full gap-1 mt-4" key={history.length}>
        {selectedFile && <FileUploadMessage file={selectedFile} />}
        <HumanMessageText content={inputVal} />
        <div className="flex flex-col gap-1 w-full max-w-fit mr-auto mt-2">
          {element.ui}
        </div>
      </div>,
    );

    // consume the value stream to obtain the final string value
    // after which we can append to our chat history state
    (async () => {
      try {
        let lastEvent = await element.lastEvent;
        
        // Handle LangGraph AgentState response
        if (lastEvent && typeof lastEvent === 'object' && !Array.isArray(lastEvent)) {
          // Cast to any to access properties that might not be on the inferred union type
          const eventData = lastEvent as any;
          // Check for draft_response (from AgentState) or answer (from QueryResponse if used)
          const response = eventData.draft_response || eventData.answer;
          
          if (response) {
            setHistory((prev) => [
              ...prev,
              ["human", inputVal],
              ["ai", response],
            ]);
            return;
          }
        }

        // Fallback for previous expected structures (legacy)
        if (Array.isArray(lastEvent)) {
          if (lastEvent[0].invoke_model && lastEvent[0].invoke_model.result) {
            setHistory((prev) => [
              ...prev,
              ["human", inputVal],
              ["ai", lastEvent[0].invoke_model.result],
            ]);
          } else if (lastEvent[1].invoke_tools) {
            setHistory((prev) => [
              ...prev,
              ["human", inputVal],
              [
                "ai",
                `Tool result: ${JSON.stringify(lastEvent[1].invoke_tools.tool_result, null)}`,
              ],
            ]);
          } else {
            setHistory((prev) => [...prev, ["human", inputVal]]);
          }
        } else if (lastEvent && lastEvent.invoke_model && lastEvent.invoke_model.result) {
          setHistory((prev) => [
            ...prev,
            ["human", inputVal],
            ["ai", lastEvent.invoke_model.result],
          ]);
        }
      } catch (e) {
        console.error("Stream processing failed:", e);
      }
    })();

    setElements(newElements);
    setInput("");
    setSelectedFile(undefined);
  }

  return (
    <Card className="flex flex-col h-[80vh] w-full border-border bg-card shadow-sm overflow-hidden">
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth"
      >
         {elements.length === 0 && (
            <div className="flex h-full flex-col items-center justify-center space-y-4 text-center p-8 text-muted-foreground opacity-50">
               <div className="p-6 rounded-full bg-muted/50">
                    <Send className="h-10 w-10" />
               </div>
               <p className="text-lg font-medium">Start a conversation</p>
            </div>
         )}
        <LocalContext.Provider value={onSubmit}>
          <div className="flex flex-col w-full gap-4">{elements}</div>
        </LocalContext.Provider>
      </div>
      
      <div className="p-4 bg-background border-t">
        <form
          onSubmit={async (e) => {
            e.stopPropagation();
            e.preventDefault();
            if (!input.trim()) return;
            await onSubmit(input);
          }}
          className="flex w-full items-center gap-2"
        >
          <Input
            placeholder="What's the latest stock price of NVDA?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="flex-1"
            autoFocus
          />
          <Button type="submit" size="icon" disabled={!input.trim()}>
            <Send className="h-4 w-4" />
            <span className="sr-only">Send</span>
          </Button>
        </form>
      </div>
    </Card>
  );
}
