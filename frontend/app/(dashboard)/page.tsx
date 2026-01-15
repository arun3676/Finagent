import { ChatInterface } from "@/components/chat/chat-interface";
import { ErrorBoundary } from "@/components/error-boundary";

export default function ChatPage() {
  return (
    <ErrorBoundary>
      <div className="h-full">
        <ChatInterface />
      </div>
    </ErrorBoundary>
  );
}
