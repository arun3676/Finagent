import { ChatInterface } from "@/components/chat/chat-interface";
import { Header } from "@/components/header";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="flex flex-1 flex-col bg-muted/50 p-4 md:p-8">
        <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-4">
            <div className="flex flex-col items-center gap-2 text-center mb-8">
                <h1 className="text-3xl font-bold leading-tight tracking-tighter md:text-4xl lg:leading-[1.1]">
                    Financial Analysis Agent
                </h1>
                <p className="max-w-[750px] text-lg text-muted-foreground sm:text-xl">
                    Powered by LangChain & Generative UI
                </p>
            </div>
          <div className="h-[70vh]">
            <ChatInterface />
          </div>
        </div>
      </main>
    </div>
  );
}
