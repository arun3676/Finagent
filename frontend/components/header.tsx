import Link from "next/link";
import { Bot, FileText, Activity } from "lucide-react";
import { Button } from "./ui/button";

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 max-w-screen-2xl items-center">
        <div className="mr-4 flex">
          <Link className="mr-6 flex items-center space-x-2" href="/">
            <Bot className="h-6 w-6" />
            <span className="hidden font-bold sm:inline-block">
              Financial Agent
            </span>
          </Link>
          <nav className="flex items-center gap-6 text-sm">
            <Link
              href="/citations-demo"
              className="transition-colors hover:text-foreground/80 text-foreground/60 flex items-center gap-1"
            >
              <FileText className="h-4 w-4" />
              Citations
            </Link>
            <Link
              href="/observe"
              className="transition-colors hover:text-foreground/80 text-foreground/60 flex items-center gap-1"
            >
              <Activity className="h-4 w-4" />
              Trace Viewer
            </Link>
          </nav>
        </div>
        <div className="flex flex-1 items-center justify-end space-x-2">
           {/* Placeholder for future auth or settings */}
        </div>
      </div>
    </header>
  );
}
