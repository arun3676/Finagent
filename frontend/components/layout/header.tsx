"use client";

import { useHealth } from "@/lib/hooks/use-health";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Settings, Menu, Circle } from "lucide-react";
import { useUIStore } from "@/lib/stores/ui-store";

export function Header() {
  const { health, isLoading, error } = useHealth();
  const { toggleSidebar } = useUIStore();

  const getStatusColor = () => {
    if (isLoading) return "text-muted-foreground";
    if (error || health?.status === "unhealthy") return "text-red-500";
    if (health?.status === "degraded") return "text-yellow-500";
    return "text-green-500";
  };

  const getStatusText = () => {
    if (isLoading) return "Connecting...";
    if (error) return "Disconnected";
    return health?.status === "healthy" ? "Connected" : health?.status || "Unknown";
  };

  return (
    <header className="h-14 border-b px-4 flex items-center justify-between bg-background">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" onClick={toggleSidebar} className="lg:hidden">
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex items-center gap-2">
          <h1 className="font-semibold">FinAgent Research</h1>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Backend Status */}
        <div className="flex items-center gap-2 text-sm">
          <Circle className={`h-2 w-2 fill-current ${getStatusColor()}`} />
          <span className="text-muted-foreground hidden sm:inline">
            Backend: {getStatusText()}
          </span>
        </div>

        {/* Component Status Badges */}
        {health && !isLoading && (
          <div className="hidden md:flex items-center gap-2">
            <Badge variant={health.components.qdrant === "up" ? "secondary" : "destructive"} className="text-xs">
              Qdrant {health.components.qdrant}
            </Badge>
            <Badge variant={health.components.llm === "up" ? "secondary" : "destructive"} className="text-xs">
              LLM {health.components.llm}
            </Badge>
          </div>
        )}

        <Button variant="ghost" size="icon">
          <Settings className="h-5 w-5" />
        </Button>
      </div>
    </header>
  );
}
