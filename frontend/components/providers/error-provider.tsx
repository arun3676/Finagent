"use client";

import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { AlertCircle, X } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

interface ErrorContextType {
  showError: (message: string, title?: string) => void;
  clearError: () => void;
}

const ErrorContext = createContext<ErrorContextType | undefined>(undefined);

export function useError() {
  const context = useContext(ErrorContext);
  if (!context) {
    throw new Error("useError must be used within ErrorProvider");
  }
  return context;
}

interface ErrorProviderProps {
  children: ReactNode;
}

export function ErrorProvider({ children }: ErrorProviderProps) {
  const [error, setError] = useState<{ message: string; title?: string } | null>(null);

  const showError = useCallback((message: string, title?: string) => {
    setError({ message, title });
    setTimeout(() => setError(null), 10000);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return (
    <ErrorContext.Provider value={{ showError, clearError }}>
      {children}
      
      {error && (
        <div className="fixed bottom-4 right-4 z-50 max-w-md animate-in slide-in-from-bottom-5">
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>{error.title || "Error"}</AlertTitle>
            <AlertDescription>{error.message}</AlertDescription>
            <Button
              variant="ghost"
              size="icon"
              className="absolute top-2 right-2 h-6 w-6"
              onClick={clearError}
            >
              <X className="h-4 w-4" />
            </Button>
          </Alert>
        </div>
      )}
    </ErrorContext.Provider>
  );
}
