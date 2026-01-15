"use client";

import { useState, KeyboardEvent } from "react";
import { Search, Plus, Loader2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { DocumentType } from "@/types/company";

interface TickerInputProps {
  onSubmit: (ticker: string, documentTypes: DocumentType[]) => void;
  isLoading?: boolean;
}

const DOCUMENT_TYPE_OPTIONS: { value: DocumentType; label: string }[] = [
  { value: "SEC_10K", label: "10-K (Annual Report)" },
  { value: "SEC_10Q", label: "10-Q (Quarterly Report)" },
  { value: "SEC_8K", label: "8-K (Current Report)" },
  { value: "EARNINGS_CALL", label: "Earnings Call Transcript" },
];

export function TickerInput({ onSubmit, isLoading = false }: TickerInputProps) {
  const [ticker, setTicker] = useState("");
  const [selectedTypes, setSelectedTypes] = useState<DocumentType[]>(["SEC_10K"]);
  const [error, setError] = useState<string | null>(null);

  const validateTicker = (value: string): boolean => {
    // Ticker symbols: 1-5 uppercase letters
    const tickerRegex = /^[A-Z]{1,5}$/;
    return tickerRegex.test(value.toUpperCase());
  };

  const handleSubmit = () => {
    const normalizedTicker = ticker.toUpperCase().trim();
    
    if (!normalizedTicker) {
      setError("Please enter a ticker symbol");
      return;
    }

    if (!validateTicker(normalizedTicker)) {
      setError("Invalid ticker format (1-5 letters, e.g., AAPL, MSFT)");
      return;
    }

    if (selectedTypes.length === 0) {
      setError("Please select at least one document type");
      return;
    }

    setError(null);
    onSubmit(normalizedTicker, selectedTypes);
    setTicker("");
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSubmit();
    }
  };

  const toggleDocType = (type: DocumentType) => {
    setSelectedTypes((prev) =>
      prev.includes(type)
        ? prev.filter((t) => t !== type)
        : [...prev, type]
    );
  };

  return (
    <div className="space-y-4 p-6 border rounded-lg bg-card">
      <div className="space-y-2">
        <h3 className="font-semibold text-lg">Add Company</h3>
        <p className="text-sm text-muted-foreground">
          Enter a stock ticker to index SEC filings and earnings calls
        </p>
      </div>

      {/* Ticker Input */}
      <div className="flex gap-2">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            value={ticker}
            onChange={(e) => {
              setTicker(e.target.value.toUpperCase());
              setError(null);
            }}
            onKeyDown={handleKeyDown}
            placeholder="Enter ticker (e.g., AAPL, NVDA)"
            className="pl-10 uppercase"
            disabled={isLoading}
            maxLength={5}
          />
        </div>
        <Button onClick={handleSubmit} disabled={isLoading || !ticker.trim()}>
          {isLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Plus className="h-4 w-4" />
          )}
          <span className="ml-2 hidden sm:inline">Add</span>
        </Button>
      </div>

      {/* Error Message */}
      {error && (
        <p className="text-sm text-destructive">{error}</p>
      )}

      {/* Document Type Selection */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Document Types to Index</label>
        <div className="flex flex-wrap gap-2">
          {DOCUMENT_TYPE_OPTIONS.map((option) => (
            <Button
              key={option.value}
              variant={selectedTypes.includes(option.value) ? "default" : "outline"}
              size="sm"
              onClick={() => toggleDocType(option.value)}
              disabled={isLoading}
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>

      {/* Help Text */}
      <div className="text-xs text-muted-foreground bg-muted/50 p-3 rounded">
        <p className="font-medium mb-1">💡 Tips:</p>
        <ul className="space-y-1 list-disc list-inside">
          <li>10-K filings contain annual financial data and risk factors</li>
          <li>10-Q filings have quarterly updates</li>
          <li>Earnings calls include management guidance and Q&A</li>
        </ul>
      </div>
    </div>
  );
}
