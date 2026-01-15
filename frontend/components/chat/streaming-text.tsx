"use client";

interface StreamingTextProps {
  text: string;
}

export function StreamingText({ text }: StreamingTextProps) {
  return (
    <span className="whitespace-pre-wrap">
      {text}
      <span className="inline-block w-2 h-4 ml-1 bg-primary animate-pulse rounded-sm" />
    </span>
  );
}
