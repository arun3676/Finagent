import "./globals.css";
import type { Metadata } from "next";

import { EndpointsContext } from "./ai/agent";
import { ReactNode } from "react";

export const metadata: Metadata = {
  title: "LangChain Gen UI",
  description: "Generative UI application with LangChain Python",
};

export default function RootLayout(props: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background font-sans antialiased">
        <EndpointsContext>{props.children}</EndpointsContext>
      </body>
    </html>
  );
}
