import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AWE - Agentic Web Explorer | Autonomous Web Extraction Framework",
  description: "A production-grade, generalizable multi-agent framework for autonomous web exploration and data extraction. Works with small language models through Tree of Thought reasoning.",
  keywords: ["web scraping", "autonomous agents", "AI", "LLM", "data extraction", "Playwright", "web exploration"],
  openGraph: {
    title: "AWE - Agentic Web Explorer",
    description: "Autonomous web extraction at scale with Tree of Thought reasoning",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
        suppressHydrationWarning
      >
        {children}
      </body>
    </html>
  );
}
