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
  title: "CogNexus | Deep Research Chat for Any Website",
  description: "Paste a website link and chat with a multi-agent researcher: grounded answers ranked by Tree-of-Thought confidence, internal structure, security audits with fixes, similar-site discovery and comparisons.",
  keywords: ["web scraping", "autonomous agents", "AI", "LLM", "data extraction", "Playwright", "web exploration"],
  openGraph: {
    title: "CogNexus | Deep Research Chat for Any Website",
    description: "Multi-agent, Tree-of-Thought deep research for any website",
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
        {/* Suppress errors from browser extensions (MetaMask, etc.) */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              window.addEventListener('error', function(e) {
                if (e.filename && e.filename.startsWith('chrome-extension://')) {
                  e.stopImmediatePropagation();
                  e.preventDefault();
                }
              });
              window.addEventListener('unhandledrejection', function(e) {
                var msg = e.reason && (e.reason.message || String(e.reason));
                if (msg && (msg.includes('MetaMask') || msg.includes('chrome-extension'))) {
                  e.stopImmediatePropagation();
                  e.preventDefault();
                }
              });
            `,
          }}
        />
        {children}
      </body>
    </html>
  );
}
