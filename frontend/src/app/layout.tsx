import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import "bootstrap-icons/font/bootstrap-icons.css";

// App-wide providers (theme, SWR, notifications, tasks, etc.)
import { Providers } from "./providers";

// Layout components
import Navigation from "@/components/layout/Navigation";
import Breadcrumbs from "@/components/layout/Breadcrumbs";
import ErrorBoundary from "@/components/layout/ErrorBoundary";

// UI utilities
import { Toaster } from "@/components/ui/Toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AlphaDetect – Advanced Pose Detection",
  description:
    "Modern end-to-end pose-detection platform powered by AlphaPose (CLI ✦ FastAPI ✦ Next.js).",
  keywords: [
    "pose detection",
    "AlphaPose",
    "computer vision",
    "FastAPI",
    "Next.js",
  ],
  authors: [{ name: "AlphaDetect Team" }],
  viewport: "width=device-width, initial-scale=1",
  icons: {
    icon: "/favicon.ico",
    apple: "/apple-icon.png",
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
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100`}
      >
        {/* Skip-link for keyboard users */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only absolute top-2 left-2 z-50 rounded bg-blue-600 px-4 py-2 text-white"
        >
          Skip to content
        </a>

        <Providers>
          {/* Sticky header */}
          <header className="sticky top-0 z-40 w-full border-b bg-white/90 backdrop-blur dark:bg-gray-800/90">
            <div className="container mx-auto px-4">
              <Navigation />
            </div>
          </header>

          {/* Main content */}
          <main
            id="main-content"
            className="container mx-auto flex-1 px-4 py-6"
          >
            <ErrorBoundary>
              <div className="mb-6">
                <Breadcrumbs />
              </div>
              <div className="rounded-lg border bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
                {children}
              </div>
            </ErrorBoundary>
          </main>

          {/* Footer */}
          <footer className="border-t bg-white py-6 dark:bg-gray-800">
            <div className="container mx-auto px-4 text-center text-sm text-gray-500 dark:text-gray-400">
              <p>© {new Date().getFullYear()} AlphaDetect. All rights reserved.</p>
              <p className="mt-2">
                Powered by{" "}
                <a
                  href="https://github.com/MVIG-SJTU/AlphaPose"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:underline dark:text-blue-400"
                >
                  AlphaPose
                </a>
              </p>
            </div>
          </footer>

          {/* Global toast notifications */}
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
