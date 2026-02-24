import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Toaster } from "sonner";
import { DataProvider } from "@/context/DataContext";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";
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
  title: "Analytico - AI Data Visualization",
  description: "Upload CSV files and generate charts with natural language using AI",
  keywords: ["data visualization", "charts", "AI", "CSV", "analytics"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-background font-sans antialiased`}
      >
        <DataProvider>
          {children}
          <Toaster 
            position="top-right"
            toastOptions={{
              style: {
                background: 'hsl(var(--card))',
                border: '1px solid hsl(var(--border))',
                color: 'hsl(var(--foreground))',
              },
            }}
          />
        </DataProvider>
      </body>
    </html>
  );
}
