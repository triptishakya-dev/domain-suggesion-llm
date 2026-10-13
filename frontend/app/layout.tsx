import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Domain Suggestion AI",
  description: "Chat with a locally trained domain suggestion Transformer.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className="h-full antialiased"
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
