"use client";

import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from "react";

type Message = { id: string; role: "user" | "assistant"; content: string };

const examples = [
  "I build React apps with TypeScript and Tailwind CSS.",
  "I work with Python, SQL, data analysis, and dashboards.",
  "I enjoy cloud infrastructure, Docker, Kubernetes, and automation.",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  async function sendMessage(event?: FormEvent) {
    event?.preventDefault();
    const text = message.trim();
    if (!text || isLoading) return;
    setError("");
    setMessage("");
    setMessages((current) => [...current, { id: crypto.randomUUID(), role: "user", content: text }]);
    setIsLoading(true);
    try {
      const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text }) });
      const data: unknown = await response.json();
      if (!response.ok || typeof data !== "object" || data === null || !("reply" in data)) {
        const problem = typeof data === "object" && data !== null && "error" in data ? String(data.error) : "Something went wrong. Please try again.";
        throw new Error(problem);
      }
      setMessages((current) => [...current, { id: crypto.randomUUID(), role: "assistant", content: String(data.reply) }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to reach the model.");
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  }

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-6 text-slate-100 sm:px-8">
      <section className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-4xl flex-col overflow-hidden rounded-3xl border border-slate-700 bg-slate-900 shadow-2xl shadow-indigo-950/40">
        <header className="border-b border-slate-700 bg-slate-900/90 px-6 py-5">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-indigo-300">Local AI</p>
          <h1 className="mt-1 text-xl font-bold tracking-tight">Domain Suggestion AI</h1>
          <p className="mt-1 text-sm text-slate-400">Describe your skills or project. The Transformer suggests a matching domain.</p>
        </header>
        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-6">
          {messages.length === 0 && <div className="mx-auto mt-10 max-w-2xl text-center"><div className="mx-auto grid size-14 place-items-center rounded-2xl bg-indigo-500/15 text-2xl">✦</div><h2 className="mt-4 text-2xl font-semibold">What are you building?</h2><p className="mt-2 text-sm leading-6 text-slate-400">Share your strengths, tools, or project idea to get a domain suggestion.</p><div className="mt-6 grid gap-3 text-left sm:grid-cols-3">{examples.map((example) => <button key={example} onClick={() => setMessage(example)} className="rounded-xl border border-slate-700 bg-slate-800 p-3 text-left text-sm text-slate-300 transition hover:border-indigo-400 hover:bg-slate-800/70">{example}</button>)}</div></div>}
          {messages.map((item) => <article key={item.id} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}><div className={`max-w-[86%] whitespace-pre-wrap rounded-2xl px-4 py-3 text-sm leading-6 ${item.role === "user" ? "rounded-br-md bg-indigo-500 text-white" : "rounded-bl-md border border-slate-700 bg-slate-800 text-slate-100"}`}>{item.content}</div></article>)}
          {isLoading && <div className="w-fit rounded-2xl rounded-bl-md border border-slate-700 bg-slate-800 px-4 py-3 text-sm text-slate-300">Generating your suggestion<span className="animate-pulse">...</span></div>}
          <div ref={bottomRef} />
        </div>
        <form onSubmit={sendMessage} className="border-t border-slate-700 bg-slate-900 p-4 sm:p-5">
          {error && <p role="alert" className="mb-3 rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">{error}</p>}
          <div className="flex items-end gap-3 rounded-2xl border border-slate-600 bg-slate-800 p-2 focus-within:border-indigo-400"><textarea value={message} onChange={(event) => setMessage(event.target.value)} onKeyDown={handleKeyDown} maxLength={1000} rows={2} placeholder="Example: I build React apps with TypeScript..." className="min-h-12 flex-1 resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-slate-500" /><button type="submit" disabled={!message.trim() || isLoading} className="rounded-xl bg-indigo-500 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-40">Send</button></div>
          <p className="mt-2 px-2 text-xs text-slate-500">Press Enter to send · Shift + Enter for a new line</p>
        </form>
      </section>
    </main>
  );
}
