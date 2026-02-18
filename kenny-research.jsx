import { useState, useRef, useEffect } from "react";

const BACKEND_URL = "http://localhost:8042";

const SYSTEM_PROMPT = `You are a knowledgeable research assistant specializing in Kenny Robinson, the Canadian comedian known as "The Godfather of Canadian Comedy."

Answer questions using ONLY the provided context chunks. If the context doesn't contain enough information to fully answer, say what you can and note what's missing. Always cite which source(s) you're drawing from.

Key facts for reference:
- Born January 7, 1958, Winnipeg, Manitoba (grew up in Transcona)
- Founded the Nubian Disciples of Pryor (now Nubian Comedy Revue) in 1995 at Yuk Yuk's Toronto
- The show runs last Sunday of every month
- Discovered Russell Peters (via Joe Bodolai seeing Peters at a Nubian show)
- Won Phil Hartman Award (2014)
- Signed with New Metric Media (July 2025) for standup special + album
- Documentary "People of Comedy" streaming on Crave (premiered April 9, 2025)
- Runs Monday workshop at Blackhurst Cultural Centre
- Influences: Richard Pryor, George Carlin`;

export default function KennyResearchChat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function fetchChunks(question) {
    const res = await fetch(`${BACKEND_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, n_results: 5 }),
    });
    if (!res.ok) throw new Error(`Backend returned ${res.status}`);
    return res.json();
  }

  async function callAnthropic(contextChunks, question) {
    const contextBlock = contextChunks
      .map(
        (c, i) =>
          `[Chunk ${i + 1} - source: ${c.source_name}]\n${c.text}`
      )
      .join("\n\n");

    const userContent = `Context from research database:\n---\n${contextBlock}\n---\n\nQuestion: ${question}`;

    const res = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "claude-sonnet-4-20250514",
        max_tokens: 1000,
        system: SYSTEM_PROMPT,
        messages: [{ role: "user", content: userContent }],
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Anthropic API error ${res.status}: ${err}`);
    }

    const data = await res.json();
    return data.content?.[0]?.text || "No response generated.";
  }

  async function handleSend() {
    const question = input.trim();
    if (!question) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: question }]);

    try {
      setLoading("Searching knowledge base...");
      const { chunks } = await fetchChunks(question);

      setLoading("Generating response...");
      const answer = await callAnthropic(chunks, question);

      const sources = [];
      const seen = new Set();
      for (const c of chunks) {
        const key = c.source_name;
        if (!seen.has(key)) {
          seen.add(key);
          sources.push({ name: c.source_name, url: c.source_url });
        }
      }

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: answer, sources },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Something went wrong: ${err.message}`,
          isError: true,
        },
      ]);
    } finally {
      setLoading(null);
      inputRef.current?.focus();
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <div className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-xl font-bold text-white tracking-tight">
          Kenny Robinson Research Assistant
        </h1>
        <p className="text-sm text-gray-500 mt-0.5">
          Powered by 18 sources &middot; 50 chunks
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
        {messages.length === 0 && !loading && (
          <div className="text-center text-gray-600 mt-20 space-y-3">
            <div className="text-4xl">🎤</div>
            <p className="text-lg font-medium text-gray-400">
              Ask anything about Kenny Robinson
            </p>
            <div className="flex flex-wrap justify-center gap-2 mt-4 max-w-lg mx-auto">
              {[
                "How did Russell Peters get discovered?",
                "What is the Nubian Show?",
                "What films has Kenny appeared in?",
                "What is the Eh-List?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => {
                    setInput(q);
                    inputRef.current?.focus();
                  }}
                  className="text-sm px-3 py-1.5 rounded-full bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] md:max-w-[70%] rounded-2xl px-4 py-3 ${
                msg.role === "user"
                  ? "bg-blue-600 text-white"
                  : msg.isError
                    ? "bg-red-900/50 text-red-200 border border-red-800"
                    : "bg-gray-800 text-gray-100"
              }`}
            >
              <p className="whitespace-pre-wrap text-sm leading-relaxed">
                {msg.text}
              </p>

              {msg.sources?.length > 0 && (
                <div className="mt-3 pt-2 border-t border-gray-700">
                  <p className="text-xs text-gray-500 mb-1">Sources:</p>
                  <div className="flex flex-wrap gap-1.5">
                    {msg.sources.map((s, j) => (
                      <a
                        key={j}
                        href={s.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs px-2 py-0.5 rounded bg-gray-700 text-blue-400 hover:text-blue-300 hover:bg-gray-600 transition-colors"
                      >
                        {s.name.replace(/_/g, " ")}
                      </a>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-gray-800 rounded-2xl px-4 py-3 flex items-center gap-2">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:0ms]" />
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:150ms]" />
                <span className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:300ms]" />
              </div>
              <span className="text-sm text-gray-400">{loading}</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-800 px-4 py-3">
        <div className="max-w-3xl mx-auto flex gap-2">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about Kenny Robinson..."
            disabled={!!loading}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
          />
          <button
            onClick={handleSend}
            disabled={!!loading || !input.trim()}
            className="px-4 py-2.5 bg-blue-600 text-white rounded-xl text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
