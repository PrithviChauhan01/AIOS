"use client";
import { useState, useRef, useEffect } from "react";

interface Message {
  role: "user" | "aios";
  content: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [onboardingQuestion, setOnboardingQuestion] = useState("");
  const [awaitingOnboarding, setAwaitingOnboarding] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/token")
      .then((r) => r.json())
      .then((d) => setToken(d.token));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    if (!input.trim() || !token) return;
    const userMsg = input.trim();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setLoading(true);

    const res = await fetch("http://127.0.0.1:5000/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: userMsg,
        session_id: "main",
        onboarding_answer: awaitingOnboarding,
      }),
    });

    const data = await res.json();
    setMessages((m) => [...m, { role: "aios", content: data.response }]);

    if (data.onboarding_question) {
      setOnboardingQuestion(data.onboarding_question);
      setAwaitingOnboarding(true);
      setMessages((m) => [...m, { role: "aios", content: data.onboarding_question }]);
    } else {
      setAwaitingOnboarding(false);
      setOnboardingQuestion("");
    }

    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-black text-white flex flex-col">
      <div className="border-b border-zinc-800 px-6 py-4">
        <span className="text-xs tracking-widest text-zinc-500 uppercase">AIOS</span>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-xl px-4 py-2 rounded text-sm ${
              m.role === "user"
                ? "bg-zinc-800 text-white"
                : "text-zinc-300"
            }`}>
              {m.role === "aios" && (
                <span className="text-xs text-zinc-600 mr-2 uppercase tracking-widest">AIOS</span>
              )}
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-zinc-600 text-sm">...</div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="border-t border-zinc-800 px-6 py-4 flex gap-3">
        <input
          className="flex-1 bg-zinc-900 text-white text-sm px-4 py-2 rounded outline-none placeholder-zinc-600"
          placeholder="Speak..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          autoFocus
        />
        <button
          onClick={send}
          className="text-xs text-zinc-500 hover:text-white transition px-3"
        >
          Send
        </button>
      </div>
    </main>
  );
}