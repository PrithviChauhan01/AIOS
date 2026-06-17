"use client";
import { useState, useRef, useEffect } from "react";

type Mood = "warm" | "neutral" | "sharp" | "soft";

interface Message {
  role: "user" | "aios";
  content: string;
  mood?: Mood;
}

// Subtle per-mood tint for AIOS messages — thin left border + faint glow.
// Kept dark and minimal so it reads as a hint, not a highlight.
const MOOD_STYLES: Record<Mood, string> = {
  warm: "border-amber-500/60 shadow-[-4px_0_12px_-6px_rgba(245,158,11,0.5)]",
  neutral: "border-zinc-700",
  sharp: "border-red-500/60 shadow-[-4px_0_12px_-6px_rgba(239,68,68,0.5)]",
  soft: "border-blue-500/60 shadow-[-4px_0_12px_-6px_rgba(59,130,246,0.5)]",
};

function moodStyle(mood?: Mood): string {
  return MOOD_STYLES[mood ?? "neutral"];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [onboardingQuestion, setOnboardingQuestion] = useState("");
  const [awaitingOnboarding, setAwaitingOnboarding] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
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
        voice: voiceEnabled,
      }),
    });

    const data = await res.json();
    setMessages((m) => [...m, { role: "aios", content: data.response, mood: data.mood }]);

    if (data.onboarding_question) {
      setOnboardingQuestion(data.onboarding_question);
      setAwaitingOnboarding(true);
      setMessages((m) => [...m, { role: "aios", content: data.onboarding_question, mood: data.mood }]);
    } else {
      setAwaitingOnboarding(false);
      setOnboardingQuestion("");
    }

    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-black text-white flex flex-col">
      <div className="border-b border-zinc-800 px-6 py-4 flex items-center justify-between">
        <span className="text-xs tracking-widest text-zinc-500 uppercase">AIOS</span>
        <button
          onClick={() => setVoiceEnabled((v) => !v)}
          aria-pressed={voiceEnabled}
          title={voiceEnabled ? "Voice on" : "Voice off"}
          className={`transition ${
            voiceEnabled ? "text-white" : "text-zinc-600 hover:text-zinc-400"
          }`}
        >
          {voiceEnabled ? (
            // speaker with sound waves
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 5 6 9H2v6h4l5 4V5z" />
              <path d="M15.5 8.5a5 5 0 0 1 0 7" />
              <path d="M19 5a9 9 0 0 1 0 14" />
            </svg>
          ) : (
            // muted speaker
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M11 5 6 9H2v6h4l5 4V5z" />
              <line x1="22" y1="9" x2="16" y2="15" />
              <line x1="16" y1="9" x2="22" y2="15" />
            </svg>
          )}
        </button>
      </div>

      <div className="border-b border-zinc-800 px-6 py-4 flex gap-3">
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

      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-xl px-4 py-2 rounded text-sm ${
              m.role === "user"
                ? "bg-zinc-800 text-white"
                : `text-zinc-300 border-l-2 ${moodStyle(m.mood)}`
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
    </main>
  );
}
