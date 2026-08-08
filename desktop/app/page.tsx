"use client";
import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { useToken } from "@/lib/token-context";
import { useSession } from "@/lib/session-context";
import { API_BASE } from "@/lib/api";
import Orb, { type OrbState } from "@/components/Orb";

// AIOS-only markdown rendering (deliverables — numbered lead/job lists, bold
// names, multi-line fields — otherwise collapse to a wall of text). No
// rehype-raw is used, so react-markdown never parses embedded HTML tags in the
// model's output as real markup — they render as inert escaped text, same as
// any other unsafe-looking string. remarkBreaks turns the single line breaks
// the backend puts between fields into real <br>s (bare CommonMark would
// otherwise join them back into one line). User messages stay plain text —
// this component is never used for the user role.
function AiosMarkdown({ content }: { content: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
      {content}
    </ReactMarkdown>
  );
}

type Mood = "warm" | "neutral" | "sharp" | "soft";

interface Message {
  role: "user" | "aios";
  content: string;
  mood?: Mood;
}

function moodClass(mood?: Mood): string {
  if (mood === "warm" || mood === "soft" || mood === "sharp") return `mood-${mood}`;
  return "";
}

export default function Home() {
  const { token } = useToken();
  const { sessionId } = useSession();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [onboardingQuestion, setOnboardingQuestion] = useState("");
  const [awaitingOnboarding, setAwaitingOnboarding] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  // Which session is live RIGHT NOW, readable from inside an awaited send() — the
  // closure's `sessionId` is whatever it was when the send started, which is exactly
  // the value we must not trust once New Chat has moved on.
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Restore recent turns for the active session on every mount (including
  // remounting after nav away and back — the page component unmounts on route
  // change), and again whenever New Chat swaps sessionId to a fresh id (that
  // fetch just comes back empty, which is exactly the blank slate New Chat
  // wants — the old session's turns stay in SQLite and surface via the
  // dashboard's Conversations view instead).
  //
  // No "already loaded this session" ref guard here — that pattern breaks
  // under React Strict Mode's double-invoke-on-mount (on by default for the
  // App Router): the first invoke's cleanup sets `cancelled`, and a ref guard
  // would block the second invoke from starting its own fetch, so the only
  // fetch that ever ran has its response discarded. The `cancelled` flag
  // below is already the correct/sufficient guard against the stale first
  // response — let the effect re-run on every mount.
  useEffect(() => {
    if (!token) return;
    setMessages([]);
    let cancelled = false;

    fetch(`${API_BASE}/conversations?session_id=${encodeURIComponent(sessionId)}&limit=30`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => r.json())
      .then((turns: { role: string; content: string; mood: Mood | null }[]) => {
        if (cancelled) return;
        const prior: Message[] = turns.map((t) => ({
          role: t.role === "user" ? "user" : "aios",
          content: t.content,
          mood: t.mood ?? undefined,
        }));
        setMessages((m) => [...prior, ...m]);
      })
      .catch(() => {}); // no history yet / offline — chat still works from empty state

    return () => {
      cancelled = true;
    };
  }, [token, sessionId]);

  const send = async () => {
    if (!input.trim() || !token) return;
    const userMsg = input.trim();
    const sentSession = sessionId; // the session this turn belongs to
    setInput("");
    setMessages((m) => [...m, { role: "user", content: userMsg }]);
    setLoading(true);

    const res = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        message: userMsg,
        session_id: sessionId,
        onboarding_answer: awaitingOnboarding,
        voice: voiceEnabled,
      }),
    });

    const data = await res.json();

    // A turn can take tens of seconds. If New Chat rotated the session while this
    // request was in flight, the reply belongs to the OUTGOING session — appending it
    // here would drop the previous chat's answer into the blank new one (that is the
    // "New Chat replays the previous answer" bug). It is already persisted server-side
    // under `sentSession`, so dropping it loses nothing: it shows up in the dashboard's
    // Conversations view, and again in this pane if that session is ever reloaded.
    if (sessionIdRef.current !== sentSession) {
      setLoading(false);
      return;
    }

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

  const orbState: OrbState = loading ? "speaking" : voiceEnabled ? "listening" : "idle";

  return (
    <main className="flex-1 min-h-0 flex overflow-hidden bg-bg text-fg">
      {/* orb rail */}
      <div className="relative w-[224px] shrink-0 border-r border-border overflow-hidden flex flex-col">
        <div className="flex-1 min-h-0">
          <Orb state={orbState} />
        </div>
        <div className="shrink-0 pb-[22px] flex justify-center">
          <button
            type="button"
            onClick={() => setVoiceEnabled((v) => !v)}
            aria-pressed={voiceEnabled}
            className={`voice-toggle ${voiceEnabled ? "on" : ""}`}
          >
            <span className="voice-dot" />
            <span className="voice-label">Voice</span>
          </button>
        </div>
      </div>

      {/* chat */}
      <div className="ambient-field relative flex-1 min-h-0 flex flex-col overflow-hidden">
        <div className="relative z-10 flex-1 min-h-0 overflow-y-auto no-scrollbar px-[22px] py-[22px] flex flex-col gap-3.5">
          {messages.map((m, i) => (
            <div
              key={i}
              className={`max-w-[80%] flex flex-col gap-1.5 ${
                m.role === "user" ? "self-end items-end" : "self-start items-start"
              }`}
            >
              {m.role === "aios" && (
                <span className="text-[10px] font-semibold tracking-[0.06em] text-fg-2 px-[3px]">AIOS</span>
              )}
              <div className={m.role === "user" ? "bubble bubble-user" : `bubble bubble-aios ${moodClass(m.mood)}`}>
                {m.role === "user" ? m.content : <AiosMarkdown content={m.content} />}
              </div>
              {m.role === "aios" && m.mood && m.mood !== "neutral" && (
                <span className="text-[9px] font-medium tracking-[0.02em] text-fg-dim px-[3px]">{m.mood}</span>
              )}
            </div>
          ))}
          {loading && <div className="text-fg-dim text-sm px-[3px]">…</div>}
          <div ref={bottomRef} />
        </div>

        <div className="composer-pill m-3.5">
          <input
            className="flex-1 bg-transparent outline-none text-fg text-[13px] placeholder-fg-dim"
            placeholder="Speak…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            autoFocus
          />
          <button type="button" onClick={send} disabled={!input.trim()} className="raised-btn" aria-label="Send">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14M13 6l6 6-6 6" />
            </svg>
          </button>
        </div>
      </div>
    </main>
  );
}
