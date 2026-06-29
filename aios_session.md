# AIOS — Session Handoff

_Delta-style handoff: current state + what changed, not a full changelog. Architecture/setup detail lives in `CLAUDE.md`._

_Updated: 2026-06-28_

---

## Build state — DONE & live

- **Reminders + scheduler** — complete and verified. The earlier "partial / SQLITE_PATH split" diagnosis was wrong: the real gap was no action-execution layer (see below), not a DB mismatch. APScheduler ticks every 30s; delivery via `win11toast`, plus spoken delivery when the voice loop is live. Full `set / list / complete / delete` with natural-language time parsing.
  - _Op note:_ toasts need Windows notifications **ON** and Focus Assist **OFF**, or they silently don't appear.

- **Action-dispatch layer** — new architecture component. After triage, the orchestrator detects action intent, calls the **real** tool (`is_action`), and feeds the **real tool result** back into cognition so confirmations are truthful instead of fabricated. This is the shared pattern every action reuses (reminders, jobs, documents). **→ add to the LLD.**

- **Jobs teacher (income track)** — `domain=jobs`. Produces a deliverable role shortlist with resume routing (each role labeled **engineering** or **design**). `log_application` action writes to the `jobs` table. Roles are currently **reasoned, not live-scraped** (no job-board API yet).

- **Document storage (general — not resume-specific)** — `save / list / get / delete` for any document. Ingest: **PDF (verified) + plain text**; a `.docx` path is wired (`python-docx`) but unverified; **`.pptx` not supported**. Privacy-tiered at store time: public/private → Chroma (`mem_docs`), **secret → encrypted vault** (0 chunks, never embedded).

- **Privacy guard fix (important)** — private-tier retrieval was leaking content to cloud books. Now **both private and secret force local-only (ollama)**; only public reaches the cloud. Verified across save/get paths.

- **Cognition fast-lane routing** — live. Trivial / short-circuit turns → Groq `llama-3.1-8b-instant`; full/deliverable turns → `llama-3.3-70b`. Donna voice holds on the 8B path. Output caps: **60 tokens trivial / 900 deliverable+list** (900 after fixing mid-sentence truncation on multi-row shortlists).

---

## Voice — shipped, pending hardware

Pipeline is **complete and correct**, no further code work needed:
- Groq `whisper-large-v3-turbo` primary STT (~300–600ms), local `faster-whisper` fallback.
- TEN VAD, `min_silence=1100`; native-rate capture + resample to 16k; energy fallback; fast-model routing.

**Blocker is hardware only.** AirPods HFP mic over Windows Bluetooth gives dead/garbage input (level ~0.0009, "tape-recorder" quality) — unusable for STT. **USB mic ordered.**

➡️ **`chat.py` (text) is the reliable daily driver** until the mic lands.

---

## Key learnings

- **Action confirmations must come from the real tool result.** LLMs will happily fabricate "done" with no write behind it — build an explicit dispatch stage, don't assume one.
- **The cloud privacy guard must cover private AND secret**, not secret alone — private retrieval was the leak.
- **AirPods mic over Windows Bluetooth (HFP) is unusable for STT** — a hardware wall, not a code problem. Don't burn more time on capture code.

---

## Queue (priority order, unchanged)

1. **Resume matching** — consumes the document store.
2. **pptx/docx parsing** — finish/verify docx, add pptx.
3. **Live job-board search** — needs Tavily key; replaces reasoned shortlists with real listings.
4. **Email-send + confirm gate** — first outbound action; gate pattern in front of `is_action` tools.
5. **Async fan-out**.
6. **Dashboard**.
