# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AIOS is a single-user, locally-run personal AI assistant for one person ("Prithvi"). It is a Flask backend (`backend/`) that fronts LLM providers plus a persistent memory layer, with two clients: a terminal REPL (`backend/chat.py`) and a Next.js web UI (`desktop/`). There is no multi-tenancy — auth identifies a single hardcoded user and all data is keyed by `session_id` (default `"main"`).

## Commands

Backend (run from `backend/`):
```bash
python -m venv venv && venv\Scripts\activate   # Windows; use source venv/bin/activate on POSIX
pip install -r requirements.txt
python main.py                                  # serves on 0.0.0.0:5000, debug on unless FLASK_ENV=production
python chat.py                                  # terminal client (requires server already running)
```

Frontend (run from `desktop/`):
```bash
npm install
npm run dev      # Next.js dev server (Turbopack)
npm run build
npm start
```

There is **no test suite, linter config, or CI** in this repo. Do not invent commands for them.

### Setup gotchas
- `requirements.txt` does NOT list `groq` or `cerebras-cloud-sdk`, but `core/router.py` and `core/extractor.py` import them at module load. The server will not start without `pip install groq cerebras-cloud-sdk`. Add them to `requirements.txt` if you touch dependencies.
- Copy `backend/.env.example` to `backend/.env` and fill keys. `GROQ_API_KEY` is required for any chat or fact-extraction to work; `CEREBRAS_API_KEY` is the fallback. The other keys (Google AI, Anthropic, ElevenLabs, Picovoice, Tavily) are declared in `config.py` but not yet wired to any code.
- `python main.py` is invoked from inside `backend/`, so the default relative paths (`./db/aios.db`, `./db/chroma`) resolve against that cwd. Running from the repo root will create databases in the wrong place.

## Architecture

Request flow for a chat turn (`POST /chat` in `backend/api/chat_routes.py`):
1. Optional explicit `fact` in the payload is stored to long-term memory immediately.
2. If the message answers a pending onboarding question, it is recorded and stored as a `preference` fact.
3. Recent conversation `history` is loaded from SQLite (`core/memory.py`).
4. `core/router.py:chat()` retrieves relevant facts from ChromaDB (`core/profile.py:get_relevant_facts`), injects them into the system prompt, and calls providers.
5. Both the user message and assistant reply are persisted to SQLite.
6. `core/extractor.py:extract_and_store()` runs an LLM pass to mine new durable facts from the exchange and writes them to ChromaDB.
7. If onboarding is still active, the next unasked question is appended to the response as `onboarding_question`.

### Memory model — two stores, distinct roles
- **SQLite** (`db/aios.db`, schema in `db/sqlite_init.py`): transactional/episodic data — `conversations` (full chat log), `onboarding`, plus `sessions`, `reminders`, `jobs`, `leads`, `habits` tables that are **defined in the schema but not yet used by any code**. Treat those as the planned roadmap, not live features.
- **ChromaDB** (`db/chroma/`, init in `db/chroma_init.py`): vector store for semantic recall. Collection `long_term_memory` is the only one queried; `study_documents` and `lead_profiles` are created but unused. Facts carry a `category` metadata tag (`preference`, `habit`, `goal`, `person`, `project`, `constraint`, `general`).

The split matters: SQLite is "what was said," Chroma is "what to remember." `get_relevant_facts` does a semantic query against Chroma on every turn to build context; raw history comes from SQLite.

### Provider routing
`core/router.py` tries providers in a fixed order (Groq `llama-3.3-70b-versatile`, then Cerebras `qwen-3-32b`), returning the first success along with `provider_used` and `tokens_used`. Add a new provider by writing a `_try_*` function and inserting it into the list in `chat()`. The fact extractor (`core/extractor.py`) is hardcoded to Groq and fails silently — it must never break the main chat flow.

### Identity / system prompt
The assistant persona lives entirely in `SYSTEM_PROMPT` in `core/router.py`: terse, formal, addresses the user as "Sir." If a change touches tone or behavior of responses, that constant is the source of truth.

### Auth — note the gap
`core/auth.py` issues a 365-day JWT (`GET /token`) and provides a `require_auth` decorator, but **`/chat` does not currently apply it** — the endpoint is unprotected despite clients sending a Bearer token. `JWT_SECRET` defaults to `"change_this"`. If you add protection, decorate the route in `chat_routes.py`.

### Frontend
`desktop/` is Next.js 16 + React 19 + Tailwind v4, App Router. The entire UI is `app/page.tsx` — a client component that fetches a token from `http://127.0.0.1:5000/token` on mount and POSTs to `/chat`. The backend URL is hardcoded; CORS is open (`CORS(app)` in `main.py`).

**Read `desktop/AGENTS.md` before writing frontend code**: this is Next.js 16, which has breaking changes from older versions — consult `node_modules/next/dist/docs/` rather than relying on prior Next.js knowledge.
