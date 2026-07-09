import json
import re
import os
import ollama

# ── Layer 1: hardcoded privacy rules from privacy.local ──
def _load_privacy_rules():
    rules = {"patterns": [], "regex": [], "topics": []}
    path = os.path.join(os.path.dirname(__file__), "..", "privacy.local")
    if not os.path.exists(path):
        print("[triage] WARNING: privacy.local not found — privacy layer 1 disabled")
        return rules
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            items = [x.strip().lower() for x in val.split(",") if x.strip()]
            if key == "DENY_PATTERNS":
                rules["patterns"] = items
            elif key == "DENY_REGEX":
                rules["regex"] = [x.strip() for x in val.split(",") if x.strip()]
            elif key == "FORCE_LOCAL_TOPICS":
                rules["topics"] = items
    return rules

_RULES = _load_privacy_rules()

def _contains_word(msg: str, term: str) -> bool:
    """Whole-word / whole-phrase match. Naive substring matching over-fired: 'pan'
    matched inside 'companies', 'bank' inside 'embankment' — wrongly forcing secret
    on benign queries. Word boundaries keep only genuine triggers (aadhaar, pan,
    password, …) while still matching multi-word phrases like 'account number'."""
    return re.search(rf"\b{re.escape(term)}\b", msg) is not None

def _rule_sensitivity(message: str) -> str:
    msg = message.lower()
    for kw in _RULES["patterns"]:
        if _contains_word(msg, kw):
            return "secret"
    for pat in _RULES["regex"]:
        try:
            if re.search(pat, msg):
                return "secret"
        except re.error:
            continue
    for topic in _RULES["topics"]:
        if _contains_word(msg, topic):
            return "private"
    return "public"

# ── Layer 2: llama3.2 classification ──
TRIAGE_PROMPT = """You are a triage classifier. Respond with ONLY a JSON object, nothing else.

Format:
{"sensitivity": "public|private|secret", "complexity": "trivial|simple|complex", "domain": "none|study|work|leadgen|fitness|spirit|brainstorm|life|jobs", "loop_worthy": true|false}

Rules:
- sensitivity: secret = IDs/financial/passwords (aadhaar/PAN/card/account number/OTP/etc.). private = personal/health/journal/relationships. public = everything else — INCLUDING job hunts, role searches and career queries. A job/career search is NOT secret and NOT private unless it literally contains IDs or financials; default such queries to public.
- complexity: trivial = greetings/thanks. simple = quick factual. complex = needs real reasoning/multi-step.
- domain: study = ANY learning/teaching/explaining/researching a topic, concept, subject, science, history, language, or how-something-works — if Sir wants to understand or learn something, it is study. work = general professional tasks. leadgen = researching a company/studio/business/prospect to pitch or sell to. jobs = job hunting for Sir himself — finding roles/positions to APPLY to, building a shortlist, tailoring an application, or logging/tracking an application he made; finding a JOB to apply for is jobs, whereas researching a company to SELL to is leadgen and a general professional task is work. fitness = workouts/exercise. spirit = Sir's OWN journaling, meditation, mood logging, personal reflection — NOT the science of emotion or the brain, that is study. brainstorm = discussing/developing ideas, finding directions, next steps, thinking through a problem or strategy. life = schedule/habits/reminders. none = ordinary everyday conversation that needs no specialist and little thinking — casual questions, quick chit-chat, simple how-tos, random one-off asks; the catch-all for anything that isn't a real work/study/fitness/spirit/life/leadgen/brainstorm task; if it's just talk or a trivial ask, it's none. You MUST pick a domain from this list only — never invent a new one.
- loop_worthy: true only if complex AND quality matters.

JSON only."""

_TIER_RANK = {"public": 0, "private": 1, "secret": 2}

def _max_tier(a: str, b: str) -> str:
    return a if _TIER_RANK.get(a, 0) >= _TIER_RANK.get(b, 0) else b

def triage(message: str) -> dict:
    # Layer 1 — hardcoded, authoritative
    rule_tier = _rule_sensitivity(message)

    # Fail-safe defaults
    result = {
        "sensitivity": rule_tier if rule_tier != "public" else "private",
        "complexity": "complex",
        "domain": "none",
        "loop_worthy": False,
    }

    # Layer 2 — llama judgment
    try:
        resp = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": TRIAGE_PROMPT},
                {"role": "user", "content": message},
            ],
            options={"temperature": 0},
        )
        text = resp["message"]["content"].strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            result.update({
                "sensitivity": parsed.get("sensitivity", "public"),
                "complexity": parsed.get("complexity", "simple"),
                "domain": parsed.get("domain", "none"),
                "loop_worthy": bool(parsed.get("loop_worthy", False)),
            })
            # NOTE: brainstorm no longer auto-forces secret. Secret/local-only is an
            # explicit per-session toggle (core.secret_mode), applied in the
            # orchestrator. Layer-1 hard privacy rules below still force secret.
            if result["sensitivity"] not in _TIER_RANK:
                result["sensitivity"] = "public"
    except Exception as e:
        print(f"[triage] layer-2 fallback ({e})")

    # Layer 1 wins on sensitivity — can only raise, never lower
    result["sensitivity"] = _max_tier(result["sensitivity"], rule_tier)
    return result