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

def _rule_sensitivity(message: str) -> str:
    msg = message.lower()
    for kw in _RULES["patterns"]:
        if kw in msg:
            return "secret"
    for pat in _RULES["regex"]:
        try:
            if re.search(pat, msg):
                return "secret"
        except re.error:
            continue
    for topic in _RULES["topics"]:
        if topic in msg:
            return "private"
    return "public"

# ── Layer 2: llama3.2 classification ──
TRIAGE_PROMPT = """You are a triage classifier. Respond with ONLY a JSON object, nothing else.

Format:
{"sensitivity": "public|private|secret", "complexity": "trivial|simple|complex", "domain": "none|study|work|leadgen|fitness|spirit|kitchen|life", "loop_worthy": true|false}

Rules:
- sensitivity: secret = IDs/financial/passwords. private = personal/health/journal/relationships. public = everything else.
- complexity: trivial = greetings/thanks. simple = quick factual. complex = needs real reasoning/multi-step.
- domain: which area, or none for general chat.
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
    except Exception as e:
        print(f"[triage] layer-2 fallback ({e})")

    # Layer 1 wins on sensitivity — can only raise, never lower
    result["sensitivity"] = _max_tier(result["sensitivity"], rule_tier)
    return result