from groq import Groq
from config import Config
from core.profile import store_fact
from core.vault import vault_store

client = Groq(api_key=Config.GROQ_API_KEY)

# Tiers for which we are allowed to call the cloud extractor. Anything else
# (secret, or an unexpected value) is fail-safe routed to the local vault.
_CLOUD_TIERS = ("public", "private")

EXTRACT_PROMPT = """You extract facts about Prithvi from conversations.

Rules:
- Only extract clear, specific facts about Prithvi himself
- Ignore generic statements, questions, or assistant responses
- Return ONLY a JSON array of objects like: [{"fact": "...", "category": "..."}]
- Categories: preference, habit, goal, person, project, constraint
- If nothing worth storing — return empty array: []
- Never invent facts. Only extract what is explicitly stated.
- Keep each fact concise — one sentence max."""

def extract_and_store(user_message: str, assistant_response: str, tier: str = "public"):
    # ── SECRET (or any non-cloud tier) → NEVER touch Groq. ──
    # The raw user message goes straight to the encrypted local vault: no cloud
    # round-trip, no embedding. This is the path that closes the leak.
    if tier not in _CLOUD_TIERS:
        try:
            vault_store(user_message, {"category": "secret_turn", "tier": "secret"})
        except Exception as e:
            # Don't break the chat flow; but make a vault failure visible rather
            # than silently dropping a secret. We still never fall back to cloud.
            print(f"[extractor] vault_store failed for secret turn: {e}")
        return

    # ── PUBLIC / PRIVATE → extract via Groq, tagging each fact with the tier. ──
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": f"User said: {user_message}\nAssistant replied: {assistant_response}"}
            ],
            max_tokens=256
        )
        import json
        text = response.choices[0].message.content.strip()
        facts = json.loads(text)
        for item in facts:
            if "fact" in item and "category" in item:
                store_fact(item["fact"], item["category"], tier=tier)
    except Exception:
        pass  # silent — never break the main flow
