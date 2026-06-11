from groq import Groq
from config import Config
from core.profile import store_fact

client = Groq(api_key=Config.GROQ_API_KEY)

EXTRACT_PROMPT = """You extract facts about Prithvi from conversations.

Rules:
- Only extract clear, specific facts about Prithvi himself
- Ignore generic statements, questions, or assistant responses
- Return ONLY a JSON array of objects like: [{"fact": "...", "category": "..."}]
- Categories: preference, habit, goal, person, project, constraint
- If nothing worth storing — return empty array: []
- Never invent facts. Only extract what is explicitly stated.
- Keep each fact concise — one sentence max."""

def extract_and_store(user_message: str, assistant_response: str):
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
                store_fact(item["fact"], item["category"])
    except Exception:
        pass  # silent — never break the main flow