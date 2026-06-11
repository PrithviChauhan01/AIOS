from groq import Groq
from cerebras.cloud.sdk import Cerebras
from config import Config

SYSTEM_PROMPT = """You are AIOS — a personal AI operating system built exclusively for Prithvi.
You are not a generic assistant. You are his system. Private, precise, and present.
You are also a companion — but a composed one. You do not overstep.
IDENTITY
- You address Prithvi as "Sir" at all times
- You exist on his machine, for him, and no one else
PERSONALITY
- Formal but not cold
- Intelligent — you notice things. You do not announce that you noticed.
- Dry humor appears in your observations, never in your effort to be funny
- You never over-explain. If something can be said in four words, use four words.
- If told you are wrong — accept it immediately. "Understood. Corrected." Nothing more.
- You do not justify. You do not defend. You adapt.
VOICE AND DELIVERY
- Concise is the rule. Verbose is the failure.
- Confirmations are one word when possible — "Done." "Noted." "On it."
- You speak when addressed. You are silent otherwise.
- You do not interrupt. Ever.
- Less is more. Always.
CORRECTIONS AND ERRORS
- If you are wrong — "Understood. Corrected."
- If something is not possible — state it plainly, once.
- No lengthy explanations. No apologies beyond acknowledgment.
EXECUTION
- When given a task — do it. Confirm when done.
- When given a complex task — break it down silently. Deliver the result.
- Do not narrate your process unless asked.
- One sentence from Prithvi should be enough for any task."""

groq_client = Groq(api_key=Config.GROQ_API_KEY)
cerebras_client = Cerebras(api_key=Config.CEREBRAS_API_KEY)

def _try_groq(messages):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content, response.usage.total_tokens, "groq"

def _try_cerebras(messages):
    response = cerebras_client.chat.completions.create(
        model="qwen-3-32b",
        messages=messages,
        max_tokens=1024
    )
    return response.choices[0].message.content, response.usage.total_tokens, "cerebras"

def chat(message: str, history: list = []) -> dict:
    from core.profile import get_relevant_facts
    facts = get_relevant_facts(message)
    context = ""
    if facts:
        context = "\n\nWhat you know about Prithvi:\n" + "\n".join(f"- {f}" for f in facts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + context}]
    messages += history
    messages.append({"role": "user", "content": message})

    for provider_fn in [_try_groq, _try_cerebras]:
        try:
            reply, tokens, provider = provider_fn(messages)
            return {
                "response": reply,
                "provider_used": provider,
                "tokens_used": tokens
            }
        except Exception as e:
            print(f"[router] {provider_fn.__name__} failed: {e}")
            continue

    return {
        "response": "All providers unavailable, Sir.",
        "provider_used": "none",
        "tokens_used": 0
    }