import sqlite3
from datetime import datetime, timedelta
from config import Config

ONBOARDING_QUESTIONS = [
    "What time do you usually start your day, Sir?",
    "What are you working toward right now?",
    "What should I always remember about you?",
    "What should I never interrupt you for?",
    "How do you prefer I deliver information — brief summaries or full detail?",
    "Who are the key people in your life I should know about?",
    "What are your current projects outside of AIOS?"
]

def is_onboarding_active() -> bool:
    conn = sqlite3.connect(Config.SQLITE_PATH)
    cursor = conn.execute("SELECT asked_at FROM onboarding ORDER BY asked_at ASC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return True  # no questions asked yet — onboarding active

    first_question_date = datetime.fromisoformat(row[0])
    return datetime.now() < first_question_date + timedelta(days=7)

def get_next_question() -> str | None:
    if not is_onboarding_active():
        return None

    conn = sqlite3.connect(Config.SQLITE_PATH)
    cursor = conn.execute("SELECT question_asked FROM onboarding")
    asked = {row[0] for row in cursor.fetchall()}
    conn.close()

    for question in ONBOARDING_QUESTIONS:
        if question not in asked:
            return question

    return None  # all questions asked

def record_question(question: str, answer: str):
    conn = sqlite3.connect(Config.SQLITE_PATH)
    conn.execute(
        "INSERT INTO onboarding (question_asked, answer) VALUES (?, ?)",
        (question, answer)
    )
    conn.commit()
    conn.close()

def get_pending_question() -> str | None:
    if not is_onboarding_active():
        return None

    conn = sqlite3.connect(Config.SQLITE_PATH)
    cursor = conn.execute(
        "SELECT question_asked FROM onboarding WHERE answer IS NULL OR answer = ''"
    )
    row = cursor.fetchone()
    conn.close()

    return row[0] if row else None