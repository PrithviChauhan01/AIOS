from core.triage import triage

tests = [
    "hey",
    "what's my PAN number",
    "my password is hunter2",
    "explain project management from my notes",
    "research this design studio for outreach",
    "how am I feeling lately",
    "account number 123456789012",
]

for t in tests:
    print(f"\n{t!r}")
    print(triage(t))