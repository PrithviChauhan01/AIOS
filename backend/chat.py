import requests
import sys

BASE_URL = "http://127.0.0.1:5000"
SESSION_ID = "main"

def get_token():
    r = requests.get(f"{BASE_URL}/token")
    return r.json()["token"]

def chat(message, token, onboarding_answer=False):
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "message": message,
        "session_id": SESSION_ID,
        "onboarding_answer": onboarding_answer
    }
    r = requests.post(f"{BASE_URL}/chat", json=body, headers=headers)
    return r.json()

def main():
    token = get_token()
    print("AIOS online. Type to speak. Ctrl+C to exit.\n")

    answering_onboarding = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue

            result = chat(user_input, token, onboarding_answer=answering_onboarding)
            print(f"AIOS: {result['response']}\n")

            if "onboarding_question" in result:
                print(f"AIOS: {result['onboarding_question']}\n")
                answering_onboarding = True
            else:
                answering_onboarding = False

        except KeyboardInterrupt:
            print("\nOffline.")
            sys.exit(0)

if __name__ == "__main__":
    main()