"""
Multi-turn conversation demo (stateless / OpenAI-style).

The server is stateless: just append every user / assistant turn to the
messages array on the client side and send the full history each call,
exactly like the real OpenAI API.

Start the server in another terminal first:
    python claude_api.py

Then run:
    python multi_turn_demo.py
"""
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-used",
)

# Maintain the conversation history on the client side
history = [
    {"role": "system",
     "content": "You are a concise assistant. Keep answers short."},
]


def chat(user_msg: str, model: str = "default") -> str:
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(model=model, messages=history)
    reply = r.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply


def main():
    turns = [
        "My name is Alice and I'm 28 years old.",
        "What did I just tell you my name was?",
        "Double my age and add 5. What's the result?",
        "Summarize everything you know about me so far.",
    ]
    for q in turns:
        print(f"\n[USER] {q}")
        print(f"[ASSISTANT] {chat(q)}")

    print(f"\n--- history now has {len(history)} messages ---")


if __name__ == "__main__":
    main()
