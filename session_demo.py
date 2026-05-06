"""
Session-based multi-turn demo. Uses the CLI's native --session-id /
--resume to reuse the prompt cache, saving tokens on long conversations.

Difference vs multi_turn_demo.py:

  * multi_turn_demo.py: client sends the full messages array every turn;
    the server concatenates it into one big prompt; only the system
    prompt benefits from prompt caching, the dialog history portion is
    re-prefilled every time.

  * session_demo.py: client passes a stable `session_id` via extra_body;
    the server only forwards the latest user message; the CLI manages
    history on its side and the prefix stays cache-friendly across
    turns.

Constraints:
  * The same session_id cannot be hit concurrently. The server enforces
    a per-session asyncio.Lock for serialization.
  * To regenerate, fork, or rewrite history, start a new session_id.

Start the server in another terminal first:
    python claude_api.py

Then run:
    python session_demo.py
"""
import time
import uuid

from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-used",
)

SID = str(uuid.uuid4())   # one session_id for the whole conversation
print(f"session id: {SID}")


def chat(user_msg: str) -> str:
    t0 = time.time()
    r = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": user_msg}],
        # session_id is this server's extension field. Pass it via the
        # OpenAI SDK's extra_body channel so the standard schema stays
        # untouched.
        extra_body={"session_id": SID},
    )
    dt = time.time() - t0
    u = r.usage
    cache_read = (
        getattr(u, "cache_read_input_tokens", None)
        or getattr(u, "model_extra", {}).get("cache_read_input_tokens", 0)
    )
    cache_create = (
        getattr(u, "cache_creation_input_tokens", None)
        or getattr(u, "model_extra", {}).get("cache_creation_input_tokens", 0)
    )
    print(f"  [{dt:.1f}s] in={u.prompt_tokens} out={u.completion_tokens}"
          f" cache_read={cache_read} cache_create={cache_create}")
    return r.choices[0].message.content


def main():
    turns = [
        # First turn seeds context. Later turns should benefit from
        # prompt cache hits on this prefix.
        "Please remember: my name is Alice, I'm 28, I love hiking, "
        "I work in San Francisco as a backend engineer.",
        "What is my name?",
        "What do I do for a living?",
        "Summarize everything you know about me.",
    ]
    for i, q in enumerate(turns, 1):
        print(f"\n[turn {i}] USER: {q}")
        print(f"[turn {i}] ASSISTANT: {chat(q)}")


if __name__ == "__main__":
    main()
