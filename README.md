# claude-cli → OpenAI-compatible API

Wraps a locally-logged-in `claude` CLI as an OpenAI-compatible HTTP service. Adds concurrency, retry, streaming, and an optional session-based mode that keeps the prompt cache warm.

> 中文版: see [README.md](./README.md)

## Quick start

```bash
pip install -r requirements.txt
python claude_api.py            # serves http://127.0.0.1:8765
python parallel_client_demo.py  # in another terminal: parallel calls
python multi_turn_demo.py       # stateless multi-turn chat
python session_demo.py          # session-based multi-turn (cache-friendly)
```

Point any OpenAI SDK at it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="not-used")
r = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "hello"}],
)
print(r.choices[0].message.content)
```

---

## Choosing a model name

The `claude` CLI's `--model` flag accepts a few aliases plus full IDs. Straight from the [official docs](https://code.claude.com/docs/en/model-config):

| Model alias  | Behavior |
|---|---|
| `default`    | Special value that clears any model override and reverts to the recommended model for your account type. Not itself a model alias. |
| `best`       | Most capable available model, currently equivalent to `opus`. |
| `sonnet`     | Latest Sonnet model for daily coding tasks. |
| `opus`       | Latest Opus model for complex reasoning. |
| `haiku`      | Fast, cheap Haiku for simple tasks. |
| `sonnet[1m]` | Sonnet with a 1M-token context window for long sessions. |
| `opus[1m]`   | Opus with a 1M-token context window. |
| `opusplan`   | Uses Opus during plan mode, Sonnet for execution. |

You can also pass a **full model ID** (e.g. `claude-sonnet-4-6`) to pin a specific version. Anything you send is forwarded to `claude --model` as-is.

### Server-side conventions

To play nicely with OpenAI clients that *require* a `model` field, the server treats a few placeholder strings specially:

- `default` / `auto` / `claude` / `claude-cli` / empty string → **no** `--model` flag is added; the CLI uses your account default (matches the official `default` alias behaviour)
- Anything else is forwarded verbatim, validated by the CLI

Practical guidance:

- **Don't know what to pick** → `default`, follow your account
- **Want speed/cheap** → `haiku`
- **Want the strongest** → `opus` or `best`
- **Long conversations / big docs** → `sonnet[1m]`
- **Pin a specific snapshot** → use a full ID like `claude-sonnet-4-6`

You can also set a global default via env var so clients can keep sending `default`:

```bash
CLAUDE_DEFAULT_MODEL=sonnet[1m] python claude_api.py
```

To verify what your local CLI version actually accepts:

```bash
echo "ping" | claude -p --model sonnet --output-format json | head
# look at "is_error" in the response
```

---

## Multi-turn conversations

Two styles. **Method 1 is the default and matches OpenAI exactly.**

### Method 1: stateless — client keeps the history, sends full `messages` each call

The server is stateless. It concatenates your messages into a `User: ... / Assistant: ...` block and pipes it to the CLI. You just maintain history on the client side as you would with the real OpenAI API:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x")

history = [{"role": "system", "content": "You are a concise assistant."}]

def chat(user_msg: str) -> str:
    history.append({"role": "user", "content": user_msg})
    r = client.chat.completions.create(model="default", messages=history)
    reply = r.choices[0].message.content
    history.append({"role": "assistant", "content": reply})
    return reply

print(chat("My name is Alice"))
print(chat("What did I just tell you my name was?"))   # remembers
```

Pros:
- Clean protocol; works with every OpenAI client and framework (LangChain, LlamaIndex, open-webui, LobeChat, etc.)
- Stateless server, concurrency-friendly
- You control the history — easy to prune, fork, or splice in tool results

Cons: every call resends the full history, which gets expensive on long conversations. Also, the dialog history portion does **not** get prompt-cache hits (only the system prompt does), because the CLI sees one big user message rather than a real multi-message conversation.

Full runnable example: [`multi_turn_demo.py`](./multi_turn_demo.py).

### Method 2: pass a `session_id` — reuse the CLI session and prompt cache

The server supports an **optional extension field** `session_id`. When you pass it, the server takes the native `--session-id` / `--resume` path: history lives inside the CLI, only the latest user message goes over the wire each turn, and the prompt prefix hits cache reliably.

OpenAI Python SDK forwards arbitrary fields via `extra_body` (vanilla OpenAI clients won't even notice — the protocol stays compatible):

```python
import uuid
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="x")

SID = str(uuid.uuid4())   # one per conversation

def chat(user_msg: str) -> str:
    r = client.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": user_msg}],
        extra_body={"session_id": SID},        # <-- the key bit
    )
    return r.choices[0].message.content

print(chat("My name is Alice and I'm a backend engineer in SF"))
print(chat("Where did I say I work?"))         # CLI continues the session
```

Full runnable example: [`session_demo.py`](./session_demo.py). It prints `cache_read_input_tokens` / `cache_creation_input_tokens` per turn so you can see the cache benefit directly.

#### How the server handles `session_id`

- Request comes in with `session_id` → take only the **last user message** from `messages`, pass it as the prompt
- First time we've seen this session_id → invoke with `--session-id <id>` (plus `--system-prompt` if any)
- Subsequent calls with the same session_id → invoke with `--resume <id>`, **no** new `--system-prompt` (the session already has its own)
- If the CLI says the session doesn't exist (server restart, on-disk session cleared) → automatic fallback to recreating it
- Concurrent calls on the same session_id → serialized via a per-session `asyncio.Lock`. Different session_ids still run in parallel.

#### Constraints

1. **One session at a time**: the server serializes for you. If you want parallelism, use multiple session_ids.
2. **No history rewind / edits**: the CLI session is append-only. To regenerate or fork, start a new session_id.
3. **System prompt is fixed at session creation**: if later requests change the `system` content, the server ignores it (the CLI doesn't support swapping the system prompt mid-session).
4. **session_id must be a valid UUID**: required by `claude --session-id`. Just use `uuid.uuid4()`.

#### Without `session_id`, nothing changes

Existing clients still work exactly as before via the stateless path. You can mix the two: long, sequential conversations on session_id, one-off batch tasks stateless.

---

## Concurrency / retry / timeout tuning

All via env vars (restart to apply):

| Variable | Default | Description |
|---|---|---|
| `CLAUDE_MAX_CONCURRENCY` | 4 | Max simultaneous `claude` subprocesses. Lower it if your account hits rate limits often. |
| `CLAUDE_TIMEOUT` | 300 | Per-call timeout in seconds. Subprocess is killed on timeout. |
| `CLAUDE_MAX_RETRIES` | 5 | Retry attempts on failure / timeout. |
| `CLAUDE_BASE_BACKOFF` | 2.0 | Base for exponential backoff. Attempt n waits `2 * 2^(n-1)` seconds, capped at 60. |
| `CLAUDE_API_PORT` | 8765 | HTTP port. |
| `CLAUDE_DEFAULT_MODEL` | (none) | Model used when the client sends a placeholder. |
| `CLAUDE_BIN` | `claude` | Path to the CLI binary. |

```bash
CLAUDE_MAX_CONCURRENCY=2 CLAUDE_MAX_RETRIES=8 python claude_api.py
```

---

## Known limitations

- `temperature` / `top_p` / `max_tokens` and similar OpenAI knobs are **silently ignored** — the CLI doesn't expose them. The server accepts the fields purely for protocol compatibility.
- Streaming is *pseudo-streaming*: the server collects the full response from the CLI, then slices it into SSE chunks. Upside: retry stays effective during generation — a half-emitted response can never get stuck mid-flight. Downside: you don't see token-by-token output as the model generates.
- No auth. Bind to `127.0.0.1` only. If you need to expose it on a network, put a reverse proxy with a token in front.

Sources: [Claude Code model configuration](https://code.claude.com/docs/en/model-config)
