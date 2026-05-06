"""
Wrap the locally-logged-in `claude` CLI as an OpenAI-compatible HTTP API.

Endpoints:
  POST /v1/chat/completions   (supports stream=true via SSE)
  GET  /v1/models
  GET  /health

Features:
  * Async concurrency with asyncio.Semaphore
  * Exponential-backoff retry on failure / timeout
  * messages -> claude CLI prompt + --system-prompt conversion
  * Pseudo-streaming SSE (chunk slicing) compatible with all OpenAI clients
  * Optional `session_id` extension field that uses native --session-id /
    --resume so prompt cache stays warm on long conversations

Run:
  pip install fastapi uvicorn pydantic
  python claude_api.py            # default http://127.0.0.1:8765
"""
import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ---------- Configuration ----------
CLAUDE_BIN      = os.environ.get("CLAUDE_BIN", "claude")
MAX_CONCURRENCY = int(os.environ.get("CLAUDE_MAX_CONCURRENCY", "4"))
REQUEST_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", "300"))
MAX_RETRIES     = int(os.environ.get("CLAUDE_MAX_RETRIES", "5"))
BASE_BACKOFF    = float(os.environ.get("CLAUDE_BASE_BACKOFF", "2.0"))
MAX_BACKOFF     = 60.0
PORT            = int(os.environ.get("CLAUDE_API_PORT", "8765"))
DEFAULT_MODEL   = os.environ.get("CLAUDE_DEFAULT_MODEL") or None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("claude-api")

# Global concurrency gate
_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
# Set of session_ids we've already created. Used to decide between
# --session-id (create) vs --resume (continue).
_known_sessions: set = set()
# One asyncio.Lock per session so calls within the same session are
# serialized (the CLI's session file is not safe for concurrent writes).
_session_locks: Dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    return _session_locks.setdefault(session_id, asyncio.Lock())


# ---------- Model name resolution ----------
_PLACEHOLDER_MODELS = {"default", "auto", "claude", "claude-cli", ""}


def _resolve_model(m: Optional[str]) -> Optional[str]:
    """Decide whether to forward `model` to the claude CLI.

    - Empty / placeholder string -> None (no --model, use CLI default)
    - Anything else is forwarded as-is. The CLI itself validates aliases
      such as sonnet / opus / haiku / best / sonnet[1m] / opusplan, or
      full model IDs like claude-sonnet-4-6.
    """
    if not m:
        return None
    m = m.strip()
    if m.lower() in _PLACEHOLDER_MODELS:
        return None
    return m


# ---------- messages -> text helpers ----------
def _flatten_content(c: Any) -> str:
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        # OpenAI multimodal: [{"type":"text","text":"..."}, ...]
        return "\n".join(
            p.get("text", "") for p in c
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return str(c)


def _system_from_messages(messages: List[Dict[str, Any]]) -> Optional[str]:
    parts: List[str] = []
    for m in messages:
        if m.get("role") == "system":
            t = _flatten_content(m.get("content")).strip()
            if t:
                parts.append(t)
    return "\n\n".join(parts) if parts else None


def _last_user_message(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            t = _flatten_content(m.get("content")).strip()
            if t:
                return t
    raise ValueError("no usable user message in messages array")


def messages_to_prompt(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """Stateless path: concatenate the whole history into a single prompt
    and pull the system message out separately."""
    system_parts: List[str] = []
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "user")
        text = _flatten_content(m.get("content")).strip()
        if not text:
            continue
        if role == "system":
            system_parts.append(text)
        elif role == "user":
            lines.append(f"User: {text}")
        elif role == "assistant":
            lines.append(f"Assistant: {text}")
        else:
            lines.append(f"{role.capitalize()}: {text}")

    system = "\n\n".join(system_parts) if system_parts else None
    if len(lines) == 1 and lines[0].startswith("User: "):
        prompt = lines[0][len("User: "):]
    else:
        prompt = "\n\n".join(lines) + "\n\nAssistant:"
    return system, prompt


# ---------- claude CLI invocation ----------
_SESSION_LOST_KEYWORDS = (
    "not found", "no such", "does not exist",
    "could not find", "no session", "session.*lost",
)


async def _run_claude(args: List[str], stdin_text: str, timeout: int
                      ) -> Tuple[str, Dict[str, Any]]:
    """Run claude CLI once. Return (result_text, usage_dict)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(stdin_text.encode("utf-8")),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        raise TimeoutError(f"claude CLI timed out (>{timeout}s)")

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"claude CLI exit={proc.returncode}: {err[:500]}")

    raw = stdout.decode("utf-8", errors="replace").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}

    # The CLI may exit=0 but set is_error=true (e.g. rate limit, not
    # logged in). Surface that as RuntimeError so retry logic kicks in.
    if isinstance(data, dict) and data.get("is_error"):
        msg = data.get("result") or data.get("error") or "is_error=true"
        raise RuntimeError(f"claude API error: {msg}")

    text = data.get("result") or data.get("content") or data.get("text") or raw
    return text, (data.get("usage") or {})


async def _call_stateless(prompt: str, system: Optional[str],
                          model: Optional[str], timeout: int):
    args = [CLAUDE_BIN, "-p", "--output-format", "json"]
    if model:
        args += ["--model", model]
    if system:
        args += ["--system-prompt", system]
    return await _run_claude(args, prompt, timeout)


async def _call_session_attempt(last_msg: str, system: Optional[str],
                                model: Optional[str], timeout: int,
                                session_id: str, resuming: bool):
    """resuming=True -> --resume; resuming=False -> --session-id (create)."""
    args = [CLAUDE_BIN, "-p", "--output-format", "json"]
    if resuming:
        args += ["--resume", session_id]
        # When resuming, do NOT pass --system-prompt; the existing
        # session already carries the system prompt from when it was
        # created.
    else:
        args += ["--session-id", session_id]
        if system:
            args += ["--system-prompt", system]
    if model:
        args += ["--model", model]
    return await _run_claude(args, last_msg, timeout)


async def _call_session(last_msg: str, system: Optional[str],
                        model: Optional[str], timeout: int, session_id: str):
    """Session-aware call with a 'session lost' fallback: try --resume
    first; if the CLI says the session doesn't exist, retry as a fresh
    --session-id (this happens when the server restarts but the client
    keeps using its old session_id, or when the CLI's on-disk session
    file was cleared)."""
    is_known = session_id in _known_sessions
    try:
        result = await _call_session_attempt(
            last_msg, None if is_known else system,
            model, timeout, session_id, resuming=is_known,
        )
    except RuntimeError as e:
        msg = str(e).lower()
        if is_known and any(kw in msg for kw in _SESSION_LOST_KEYWORDS):
            log.warning(f"session {session_id} lost, recreating")
            _known_sessions.discard(session_id)
            result = await _call_session_attempt(
                last_msg, system, model, timeout, session_id, resuming=False,
            )
        else:
            raise
    _known_sessions.add(session_id)
    return result


async def call_claude(messages: List[Dict[str, Any]],
                      model: Optional[str] = None,
                      timeout: int = REQUEST_TIMEOUT,
                      max_retries: int = MAX_RETRIES,
                      session_id: Optional[str] = None):
    """Public entry point: retry + concurrency limit + optional session reuse."""
    last_err: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            if session_id:
                # Serialize within a single session; different sessions
                # still run in parallel.
                async with _get_session_lock(session_id):
                    async with _semaphore:
                        last_msg = _last_user_message(messages)
                        system = _system_from_messages(messages)
                        return await _call_session(
                            last_msg, system, model, timeout, session_id,
                        )
            else:
                async with _semaphore:
                    system, prompt = messages_to_prompt(messages)
                    return await _call_stateless(prompt, system, model, timeout)
        except (TimeoutError, RuntimeError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
            log.warning(f"[attempt {attempt}/{max_retries}] failed: {e}; "
                        f"retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"failed after {max_retries} retries: {last_err}")


# ---------- OpenAI-compatible schema ----------
class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: Optional[bool] = False
    # Extension field: enables CLI session reuse for prompt cache hits.
    # OpenAI Python SDK clients pass it via extra_body={"session_id": "..."}.
    session_id: Optional[str] = None
    # The fields below are ignored (not supported by the CLI). We accept
    # them only so standard OpenAI clients don't fail validation.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


# ---------- FastAPI ----------
app = FastAPI(title="claude-cli openai-compatible api")


def _new_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _usage_obj(usage: Dict[str, Any], prompt_text: str, completion_text: str
               ) -> Dict[str, int]:
    in_tok = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or max(1, len(prompt_text) // 4)
    )
    out_tok = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or max(1, len(completion_text) // 4)
    )
    # Forward cache-related fields if present so clients can observe
    # cache hit / miss behaviour.
    extra = {}
    for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
        if k in usage:
            extra[k] = int(usage[k])
    return {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        **extra,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if not req.messages:
        raise HTTPException(400, "messages must not be empty")

    model_for_cli = _resolve_model(req.model) or _resolve_model(DEFAULT_MODEL)
    model_label = req.model or DEFAULT_MODEL or "claude-cli"

    try:
        text, usage = await call_claude(
            messages=req.messages,
            model=model_for_cli,
            session_id=req.session_id,
        )
    except Exception as e:
        raise HTTPException(500, str(e))

    # Prompt text used for usage estimation. In session mode we only
    # actually send the last user message to the CLI.
    if req.session_id:
        prompt_for_usage = _last_user_message(req.messages)
    else:
        prompt_for_usage = messages_to_prompt(req.messages)[1]

    chat_id = _new_id()
    created = int(time.time())
    usage_payload = _usage_obj(usage, prompt_for_usage, text)

    # ---- Non-streaming ----
    if not req.stream:
        return {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_label,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": usage_payload,
        }

    # ---- Streaming (pseudo-stream over SSE) ----
    async def event_stream():
        head = {
            "id": chat_id, "object": "chat.completion.chunk", "created": created,
            "model": model_label,
            "choices": [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}],
        }
        yield f"data: {json.dumps(head, ensure_ascii=False)}\n\n"

        CHUNK = 48
        for i in range(0, len(text), CHUNK):
            piece = text[i:i + CHUNK]
            ev = {
                "id": chat_id, "object": "chat.completion.chunk", "created": created,
                "model": model_label,
                "choices": [{"index": 0, "delta": {"content": piece},
                             "finish_reason": None}],
            }
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

        tail = {
            "id": chat_id, "object": "chat.completion.chunk", "created": created,
            "model": model_label,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage_payload,
        }
        yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    now = int(time.time())
    ids = [
        "default", "best", "sonnet", "opus", "haiku",
        "sonnet[1m]", "opus[1m]", "opusplan",
        "claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5",
        "claude-sonnet-4-5", "claude-opus-4-1",
    ]
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": now,
                  "owned_by": "anthropic"} for m in ids],
    }


@app.get("/health")
async def health():
    return {
        "ok": True,
        "max_concurrency": MAX_CONCURRENCY,
        "active_sessions": len(_known_sessions),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
