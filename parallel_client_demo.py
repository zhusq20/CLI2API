"""
Parallel-call demo using the OpenAI SDK.

Start the server in another terminal first:
    python claude_api.py

Then run:
    python parallel_client_demo.py
"""
import asyncio
import time

from openai import AsyncOpenAI

client = AsyncOpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="not-used",          # any string works; we don't auth
)

PROMPTS = [
    "Explain quantum computing in one sentence.",
    "Explain Python in one sentence.",
    "Explain RAG in one sentence.",
    "Explain transformers in one sentence.",
    "Explain mixture-of-experts in one sentence.",
]


async def ask(prompt: str) -> str:
    r = await client.chat.completions.create(
        # Anything the server recognises as a placeholder ('default' /
        # 'auto' / 'claude' / 'claude-cli' / '') will fall through to
        # the CLI's account default. Pass a concrete value like 'sonnet'
        # / 'opus' / 'haiku' / 'claude-sonnet-4-6' to force a model;
        # available values depend on your local CLI version, check with
        # `claude --help | grep -A2 -- --model`.
        model="default",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content


async def main():
    t0 = time.time()
    results = await asyncio.gather(
        *(ask(p) for p in PROMPTS),
        return_exceptions=True,
    )
    for p, r in zip(PROMPTS, results):
        print(f"\nQ: {p}\nA: {r}")
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
