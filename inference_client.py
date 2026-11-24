"""Concurrent, rate-limited, retrying client for generating code candidates.

This stands in for the "inference serving client" side of an RL rollout
pipeline: many workers need model outputs concurrently, the client must
survive transient API errors, and concurrency must be bounded so the
rollout fleet doesn't overwhelm the provider (or its own budget).
"""

import asyncio
import random

from anthropic import AsyncAnthropic, APIConnectionError, APIStatusError

MODEL = "claude-haiku-5-5"
MAX_RETRIES = 4
MAX_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a careful Python programmer. Given a function signature and a "
    "description, respond with ONLY the function implementation as valid "
    "Python code. No explanations, no markdown fences, no extra text."
)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


class InferenceClient:
    """Wraps the Claude API with bounded concurrency and exponential backoff."""

    def __init__(self, api_key: str, max_concurrency: int = 5):
        self.client = AsyncAnthropic(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_candidate(self, prompt: str) -> str:
        async with self.semaphore:
            last_error = None
            for attempt in range(MAX_RETRIES):
                try:
                    response = await self.client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = "".join(
                        block.text for block in response.content if block.type == "text"
                    )
                    return _strip_code_fences(text)
                except (APIConnectionError, APIStatusError) as e:
                    last_error = e
                    if attempt == MAX_RETRIES - 1:
                        raise
                    backoff = (2 ** attempt) + random.random()
                    await asyncio.sleep(backoff)
            raise last_error
