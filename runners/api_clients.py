# runners/api_clients.py
"""Unified API client interface for OpenAI, Anthropic, and Google Gemini."""
from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@dataclass
class GenerationResult:
    """Result of a generation request."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model: str
    ok: bool = True
    error: Optional[str] = None
    reasoning_text: Optional[str] = None   # separate chain-of-thought (reasoning models)
    finish_reason: Optional[str] = None    # "stop" | "length" | ...


class LLMClient(ABC):
    """Abstract base class for LLM API clients."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        """Generate a response to the given prompt."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4o, o1, o3, etc.)."""

    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI, RateLimitError, APIError
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self._rate_limit_error = RateLimitError
        self._api_error = APIError

    @property
    def model_name(self) -> str:
        return self.model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((Exception,)),  # Will be more specific
    )
    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        start = time.perf_counter()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            # o1/o3 models don't support system messages or temperature
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 4096,
            }
            if not self.model.startswith("o1") and not self.model.startswith("o3"):
                kwargs["temperature"] = 0.0

            response = self.client.chat.completions.create(**kwargs)
            latency = (time.perf_counter() - start) * 1000

            return GenerationResult(
                text=response.choices[0].message.content or "",
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                latency_ms=latency,
                model=self.model,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return GenerationResult(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model=self.model,
                ok=False,
                error=str(e),
            )


class AnthropicClient(LLMClient):
    """Anthropic API client (Claude models)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        start = time.perf_counter()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system or "",
                messages=[{"role": "user", "content": prompt}],
            )
            latency = (time.perf_counter() - start) * 1000

            text = ""
            if response.content:
                text = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])

            return GenerationResult(
                text=text,
                input_tokens=response.usage.input_tokens if response.usage else 0,
                output_tokens=response.usage.output_tokens if response.usage else 0,
                latency_ms=latency,
                model=self.model,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return GenerationResult(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model=self.model,
                ok=False,
                error=str(e),
            )


class GeminiClient(LLMClient):
    """Google Gemini API client."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        self._model_name = model
        self.model = genai.GenerativeModel(model)

    @property
    def model_name(self) -> str:
        return self._model_name

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        start = time.perf_counter()
        try:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = self.model.generate_content(
                full_prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 4096},
            )
            latency = (time.perf_counter() - start) * 1000

            text = response.text if hasattr(response, 'text') else ""

            # Gemini doesn't always provide token counts
            input_tokens = 0
            output_tokens = 0
            if hasattr(response, 'usage_metadata'):
                input_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                output_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)

            return GenerationResult(
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency,
                model=self._model_name,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return GenerationResult(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model=self._model_name,
                ok=False,
                error=str(e),
            )


class OllamaClient(LLMClient):
    """Ollama local model client."""

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        import ollama
        self._ollama = ollama
        self.model = model

    @property
    def model_name(self) -> str:
        return self.model

    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        start = time.perf_counter()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            kwargs = {"model": self.model, "messages": messages}
            if fmt is not None:  # structured output: "json" or a JSON-schema dict
                kwargs["format"] = fmt
            response = self._ollama.chat(**kwargs)
            latency = (time.perf_counter() - start) * 1000

            return GenerationResult(
                text=response["message"]["content"],
                input_tokens=response.get("prompt_eval_count", 0),
                output_tokens=response.get("eval_count", 0),
                latency_ms=latency,
                model=self.model,
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return GenerationResult(
                text="",
                input_tokens=0,
                output_tokens=0,
                latency_ms=latency,
                model=self.model,
                ok=False,
                error=str(e),
            )


class SambaNovaClient(LLMClient):
    """SambaNova Cloud client (OpenAI-compatible REST API).

    Model IDs as listed by GET /v1/models, e.g. "MiniMax-M2.7", "DeepSeek-V3.2",
    "gpt-oss-120b". Accepts an optional "sambanova/" prefix which is stripped for
    the API call. Reasoning models return their chain-of-thought in a separate
    `message.reasoning` field; we keep it out of `text` (what the judge sees) and
    expose it as `reasoning_text`.

    Env: SAMBANOVA_API_KEY (required), SAMBANOVA_BASE_URL (optional),
         SAMBANOVA_TEMPERATURE (default 0.0), SAMBANOVA_TOP_P (default: omitted),
         SAMBANOVA_MAX_TOKENS (default 16384), SAMBANOVA_TIMEOUT (default 600 s),
         SAMBANOVA_STREAM=1 (SSE streaming; avoids gateway timeouts on long outputs).
    """

    RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

    def __init__(self, model: str = "MiniMax-M2.7", temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None, max_attempts: int = 8):
        import httpx
        self._httpx = httpx
        self.model = model[len("sambanova/"):] if model.lower().startswith("sambanova/") else model
        self.base_url = os.environ.get("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1").rstrip("/")
        self.api_key = os.environ["SAMBANOVA_API_KEY"]
        self.temperature = float(os.environ.get("SAMBANOVA_TEMPERATURE", 0.0)) if temperature is None else temperature
        self.max_tokens = int(os.environ.get("SAMBANOVA_MAX_TOKENS", 16384)) if max_tokens is None else max_tokens
        self.stream = os.environ.get("SAMBANOVA_STREAM", "0") == "1"  # SSE for very long generations
        _tp = os.environ.get("SAMBANOVA_TOP_P")
        self.top_p: Optional[float] = float(_tp) if _tp else None
        self.max_attempts = max_attempts
        read_timeout = float(os.environ.get("SAMBANOVA_TIMEOUT", 600))  # seconds per request
        self._client = httpx.Client(timeout=httpx.Timeout(read_timeout, connect=30.0))

    @property
    def model_name(self) -> str:
        return self.model

    def _post(self, payload: dict) -> dict:
        """POST with retry on rate limits / transient server errors."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            try:
                r = self._client.post(url, headers=headers, json=payload)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in self.RETRY_STATUS:
                    retry_after = r.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else min(60.0, 2.0 * (2 ** attempt))
                    last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            except (self._httpx.TimeoutException, self._httpx.TransportError) as e:
                last_err = e
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
        raise RuntimeError(f"giving up after {self.max_attempts} attempts: {last_err}")

    def _post_stream(self, payload: dict) -> dict:
        """Streamed request (SSE). Keeps the connection alive for very long generations
        that otherwise hit the provider's gateway timeout. Returns a dict shaped like
        the non-streamed response: choices[0].message.{content,reasoning},
        choices[0].finish_reason, usage."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = dict(payload, stream=True, stream_options={"include_usage": True})
        last_err: Optional[Exception] = None
        for attempt in range(self.max_attempts):
            content, reasoning, finish, usage = [], [], None, {}
            try:
                with self._client.stream("POST", url, headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        body = r.read().decode("utf-8", "replace")[:300]
                        if r.status_code in self.RETRY_STATUS:
                            last_err = RuntimeError(f"HTTP {r.status_code}: {body}")
                            time.sleep(min(60.0, 2.0 * (2 ** attempt)))
                            continue
                        raise RuntimeError(f"HTTP {r.status_code}: {body}")
                    for line in r.iter_lines():
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        if chunk.get("usage"):
                            usage = chunk["usage"]
                        for ch in chunk.get("choices") or []:
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                content.append(d["content"])
                            rs = d.get("reasoning") or d.get("reasoning_content")
                            if rs:
                                reasoning.append(rs)
                            if ch.get("finish_reason"):
                                finish = ch["finish_reason"]
                return {"choices": [{"message": {"content": "".join(content),
                                                 "reasoning": "".join(reasoning) or None},
                                     "finish_reason": finish}],
                        "usage": usage}
            except (self._httpx.TimeoutException, self._httpx.TransportError) as e:
                last_err = e
                time.sleep(min(60.0, 2.0 * (2 ** attempt)))
        raise RuntimeError(f"giving up after {self.max_attempts} attempts: {last_err}")

    def generate(self, prompt: str, system: Optional[str] = None, fmt=None) -> GenerationResult:
        start = time.perf_counter()
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
            if self.top_p is not None:
                payload["top_p"] = self.top_p
            if fmt is not None:
                payload["response_format"] = {"type": "json_object"}
            data = self._post_stream(payload) if self.stream else self._post(payload)
            latency = (time.perf_counter() - start) * 1000
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            usage = data.get("usage") or {}
            return GenerationResult(
                text=msg.get("content") or "",
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                latency_ms=latency,
                model=self.model,
                reasoning_text=msg.get("reasoning") or msg.get("reasoning_content") or None,
                finish_reason=choice.get("finish_reason"),
            )
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return GenerationResult(
                text="", input_tokens=0, output_tokens=0, latency_ms=latency,
                model=self.model, ok=False, error=str(e),
            )


def get_client(model_name: str) -> LLMClient:
    """Factory function to get the appropriate client for a model."""
    model_lower = model_name.lower()

    # SambaNova Cloud: explicit "sambanova/" prefix, or a hosted model family
    # that is not served locally (MiniMax-M2.7, DeepSeek-V3.x, gemma-4-31B-it ...).
    # Dashed "gpt-oss-120b" is the SambaNova ID; the colon form "gpt-oss:20b" is Ollama.
    if model_lower.startswith("sambanova/") or model_lower.startswith("minimax-") \
            or model_lower.startswith("deepseek-v3") or model_lower.startswith("gemma-4-") \
            or model_lower.startswith("gpt-oss-"):
        return SambaNovaClient(model_name)

    # Local Ollama models use a "name:tag" form (e.g. "gpt-oss:120b",
    # "qwen2.5-coder:7b"). Route those to Ollama first so a local name like
    # "gpt-oss" is not misread as an OpenAI API model.
    if ":" in model_name or model_lower.startswith("gpt-oss"):
        return OllamaClient(model_name)

    # OpenAI models
    if any(model_lower.startswith(p) for p in ["gpt-", "o1", "o3"]):
        return OpenAIClient(model_name)

    # Anthropic models
    if model_lower.startswith("claude"):
        return AnthropicClient(model_name)

    # Google models
    if model_lower.startswith("gemini"):
        return GeminiClient(model_name)

    # Assume Ollama for everything else (local models)
    return OllamaClient(model_name)


# Model pricing per 1M tokens (as of early 2025)
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "o1": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    # SambaNova Cloud (developer tier: $0 for this project's runs)
    "MiniMax-M2.7": {"input": 0.0, "output": 0.0},
    # Local (free)
    "qwen2.5-coder:7b": {"input": 0.0, "output": 0.0},
    "codellama:7b": {"input": 0.0, "output": 0.0},
    "deepseek-coder:6.7b": {"input": 0.0, "output": 0.0},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD for a generation."""
    pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000
