"""Provider adapters. Each client takes (image bytes, mime, prompt) and returns an ApiResult."""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field


@dataclass
class ApiResult:
    text: str = ""
    raw: object = None
    usage: dict = field(default_factory=dict)
    error: str | None = None


class RetryableError(Exception):
    """Raised for errors worth retrying (rate limit, server error, timeout)."""


class BaseClient:
    model_name = "unknown"

    def __init__(self, cfg: dict, api_cfg: dict):
        self.cfg = cfg
        self.api = api_cfg
        self.timeout = float(api_cfg.get("timeout", 120))
        self.max_tokens = int(api_cfg.get("max_tokens", 4096))

    def call(self, image: bytes, mime: str, prompt: str, filename: str, expected: str | None) -> ApiResult:
        raise NotImplementedError


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatibleClient(BaseClient):
    """Chat Completions with an image data URL. Works with Microsoft Foundry / Azure OpenAI v1 endpoints."""

    def __init__(self, cfg, api_cfg):
        super().__init__(cfg, api_cfg)
        from openai import OpenAI

        c = api_cfg.get("openai_compatible", {})
        if not c.get("base_url"):
            raise ValueError("api.openai_compatible.base_url is empty (set FOUNDRY_BASE_URL?)")
        self.c = c
        self.model_name = c.get("model", "")
        # Retries are handled by the runner, so the SDK's own retries are off.
        self.client = OpenAI(base_url=c["base_url"], api_key=c.get("api_key") or "none",
                             timeout=self.timeout, max_retries=0)

    def call(self, image, mime, prompt, filename, expected):
        import openai

        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        content = [
            {"type": "image_url", "image_url": {"url": data_url, "detail": self.c.get("image_detail", "high")}},
            {"type": "text", "text": prompt},
        ]
        kwargs = dict(model=self.model_name, messages=[{"role": "user", "content": content}],
                      max_tokens=self.max_tokens)
        if self.c.get("temperature") is not None:
            kwargs["temperature"] = self.c["temperature"]
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError,
                openai.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        except openai.APIStatusError as e:
            return ApiResult(error=f"HTTP {e.status_code}: {e.message}")
        choice = resp.choices[0]
        usage = resp.usage.model_dump() if resp.usage else {}
        error = None
        if choice.finish_reason == "length":
            error = "truncated (finish_reason=length)"
        elif choice.finish_reason == "content_filter":
            error = "content_filter"
        return ApiResult(text=choice.message.content or "", raw=resp.model_dump(), usage=usage, error=error)


# --------------------------------------------------------------------------- Anthropic (Claude)
class AnthropicClient(BaseClient):
    """Claude Messages API as a mock / reference model."""

    def __init__(self, cfg, api_cfg):
        super().__init__(cfg, api_cfg)
        import anthropic

        c = api_cfg.get("anthropic", {})
        self.c = c
        self.model_name = c.get("model", "claude-opus-5-5")
        common = dict(timeout=self.timeout, max_retries=0)
        if c.get("foundry_resource"):
            self.client = anthropic.AnthropicFoundry(api_key=c.get("api_key") or None,
                                                     resource=c["foundry_resource"], **common)
        else:
            self.client = anthropic.Anthropic(api_key=c.get("api_key") or None, **common)

    def call(self, image, mime, prompt, filename, expected):
        import anthropic

        kwargs = dict(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime,
                                                 "data": base64.standard_b64encode(image).decode()}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        if self.c.get("effort"):
            kwargs["output_config"] = {"effort": self.c["effort"]}
        try:
            resp = self.client.messages.create(**kwargs)
        except (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError,
                anthropic.InternalServerError) as e:
            raise RetryableError(str(e)) from e
        except anthropic.APIStatusError as e:
            if e.status_code == 529:  # overloaded
                raise RetryableError(str(e)) from e
            return ApiResult(error=f"HTTP {e.status_code}: {e.message}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        error = None
        if resp.stop_reason == "refusal":
            error = "refusal"
        elif resp.stop_reason == "max_tokens":
            error = "truncated (max_tokens)"
        usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
        return ApiResult(text=text, raw=resp.model_dump(mode="json"), usage=usage, error=error)


# --------------------------------------------------------------------------- Generic HTTP
def _fill(template, values: dict):
    if isinstance(template, str):
        out = template
        for k, v in values.items():
            out = out.replace("{" + k + "}", v)
        return out
    if isinstance(template, dict):
        return {k: _fill(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [_fill(v, values) for v in template]
    return template


def extract_path(obj, path: str | None):
    if not path:
        return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    for part in path.split("."):
        if isinstance(obj, list):
            obj = obj[int(part)]
        elif isinstance(obj, dict):
            obj = obj[part]
        else:
            raise KeyError(path)
    return obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)


class HttpClient(BaseClient):
    def __init__(self, cfg, api_cfg):
        super().__init__(cfg, api_cfg)
        import requests

        self.requests = requests
        self.c = api_cfg.get("http", {})
        self.model_name = self.c.get("url", "http")

    def call(self, image, mime, prompt, filename, expected):
        c = self.c
        values = {"image_b64": base64.b64encode(image).decode(), "mime": mime,
                  "prompt": prompt, "filename": filename}
        headers = _fill(c.get("headers") or {}, values)
        try:
            if c.get("mode", "json") == "multipart":
                resp = self.requests.request(
                    c.get("method", "POST"), c["url"], headers=headers, timeout=self.timeout,
                    files={c.get("file_field", "file"): (filename, image, mime)},
                    data=_fill(c.get("form") or {}, values))
            else:
                resp = self.requests.request(
                    c.get("method", "POST"), c["url"], headers=headers, timeout=self.timeout,
                    json=_fill(c.get("body") or {}, values))
        except (self.requests.Timeout, self.requests.ConnectionError) as e:
            raise RetryableError(str(e)) from e
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableError(f"HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            raw = resp.json()
        except ValueError:
            raw = resp.text
        if resp.status_code >= 400:
            return ApiResult(raw=raw, error=f"HTTP {resp.status_code}: {str(raw)[:300]}")
        try:
            text = extract_path(raw, c.get("response_path"))
        except (KeyError, IndexError, ValueError, TypeError):
            return ApiResult(raw=raw, error=f"response_path '{c.get('response_path')}' not found")
        return ApiResult(text=text, raw=raw)


# --------------------------------------------------------------------------- Echo (offline)
class EchoClient(BaseClient):
    def __init__(self, cfg, api_cfg):
        super().__init__(cfg, api_cfg)
        self.mode = api_cfg.get("echo", {}).get("mode", "answer")
        self.model_name = f"echo-{self.mode}"

    def call(self, image, mime, prompt, filename, expected):
        from .metrics import gold_alternatives

        time.sleep(0.01)
        # For multi-answer ground truth ('["a", "b"]') answer like a model would: with one alternative.
        text = gold_alternatives(expected or "")[0] if self.mode == "answer" else ""
        return ApiResult(text=text, raw={"echo": self.mode})


PROVIDERS = {
    "openai_compatible": OpenAICompatibleClient,
    "anthropic": AnthropicClient,
    "http": HttpClient,
    "echo": EchoClient,
}


def make_client(cfg: dict) -> BaseClient:
    api_cfg = cfg.get("api", {})
    name = api_cfg.get("provider", "openai_compatible")
    if name not in PROVIDERS:
        raise ValueError(f"Unknown api.provider '{name}'. Choose from {list(PROVIDERS)}")
    return PROVIDERS[name](cfg, api_cfg)
