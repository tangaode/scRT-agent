from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


_LOCAL_ENV_LOADED = False


def _load_local_env_files() -> None:
    """Load local .env-style files without adding a runtime dependency."""
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED:
        return
    _LOCAL_ENV_LOADED = True
    candidates: list[Path] = []
    explicit = os.getenv("SCRTA_AGENT_ENV_FILE")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([Path.cwd() / ".env", Path.cwd() / ".scrta_agent.env"])
    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parsed = _parse_env_line(line)
            if not parsed:
                continue
            key, value = parsed
            os.environ.setdefault(key, value)


def _parse_env_line(line: str) -> tuple[str, str] | None:
    text = line.lstrip("\ufeff").strip()
    if not text or text.startswith("#"):
        return None
    if text.startswith("export "):
        text = text[len("export ") :].strip()
    if "=" not in text:
        return None
    key, value = text.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip().strip('"').strip("'")
    return key, value


def _first_value(keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return None


def _truthy(value: str | None) -> bool:
    return bool(value and value.lower() in {"1", "true", "yes", "on"})


@dataclass
class LLMClient:
    """Small OpenAI-compatible chat client.

    Workflow runs use LLM mode by default and should fail early when the model
    cannot be reached. Set use_llm=False only in tests or explicit internal
    deterministic tooling.
    """

    model: str = "gpt-5.4"
    use_llm: bool = True
    api_key: str | None = None
    base_url: str | None = None
    trust_env_proxy: bool | None = None
    verify_ssl: bool | None = None
    direct_httpx: bool | None = None
    timeout_seconds: float | None = None
    max_retries: int | None = None

    def __post_init__(self) -> None:
        _load_local_env_files()
        self.api_key = (
            self.api_key
            or _first_value(
                (
                    "SCRTA_AGENT_API_KEY",
                    "OPENAI_API_KEY",
                ),
            )
        )
        self.base_url = (
            self.base_url
            or _first_value(
                (
                    "SCRTA_AGENT_API_BASE",
                    "OPENAI_BASE_URL",
                    "OPENAI_API_BASE",
                ),
            )
        )
        if self.trust_env_proxy is None:
            self.trust_env_proxy = _truthy(os.getenv("SCRTA_AGENT_TRUST_ENV_PROXY"))
        if self.verify_ssl is None:
            verify_value = os.getenv("SCRTA_AGENT_LLM_VERIFY_SSL")
            self.verify_ssl = not bool(verify_value and verify_value.lower() in {"0", "false", "no", "off"})
        if self.direct_httpx is None:
            self.direct_httpx = _truthy(os.getenv("SCRTA_AGENT_LLM_DIRECT_HTTPX"))
        if self.timeout_seconds is None:
            self.timeout_seconds = float(os.getenv("SCRTA_AGENT_LLM_TIMEOUT", "180"))
        if self.max_retries is None:
            self.max_retries = int(os.getenv("SCRTA_AGENT_LLM_RETRIES", "3"))

    @property
    def available(self) -> bool:
        return bool(self.use_llm and self.api_key)

    def require_ready(self) -> None:
        if not self.use_llm:
            raise RuntimeError("LLM mode is disabled; scRTA workflow runs require a large language model.")
        if not self.api_key:
            raise RuntimeError(
                "LLM mode is required but no API key was found. Set OPENAI_API_KEY "
                "or SCRTA_AGENT_API_KEY."
            )
        try:
            import httpx  # noqa: F401
            from openai import OpenAI  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError('LLM mode is required. Install with: pip install -e ".[llm]"') from exc
        if os.getenv("SCRTA_AGENT_SKIP_LLM_HEALTHCHECK", "").lower() in {"1", "true", "yes", "on"}:
            return
        try:
            self.complete(
                "You are a health check for an analysis workflow.",
                "Reply with exactly: ok",
                temperature=0,
            )
        except Exception as exc:
            raise RuntimeError(
                "LLM mode is required, but the configured OpenAI-compatible endpoint failed a startup "
                f"health check for model `{self.model}`. Fix the API/base URL/proxy or try again when "
                f"the endpoint is healthy. Original error: {exc}"
            ) from exc

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        if not self.available:
            raise RuntimeError("LLM is not enabled or API key is missing.")
        try:
            from openai import OpenAI
            import httpx
        except Exception as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError("Install scrta-agent[llm] to use LLM mode.") from exc

        if self.direct_httpx:
            return self._complete_with_httpx(system_prompt, user_prompt, temperature)

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        kwargs["http_client"] = httpx.Client(
            trust_env=bool(self.trust_env_proxy),
            timeout=float(self.timeout_seconds),
            verify=bool(self.verify_ssl),
        )
        client = OpenAI(**kwargs)
        last_exc: Exception | None = None
        for attempt in range(int(self.max_retries or 0) + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as exc:
                if getattr(exc, "status_code", None) == 418:
                    try:
                        return self._complete_with_httpx(system_prompt, user_prompt, temperature)
                    except Exception:
                        pass
                last_exc = exc
                status_code = getattr(exc, "status_code", None)
                retryable = status_code in {408, 409, 425, 429} or (
                    isinstance(status_code, int) and status_code >= 500
                )
                retryable = retryable or exc.__class__.__name__ in {
                    "APIConnectionError",
                    "APITimeoutError",
                    "ReadTimeout",
                    "ConnectTimeout",
                    "InternalServerError",
                    "RateLimitError",
                }
                if attempt >= int(self.max_retries or 0) or not retryable:
                    raise
                time.sleep(min(2 ** attempt, 8))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("LLM request failed without an exception.")

    def _complete_with_httpx(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        import httpx

        base_url = (self.base_url or "https://api.openai.com/v1").rstrip("/")
        messages = [
            {
                "role": "user",
                "content": (
                    "System instructions:\n"
                    f"{system_prompt}\n\n"
                    "User request:\n"
                    f"{user_prompt}"
                ),
            }
        ]
        payload = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(int(self.max_retries or 0) + 1):
            try:
                with httpx.Client(
                    trust_env=bool(self.trust_env_proxy),
                    timeout=float(self.timeout_seconds),
                    verify=bool(self.verify_ssl),
                    follow_redirects=True,
                ) as client:
                    response = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"LLM response did not contain choices: {data}")
                message = choices[0].get("message") or {}
                return str(message.get("content") or "")
            except Exception as exc:
                last_exc = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                retryable = status_code in {408, 409, 418, 425, 429} or (
                    isinstance(status_code, int) and status_code >= 500
                )
                retryable = retryable or exc.__class__.__name__ in {
                    "ConnectError",
                    "ConnectTimeout",
                    "ReadTimeout",
                    "TimeoutException",
                    "RemoteProtocolError",
                }
                if attempt >= int(self.max_retries or 0) or not retryable:
                    raise
                time.sleep(min(2 ** attempt, 8))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Direct httpx LLM request failed without an exception.")
