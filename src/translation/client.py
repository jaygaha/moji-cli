# src/translation/client.py

import logging
import requests
import time
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)

class OllamaError(RuntimeError):
    pass

class OllamaRetryableError(OllamaError):
    pass

class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: int = 600,
        keep_alive: str = "10m",
        max_retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.max_retries = max_retries
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = self.base_url + path
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.request(method, url, timeout=self.timeout, **kwargs)

                if resp.status_code in (500, 502, 503, 504, 429):
                    raise OllamaRetryableError(
                        f"HTTP {resp.status_code}: {resp.text[:300]}"
                    )

                if resp.status_code >= 400:
                    raise OllamaError(f"HTTP {resp.status_code}: {resp.text[:500]}")

                return resp.json()

            except OllamaRetryableError as exc:
                last_error = exc
            except OllamaError:
                raise
            except requests.RequestException as exc:
                last_error = exc
            except ValueError as exc:
                # JSON decode error
                last_error = exc

            if attempt < self.max_retries:
                wait = min(30, 2**attempt)
                LOG.warning(
                    "Ollama request attempt %d/%d failed: %s. Retrying in %ss.",
                    attempt,
                    self.max_retries,
                    last_error,
                    wait,
                )
                time.sleep(wait)

        raise OllamaError(
            f"Ollama request failed after {self.max_retries} attempts: {last_error}"
        )

    def get(self, path: str) -> Dict[str, Any]:
        return self._request("GET", path)

    def chat(
        self,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        json_format: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": self.keep_alive,
        }

        if options:
            payload["options"] = options

        if json_format:
            payload["format"] = "json"

        try:
            return self._request("POST", "/api/chat", json=payload)
        except OllamaError as exc:
            # Some older Ollama versions or models may reject format=json.
            if json_format and "format" in str(exc).lower():
                LOG.warning("Retrying Ollama chat without format=json")
                payload.pop("format", None)
                return self._request("POST", "/api/chat", json=payload)
            raise


def check_ollama(client: OllamaClient, model: str) -> None:
    try:
        data = client.get("/api/tags")
        names = [m.get("name", "") for m in data.get("models", [])]
        if model not in names and not any(
            n.split(":")[0] == model.split(":")[0] for n in names
        ):
            LOG.warning(
                "Model '%s' not found in Ollama tags. Found: %s",
                model,
                ", ".join(names[:20]),
            )
    except Exception as exc:
        LOG.warning("Could not query Ollama /api/tags: %s", exc)
