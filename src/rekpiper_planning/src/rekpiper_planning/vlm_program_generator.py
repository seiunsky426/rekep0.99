"""Vision API backends for the unchanged ReKep constraint prompt."""

import base64
import hashlib
import os
from pathlib import Path
import time

import requests


class VLMBackendError(RuntimeError):
    pass


_PROVIDERS = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY",
               ("chatgpt-4o-latest", "gpt-4o")),
    "qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY",
             ("qwen3-vl-plus", "qwen3-vl-flash")),
}


class VisionChatBackend:
    def __init__(self, config, session=None):
        self.provider = str(config.get("provider", "")).lower()
        if self.provider not in _PROVIDERS:
            raise VLMBackendError("supported VLM providers are openai and qwen")
        endpoint, key_env, models = _PROVIDERS[self.provider]
        self.base_url = str(config.get("base_url", endpoint)).rstrip("/")
        self.api_key_env = str(config.get("api_key_env", key_env))
        self.model = str(config.get("model", models[0]))
        if self.base_url != endpoint or self.api_key_env != key_env:
            raise VLMBackendError("VLM endpoint and credential variable do not match provider")
        if self.model not in models:
            raise VLMBackendError("unsupported vision model for " + self.provider)
        timeout_s = config.get("timeout_s", 90.0)
        self.timeout_s = float(timeout_s)
        self.session = session or requests.Session()
        self.last_image_sha256 = ""
        if not 1.0 <= self.timeout_s <= 300.0:
            raise VLMBackendError("VLM timeout must lie in [1, 300] seconds")

    def _api_key(self):
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise VLMBackendError("missing " + self.api_key_env)
        return key

    def _image_data_url(self, image_path):
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise VLMBackendError("annotated ReKep image is unavailable")
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise VLMBackendError("annotated image is unreadable: {}".format(exc)) from exc
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            raise VLMBackendError("annotated ReKep image must be PNG")
        self.last_image_sha256 = hashlib.sha256(raw).hexdigest()
        return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")

    def generate_text(self, prompt, image_path):
        if not str(prompt).strip() or not image_path:
            raise VLMBackendError("official prompt and annotated image are required")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": str(prompt)},
                {"type": "image_url", "image_url": {
                    "url": self._image_data_url(image_path)}},
            ]}],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        if self.provider == "qwen":
            payload["enable_thinking"] = False
        transient = (requests.exceptions.ConnectionError,
                     requests.exceptions.SSLError)
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self._api_key(),
                             "Content-Type": "application/json"},
                    json=payload, timeout=self.timeout_s)
                response.raise_for_status()
                output = response.json()["choices"][0]["message"]["content"]
                if not isinstance(output, str) or not output.strip():
                    raise VLMBackendError(self.model + " returned empty content")
                return output
            except transient as exc:
                if attempt == 2:
                    raise VLMBackendError(
                        "{} connection failed after retries: {}".format(self.model, exc)) from exc
                time.sleep(0.5 * (2 ** attempt))
            except (requests.RequestException, ValueError, KeyError,
                    IndexError, TypeError) as exc:
                raise VLMBackendError("{} request failed: {}".format(self.model, exc)) from exc
        raise VLMBackendError(self.model + " request failed")


def create_backend(config, session=None):
    return VisionChatBackend(config, session=session)
