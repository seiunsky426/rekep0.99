#!/usr/bin/env python3

import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from PIL import Image
import requests

from rekpiper_planning.vlm_program_generator import (
    VLMBackendError, create_backend)


class _Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "num_stages = 1"}}]}


class _Session:
    def __init__(self):
        self.payload = None

    def post(self, url, **kwargs):
        self.payload = (url, kwargs)
        return _Response()


class VLMBackendTest(unittest.TestCase):
    def test_unsupported_providers_and_models_are_rejected(self):
        with self.assertRaises(VLMBackendError):
            create_backend({"provider": "other"})
        with self.assertRaises(VLMBackendError):
            create_backend({"provider": "openai", "model": "gpt-5"})

    def test_visual_request_is_deterministic_and_contains_image(self):
        session = _Session()
        backend = create_backend(
            {"provider": "openai", "model": "gpt-4o"}, session=session)
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (8, 8), "white").save(image_file.name)
            raw = Path(image_file.name).read_bytes()
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                output = backend.generate_text("official prompt", image_file.name)
        self.assertEqual(output, "num_stages = 1")
        url, request = session.payload
        self.assertEqual(url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(request["json"]["temperature"], 0.0)
        image_url = request["json"]["messages"][0]["content"][1]
        self.assertTrue(image_url["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(backend.last_image_sha256, hashlib.sha256(raw).hexdigest())

    def test_default_model_is_official_chatgpt_4o_latest(self):
        backend = create_backend({"provider": "openai"}, session=_Session())
        self.assertEqual(backend.model, "chatgpt-4o-latest")

    def test_missing_key_fails_closed_without_network(self):
        backend = create_backend({"provider": "openai", "model": "gpt-4o"})
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(VLMBackendError):
                backend._api_key()

    def test_qwen_uses_vision_image_and_dashscope_key(self):
        session = _Session()
        backend = create_backend({"provider": "qwen"}, session=session)
        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (8, 8), "white").save(image_file.name)
            with mock.patch.dict(os.environ, {"DASHSCOPE_API_KEY": "qwen-test",
                                               "OPENAI_API_KEY": "openai-test"}):
                backend.generate_text("official prompt", image_file.name)
        url, request = session.payload
        self.assertEqual(url, "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(request["headers"]["Authorization"], "Bearer qwen-test")
        self.assertEqual(request["json"]["model"], "qwen3-vl-plus")
        self.assertFalse(request["json"]["enable_thinking"])
        self.assertEqual(request["json"]["messages"][0]["content"][1]["type"], "image_url")

    def test_qwen_does_not_send_credentials_to_another_endpoint(self):
        for change in ({"base_url": "https://example.com/v1"},
                       {"api_key_env": "OPENAI_API_KEY"}, {"model": "qwen-plus"}):
            with self.assertRaises(VLMBackendError):
                create_backend(dict(provider="qwen", **change))
        backend = create_backend({"provider": "qwen"})
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "other"}, clear=True):
            with self.assertRaisesRegex(VLMBackendError, "missing DASHSCOPE_API_KEY"):
                backend._api_key()


if __name__ == "__main__":
    unittest.main()
