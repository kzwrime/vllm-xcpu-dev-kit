# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the deterministic multimodal API checker."""

import base64
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
import test_multimodal as multimodal

MODEL_ID = "Qwen/Qwen3.5-0.8B"


@dataclass
class MockApiState:
    model_ids: tuple[str, ...] = (MODEL_ID,)
    chat_status: int = 200
    chat_body: dict[str, Any] | str | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)


def response_for_image(payload: dict[str, Any]) -> dict[str, Any]:
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    image_bytes = base64.b64decode(
        image_url.removeprefix(multimodal.PNG_DATA_URI_PREFIX)
    )
    for image_case in multimodal.IMAGE_CASES:
        if image_bytes == image_case.path.read_bytes():
            return {
                "choices": [{"message": {"content": " ".join(image_case.keywords)}}]
            }
    return {"choices": [{"message": {"content": "unknown image"}}]}


def handler_for(state: MockApiState) -> type[BaseHTTPRequestHandler]:
    class MockApiHandler(BaseHTTPRequestHandler):
        def write_response(self, status: int, body: dict[str, Any] | str) -> None:
            encoded = (
                json.dumps(body).encode("utf-8")
                if isinstance(body, dict)
                else body.encode("utf-8")
            )
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            if self.path != "/v1/models":
                self.write_response(404, {"error": "not found"})
                return
            self.write_response(
                200,
                {"data": [{"id": model_id} for model_id in state.model_ids]},
            )

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self.write_response(404, {"error": "not found"})
                return
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            state.requests.append(payload)
            body = state.chat_body
            if body is None:
                body = response_for_image(payload)
            self.write_response(state.chat_status, body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return MockApiHandler


@contextmanager
def mock_api(state: MockApiState) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(state))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_payloads_embed_exact_png_bytes() -> None:
    for image_case in multimodal.IMAGE_CASES:
        payload = multimodal.build_payload(MODEL_ID, image_case.path)
        multimodal.validate_payload(payload)

        content = payload["messages"][0]["content"]
        assert [part["type"] for part in content] == ["text", "image_url"]
        image_url = content[1]["image_url"]["url"]
        encoded = image_url.removeprefix(multimodal.PNG_DATA_URI_PREFIX)
        assert base64.b64decode(encoded) == image_case.path.read_bytes()
        assert payload["temperature"] == 0
        assert payload["chat_template_kwargs"] == {"enable_thinking": False}
        assert "transcribe that pair as AI, never Al" in content[0]["text"]


@pytest.mark.parametrize(
    ("preset", "expected_tp", "expected_mpi"),
    [
        (
            "presets/serial/Qwen3.5-0.8B_dp1_tp1_eager_multimodal.sh",
            "1",
            False,
        ),
        (
            "presets/mpi/dense/Qwen3.5-0.8B_dp1_tp2_eager_multimodal.sh",
            "2",
            True,
        ),
    ],
)
def test_multimodal_presets_enable_images(
    preset: str,
    expected_tp: str,
    expected_mpi: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_bin_dir = str(Path(sys.prefix) / "bin")
    monkeypatch.setenv("PATH", os.pathsep.join((python_bin_dir, os.environ["PATH"])))
    script_dir = str(Path(multimodal.__file__).resolve().parent)
    preset_path = multimodal.REPO_ROOT / "vllm_scripts" / preset

    env = multimodal.source_env_with_preset(script_dir, str(preset_path))

    assert env["USER_VLLM_MODEL"] == MODEL_ID
    assert env["USER_VLLM_TP_SIZE"] == expected_tp
    assert "--language-model-only" not in env["VLLM_OPTIONAL_ARGS"]
    assert '--mm-processor-kwargs {"max_pixels":589824}' in env[
        "VLLM_OPTIONAL_ARGS"
    ]
    assert (env.get("VLLM_CPU_USE_MPI") == "1") is expected_mpi


def test_checker_passes_against_mock_api(capsys: pytest.CaptureFixture[str]) -> None:
    state = MockApiState()
    with mock_api(state) as api_url:
        result = multimodal.run_multimodal_test(api_url, MODEL_ID, timeout=2)

    output = capsys.readouterr().out
    assert result == 0
    assert len(state.requests) == len(multimodal.IMAGE_CASES)
    assert output.count("[PASS]") == len(multimodal.IMAGE_CASES)
    assert "MULTIMODAL_TEST_RESULT=PASS" in output


def test_checker_rejects_wrong_model(capsys: pytest.CaptureFixture[str]) -> None:
    state = MockApiState(model_ids=("other-model",))
    with mock_api(state) as api_url:
        result = multimodal.run_multimodal_test(api_url, MODEL_ID, timeout=2)

    output = capsys.readouterr().out
    assert result == 1
    assert not state.requests
    assert "[STARTUP_ERROR]" in output


@pytest.mark.parametrize(
    ("status", "body", "expected_marker"),
    [
        (400, {"error": "invalid image request"}, "[REQUEST_FORMAT_ERROR]"),
        (500, {"error": "bilinear kernel failed"}, "[VISION_OPERATOR_ERROR]"),
        (500, {"error": "MPI rank timed out"}, "[COMMUNICATION_ERROR]"),
        (200, "not-json", "[RESPONSE_SCHEMA_ERROR]"),
        (
            200,
            {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            "[OUTPUT_MISMATCH]",
        ),
        (
            200,
            {"choices": [{"message": {"content": "unrelated answer"}}]},
            "[OUTPUT_MISMATCH]",
        ),
    ],
)
def test_checker_reports_failures(
    status: int,
    body: dict[str, Any] | str,
    expected_marker: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = MockApiState(chat_status=status, chat_body=body)
    with mock_api(state) as api_url:
        result = multimodal.run_multimodal_test(api_url, MODEL_ID, timeout=2)

    output = capsys.readouterr().out
    assert result == 1
    assert expected_marker in output
    assert "MULTIMODAL_TEST_RESULT=FAIL" in output
