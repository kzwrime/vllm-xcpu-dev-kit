# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run deterministic OpenAI-compatible image checks against a vLLM service."""

import argparse
import base64
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUEST_TIMEOUT_SECONDS = 120.0
MAX_LOG_BODY_CHARS = 8192
PNG_DATA_URI_PREFIX = "data:image/png;base64,"
PROMPT = (
    "Read the English text in this image. Return only the exact English phrase you see. "
    "Preserve capitalization and punctuation. Treat a narrow vertical glyph immediately "
    "after a capital A as the capital letter I: transcribe that pair as AI, never Al."
)


@dataclass(frozen=True)
class ImageCase:
    path: Path
    keywords: tuple[str, ...]


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_CASES = (
    ImageCase(
        path=REPO_ROOT / "vllm/tests/multimodal/assets/image1.png",
        keywords=("hello", "ai"),
    ),
    ImageCase(
        path=REPO_ROOT / "vllm/tests/multimodal/assets/image2.png",
        keywords=("safe", "important"),
    ),
)


def format_body(body: str) -> str:
    if len(body) <= MAX_LOG_BODY_CHARS:
        return body
    return body[:MAX_LOG_BODY_CHARS] + "...<truncated>"


def source_env_with_preset(
    script_dir: str, preset_file: str | None = None
) -> dict[str, str]:
    """Load launcher variables using the same shell helpers as other tests."""
    vllm_scripts_dir = Path(script_dir).resolve().parent
    common_sh = vllm_scripts_dir / "common.sh"
    env_sh = vllm_scripts_dir / "env.sh"

    bash_command = [
        "set -e",
        f"source {shlex.quote(str(common_sh))}",
        f"load_env_file {shlex.quote(str(env_sh))}",
    ]
    if preset_file:
        bash_command.append(f"load_preset_file {shlex.quote(preset_file)}")
    else:
        bash_command.append(f"load_user_config {shlex.quote(str(vllm_scripts_dir))}")
    bash_command.extend(["apply_runtime_overrides", "env"])

    result = subprocess.run(
        ["bash", "-c", "\n".join(bash_command)],
        capture_output=True,
        text=True,
        env=os.environ,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"failed to load launcher environment: {detail}")

    env_vars: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.isidentifier():
            env_vars[key] = value
    return env_vars


def normalize_api_url(api_url: str) -> str:
    value = api_url.rstrip("/")
    if not value.endswith("/v1"):
        value += "/v1"
    return value


def build_payload(model: str, image_path: Path) -> dict[str, Any]:
    image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"{PNG_DATA_URI_PREFIX}{image_data}",
                        },
                    },
                ],
            }
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": 32,
        "temperature": 0,
    }


def validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload.get("model"), str) or not payload["model"]:
        raise ValueError("model must be a non-empty string")

    messages = payload.get("messages")
    if not isinstance(messages, list) or len(messages) != 1:
        raise ValueError("messages must contain exactly one user message")
    message = messages[0]
    if not isinstance(message, dict):
        raise TypeError("the multimodal message must be an object")
    if message.get("role") != "user":
        raise ValueError("the multimodal message must have role=user")

    content = message.get("content")
    if not isinstance(content, list) or len(content) != 2:
        raise ValueError("content must contain one text and one image_url part")
    if not all(isinstance(part, dict) for part in content):
        raise TypeError("content parts must be objects")
    content_types = [part.get("type") for part in content]
    if content_types != ["text", "image_url"]:
        raise ValueError(f"unexpected content part types: {content_types!r}")
    if not content[0].get("text"):
        raise ValueError("text content must be non-empty")

    image_url_object = content[1].get("image_url")
    if not isinstance(image_url_object, dict):
        raise TypeError("image_url content must be an object")
    image_url = image_url_object.get("url")
    if not isinstance(image_url, str) or not image_url.startswith(PNG_DATA_URI_PREFIX):
        raise ValueError("image_url must be a PNG base64 data URI")
    if not image_url.removeprefix(PNG_DATA_URI_PREFIX):
        raise ValueError("image_url data URI must contain image data")

    if payload.get("temperature") != 0:
        raise ValueError("temperature must be 0")
    if payload.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("chat_template_kwargs.enable_thinking must be false")


def request_summary(payload: dict[str, Any], image_case: ImageCase) -> dict[str, Any]:
    image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
    return {
        "model": payload["model"],
        "image": image_case.path.name,
        "image_path": str(image_case.path),
        "image_bytes": image_case.path.stat().st_size,
        "content_types": [part["type"] for part in payload["messages"][0]["content"]],
        "data_uri_prefix": image_url.split(",", 1)[0] + ",",
        "data_uri_chars": len(image_url),
        "temperature": payload["temperature"],
        "chat_template_kwargs": payload["chat_template_kwargs"],
    }


def send_json_request(
    method: str,
    url: str,
    timeout: float,
    payload: dict[str, Any] | None = None,
) -> tuple[int, str]:
    body = None
    headers = {"Authorization": "Bearer EMPTY"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def classify_backend_error(body: str) -> str:
    body_lower = body.lower()
    communication_markers = (
        "mpi",
        "all_gather",
        "communicator",
        "world size",
        "rank",
        "connection",
        "broken pipe",
        "timed out",
        "timeout",
    )
    vision_markers = (
        "vision",
        "multimodal",
        "qwen3_vl",
        "triton",
        "vit",
        "bilinear",
        "image processor",
        "kernel",
    )
    if any(marker in body_lower for marker in communication_markers):
        return "COMMUNICATION_ERROR"
    if any(marker in body_lower for marker in vision_markers):
        return "VISION_OPERATOR_ERROR"
    return "BACKEND_ERROR"


def extract_assistant_content(response: Any) -> str:
    if not isinstance(response, dict):
        raise TypeError("response JSON must be an object")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response choices must be a non-empty list")
    if not isinstance(choices[0], dict):
        raise TypeError("response choices[0] must be an object")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError("response choices[0].message must be an object")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = choices[0].get("finish_reason")
        raise ValueError(
            f"assistant content is empty or not a string; finish_reason={finish_reason!r}"
        )
    return content.strip()


def check_server_ready(api_url: str, model: str, timeout: float) -> None:
    status, raw_body = send_json_request("GET", f"{api_url}/models", timeout)
    if status < 200 or status >= 300:
        raise RuntimeError(
            f"/v1/models returned HTTP {status}: {format_body(raw_body)}"
        )
    try:
        response = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"/v1/models returned invalid JSON: {exc}") from exc
    if not isinstance(response, dict) or not isinstance(response.get("data"), list):
        raise TypeError("/v1/models response must contain a data list")
    model_ids = {
        item.get("id") for item in response.get("data", []) if isinstance(item, dict)
    }
    if model not in model_ids:
        raise RuntimeError(
            f"/v1/models does not contain {model!r}; received {sorted(model_ids)!r}"
        )


def run_multimodal_test(
    api_url: str,
    model: str,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    image_cases: tuple[ImageCase, ...] = IMAGE_CASES,
) -> int:
    api_url = normalize_api_url(api_url)
    print(f"API_BASE={api_url}")
    print(f"MODEL={model}")
    print(f"REQUEST_TIMEOUT_SECONDS={timeout}")

    try:
        check_server_ready(api_url, model, timeout)
    except (OSError, TypeError, urllib.error.URLError, RuntimeError) as exc:
        print(f"[STARTUP_ERROR] {exc}")
        return 1
    print("[STARTUP] /v1/models check passed")

    failed = 0
    for image_case in image_cases:
        print(f"\n===== {image_case.path.name} =====")
        try:
            payload = build_payload(model, image_case.path)
            validate_payload(payload)
        except (OSError, TypeError, ValueError) as exc:
            failed += 1
            print(f"[REQUEST_FORMAT_ERROR] {exc}")
            continue

        print("REQUEST_SUMMARY=" + json.dumps(request_summary(payload, image_case)))
        try:
            status, raw_body = send_json_request(
                "POST", f"{api_url}/chat/completions", timeout, payload
            )
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            failed += 1
            print(f"[COMMUNICATION_ERROR] {exc}")
            continue

        print(f"HTTP_STATUS={status}")
        print(f"RESPONSE_BODY={format_body(raw_body)}")
        if status < 200 or status >= 300:
            failed += 1
            if 400 <= status < 500:
                category = "REQUEST_FORMAT_ERROR"
            elif status >= 500:
                category = classify_backend_error(raw_body)
            else:
                category = "BACKEND_ERROR"
            print(f"[{category}] image request failed with HTTP {status}")
            continue

        try:
            response = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            failed += 1
            print(f"[RESPONSE_SCHEMA_ERROR] response is not valid JSON: {exc}")
            continue

        try:
            content = extract_assistant_content(response)
        except (TypeError, ValueError) as exc:
            failed += 1
            print(f"[OUTPUT_MISMATCH] {exc}")
            continue

        content_lower = content.lower()
        missing = [
            keyword for keyword in image_case.keywords if keyword not in content_lower
        ]
        print(f"ASSISTANT_CONTENT={content}")
        print(f"EXPECTED_KEYWORDS={list(image_case.keywords)}")
        print(f"MISSING_KEYWORDS={missing}")
        if missing:
            failed += 1
            print(f"[OUTPUT_MISMATCH] missing keywords: {missing}")
        else:
            print("[PASS] HTTP status, non-empty content, and keywords passed")

    print(f"\nMULTIMODAL_TEST_RESULT={'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate two fixed PNG inputs through an OpenAI-compatible vLLM API"
    )
    parser.add_argument(
        "-e",
        metavar="PRESET_FILE",
        help="preset file used to load USER_VLLM_MODEL and USER_VLLM_PORT",
    )
    parser.add_argument(
        "--api-url",
        help="API base URL; /v1 is added when absent, default uses USER_VLLM_PORT",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help=f"per-request timeout in seconds (default: {REQUEST_TIMEOUT_SECONDS:g})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout <= 0:
        print("[REQUEST_FORMAT_ERROR] --timeout must be greater than 0")
        return 1

    script_dir = str(Path(__file__).resolve().parent)
    try:
        env_vars = source_env_with_preset(script_dir, args.e)
    except RuntimeError as exc:
        print(f"[STARTUP_ERROR] {exc}")
        return 1

    model = env_vars.get("USER_VLLM_MODEL")
    port = env_vars.get("USER_VLLM_PORT")
    if not model:
        print("[STARTUP_ERROR] USER_VLLM_MODEL is required")
        return 1

    if args.api_url:
        api_url = args.api_url
    elif port:
        api_url = f"http://127.0.0.1:{port}/v1"
    else:
        print("[STARTUP_ERROR] USER_VLLM_PORT or --api-url is required")
        return 1
    return run_multimodal_test(api_url, model, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
