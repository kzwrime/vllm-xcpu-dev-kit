#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import time
from typing import Any

from aiohttp import ClientSession, ClientTimeout, web


LOGGER = logging.getLogger("pd_proxy")


def _choice_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    message = choice.get("message") or {}
    return message.get("content") or message.get("reasoning") or ""


def _prepend_choice_text(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0]
        if "text" in choice:
            choice["text"] = prefix + (choice.get("text") or "")
        else:
            message = choice.setdefault("message", {})
            message["content"] = prefix + (message.get("content") or "")

    usage = payload.get("usage")
    if isinstance(usage, dict):
        usage["completion_tokens"] = int(usage.get("completion_tokens") or 0) + 1
        usage["total_tokens"] = int(usage.get("total_tokens") or 0) + 1
    return payload


def _completion_decode_payload(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    decode_payload = copy.deepcopy(payload)
    prompt = decode_payload.get("prompt")
    if isinstance(prompt, str):
        decode_payload["prompt"] = prompt + prefix
    elif isinstance(prompt, list):
        decode_payload["prompt"] = [
            item + prefix if isinstance(item, str) else item for item in prompt
        ]
    return decode_payload


def _chat_decode_payload(payload: dict[str, Any], prefix: str) -> dict[str, Any]:
    decode_payload = copy.deepcopy(payload)
    messages = list(decode_payload.get("messages") or [])
    messages.append({"role": "assistant", "content": prefix})
    decode_payload["messages"] = messages
    return decode_payload


def _set_max_tokens(payload: dict[str, Any], value: int) -> None:
    if "max_completion_tokens" in payload:
        payload["max_completion_tokens"] = value
    payload["max_tokens"] = value


def _set_stream(payload: dict[str, Any], value: bool) -> None:
    payload["stream"] = value
    if not value:
        payload.pop("stream_options", None)


def _get_max_tokens(payload: dict[str, Any]) -> int:
    value = payload.get("max_completion_tokens", payload.get("max_tokens", 16))
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 16


def _parse_urls(value: str) -> list[str]:
    urls = [url.strip().rstrip("/") for url in value.split(",") if url.strip()]
    if not urls:
        raise argparse.ArgumentTypeError("URL list must not be empty")
    return urls


def _next_url(request: web.Request, urls_key: str, index_key: str) -> str:
    urls = request.app[urls_key]
    index = request.app[index_key]
    request.app[index_key] = (index + 1) % len(urls)
    return urls[index]


def _sse_chunk(
    path: str,
    model: str,
    content: str,
) -> bytes:
    chunk = {
        "id": f"chatcmpl-pd-proxy-{time.time_ns()}",
        "object": "chat.completion.chunk"
        if path.endswith("/chat/completions")
        else "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": None,
            }
        ],
    }
    if path.endswith("/chat/completions"):
        chunk["choices"][0]["delta"] = {"content": content}
    else:
        chunk["choices"][0]["text"] = content
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


async def _post_json(
    session: ClientSession,
    base_url: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    async with session.post(f"{base_url}{path}", json=payload) as response:
        text = await response.text()
        if response.status != 200:
            LOGGER.error("upstream %s%s failed: status=%s body=%r",
                         base_url, path, response.status, text[:1000])
            raise web.HTTPBadGateway(text=text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            LOGGER.error("upstream %s%s returned non-json body: %r",
                         base_url, path, text[:1000])
            raise web.HTTPBadGateway(text=text) from exc


async def _stream_decode(
    session: ClientSession,
    request: web.Request,
    decode_url: str,
    decode_payload: dict[str, Any],
    first_text: str,
) -> web.StreamResponse:
    stream_response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await stream_response.prepare(request)

    model = str(decode_payload.get("model") or "")
    if first_text:
        await stream_response.write(_sse_chunk(request.path, model, first_text))

    async with session.post(
        f"{decode_url}{request.path}", json=decode_payload
    ) as response:
        if response.status != 200:
            text = await response.text()
            LOGGER.error("decode stream failed: status=%s body=%r",
                         response.status, text[:1000])
            raise web.HTTPBadGateway(text=text)
        async for chunk in response.content.iter_chunked(8192):
            await stream_response.write(chunk)

    await stream_response.write_eof()
    return stream_response


async def _forward_models(request: web.Request) -> web.Response:
    decode_url = request.app["decode_urls"][0]
    async with ClientSession(timeout=request.app["timeout"]) as session:
        async with session.get(f"{decode_url}/v1/models") as response:
            body = await response.read()
            return web.Response(
                body=body,
                status=response.status,
                content_type=response.content_type,
            )


async def _handle_generation(request: web.Request) -> web.Response:
    payload = await request.json()
    max_tokens = _get_max_tokens(payload)
    stream = bool(payload.get("stream"))
    prefill_url = _next_url(request, "prefill_urls", "prefill_index")
    decode_url = _next_url(request, "decode_urls", "decode_index")

    prefill_payload = copy.deepcopy(payload)
    _set_stream(prefill_payload, False)
    _set_max_tokens(prefill_payload, 1)

    async with ClientSession(timeout=request.app["timeout"]) as session:
        LOGGER.info("prefill request url=%s path=%s", prefill_url, request.path)
        prefill_response = await _post_json(
            session, prefill_url, request.path, prefill_payload
        )
        first_text = _choice_text(prefill_response)
        LOGGER.info("prefill produced %r", first_text)

        if request.path.endswith("/chat/completions"):
            decode_payload = _chat_decode_payload(payload, first_text)
        else:
            decode_payload = _completion_decode_payload(payload, first_text)

        _set_max_tokens(decode_payload, max(1, max_tokens - 1))
        _set_stream(decode_payload, stream)

        LOGGER.info("decode request url=%s path=%s", decode_url, request.path)
        if stream:
            return await _stream_decode(
                session, request, decode_url, decode_payload, first_text
            )

        decode_response = await _post_json(
            session, decode_url, request.path, decode_payload
        )

    return web.json_response(_prepend_choice_text(decode_response, first_text))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local vLLM P/D proxy")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--prefill-url", required=True)
    parser.add_argument("--decode-url", required=True)
    parser.add_argument("--timeout", type=float, default=21600)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = web.Application()
    app["prefill_urls"] = _parse_urls(args.prefill_url)
    app["decode_urls"] = _parse_urls(args.decode_url)
    app["prefill_index"] = 0
    app["decode_index"] = 0
    app["timeout"] = ClientTimeout(total=args.timeout)
    app.router.add_get("/v1/models", _forward_models)
    app.router.add_post("/v1/completions", _handle_generation)
    app.router.add_post("/v1/chat/completions", _handle_generation)

    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
