#!/usr/bin/env python3
"""Minimal in-memory HTTP metadata server for local Mooncake tests."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import argparse
import threading


class MetadataHandler(BaseHTTPRequestHandler):
    store = {}
    lock = threading.Lock()

    def _key(self):
        parsed = urlparse(self.path)
        if parsed.path != "/metadata":
            return None
        values = parse_qs(parsed.query).get("key", [""])
        return values[0]

    def _send(self, status, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        key = self._key()
        if key is None:
            self._send(404, "not found")
            return
        with self.lock:
            value = self.store.get(key)
        if value is None:
            self._send(404, "metadata not found")
            return
        self._send(200, value)

    def do_PUT(self):
        key = self._key()
        if key is None:
            self._send(404, "not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        with self.lock:
            self.store[key] = body
        self._send(200, "metadata updated")

    def do_DELETE(self):
        key = self._key()
        if key is None:
            self._send(404, "not found")
            return
        with self.lock:
            self.store.pop(key, None)
        self._send(200, "metadata deleted")

    def log_message(self, fmt, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MetadataHandler)
    print(f"metadata server listening on http://{args.host}:{args.port}/metadata", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
