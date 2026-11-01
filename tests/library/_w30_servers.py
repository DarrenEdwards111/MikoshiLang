"""Local servers for the W30 (external interfaces) tests: HTTP and a raw TCP echo server.

Everything binds to 127.0.0.1 on an OS-chosen port, runs in daemon threads and is shut down by the test module's
fixture. Nothing here talks to any other host, and no real credentials are used anywhere.
"""
import json
import urllib.parse
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):          # keep test output quiet
        pass

    def _send(self, code, body, ctype="text/plain; charset=utf-8", headers=()):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _route(self):
        p = urllib.parse.urlsplit(self.path).path or "/"     # absolute URLs arrive when used as a proxy
        body = self._body()
        if p == "/text":
            return self._send(200, "hello from w30")
        if p == "/json":
            return self._send(200, json.dumps({"a": 1, "b": [1, 2, 3], "c": "x"}), "application/json")
        if p == "/csv":
            return self._send(200, "1,2\n3,4\n", "text/csv")
        if p == "/bin":
            return self._send(200, bytes(range(256)), "application/octet-stream")
        if p == "/latin":
            return self._send(200, "caf\u00e9".encode("latin-1"), "text/plain; charset=iso-8859-1")
        if p == "/big":
            return self._send(200, b"x" * 20000, "application/octet-stream")
        if p == "/echo":
            out = {"method": self.command, "path": self.path, "body": body.decode("latin-1"),
                   "headers": {k.lower(): v for k, v in self.headers.items()}}
            return self._send(200, json.dumps(out, sort_keys=True), "application/json")
        if p == "/redirect":
            return self._send(302, "", headers=[("Location", "/text")])
        if p == "/redirect303":
            return self._send(303, "", headers=[("Location", "/echo")])
        if p == "/loop":
            return self._send(302, "", headers=[("Location", "/loop")])
        if p == "/cookie":
            return self._send(200, "cookies set", headers=[("Set-Cookie", "w30a=1; Path=/"),
                                                           ("Set-Cookie", "w30b=two; Max-Age=3600; HttpOnly")])
        if p == "/named":
            return self._send(200, "named file", headers=[("Content-Disposition", 'attachment; filename="w30.txt"')])
        if p.startswith("/status/"):
            code = int(p.rsplit("/", 1)[1])
            return self._send(code, f"status {code}")
        return self._send(404, "not found")

    do_GET = do_POST = do_PUT = do_DELETE = do_HEAD = _route


class HTTPServer:
    def __init__(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class _Echo(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            data = self.request.recv(65536)
            if not data:
                return
            self.request.sendall(data.upper())


class EchoServer:
    """TCP server that answers every chunk with its upper-cased bytes."""

    def __init__(self):
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Echo)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p
