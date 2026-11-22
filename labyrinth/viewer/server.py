"""Live web viewer: watch games while the agent trains.

The training process owns a :class:`LiveViewer`. Environments push state
snapshots into a bounded ring buffer (cheap, non-blocking); a tiny stdlib HTTP
server running in a daemon thread serves them to a browser page that replays
them at a human-friendly speed and plots the training metrics.

    viewer = LiveViewer(port=8765).start()
    env = LabyrinthEnv(viewer=viewer)
    ...
    viewer.log_metrics(step, win_rate_vs_greedy=0.4)

Then open http://localhost:8765 (on a remote machine: ``ssh -L 8765:localhost:8765``).
No dependency beyond the standard library.
"""
from __future__ import annotations

import json
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..tiles import TREASURE_NAMES

_INDEX = Path(__file__).with_name("index.html")


class LiveViewer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, max_frames: int = 4000,
                 episode_every: int = 1):
        self.host, self.port = host, port
        self.frames: deque = deque(maxlen=max_frames)
        self.metrics: list[dict] = []
        self.meta: dict = {"treasure_names": TREASURE_NAMES}
        self.episode_every = max(1, episode_every)
        self._seq = 0
        self._lock = threading.Lock()
        self._server = None

    # ------------------------------------------------------------- producer
    def push(self, state, episode: int = 0, **extra) -> None:
        if episode % self.episode_every:
            return
        frame = {"episode": episode, "state": state.to_dict(), **extra}
        with self._lock:
            self._seq += 1
            frame["seq"] = self._seq
            self.frames.append(frame)

    def log_metrics(self, step: int, **values) -> None:
        with self._lock:
            self.metrics.append({"step": step, **{k: float(v) for k, v in values.items()}})

    def set_meta(self, **meta) -> None:
        with self._lock:
            self.meta.update(meta)

    # --------------------------------------------------------------- server
    def start(self) -> "LiveViewer":
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                url = urlparse(self.path)
                q = parse_qs(url.query)
                since = int(q.get("since", ["0"])[0])
                if url.path in ("/", "/index.html"):
                    body = _INDEX.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif url.path == "/frames":
                    with viewer._lock:
                        out = [f for f in viewer.frames if f["seq"] > since][:600]
                        latest = viewer._seq
                    self._json({"frames": out, "latest": latest})
                elif url.path == "/metrics":
                    with viewer._lock:
                        out = viewer.metrics[since:]
                    self._json({"metrics": out, "next": since + len(out)})
                elif url.path == "/meta":
                    with viewer._lock:
                        self._json(viewer.meta)
                else:
                    self.send_error(404)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[LiveViewer] open http://{self.host}:{self.port}")
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
