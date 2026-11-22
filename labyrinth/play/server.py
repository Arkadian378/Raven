"""Play Labyrinth against the agent in the browser (hot-seat), recording games.

    python -m labyrinth.play.server --agent mcts:runs/alphazero/best.pt:200
    # open http://127.0.0.1:8000   (use --host 0.0.0.0 to reach it from the LAN)

Seats can be humans, the trained agent, or the greedy / random baselines.
Humans share one screen: each one reveals their own treasure card on their
turn. Every game is saved after each move to ``--games-dir`` as JSON
(configuration, seed, every action with timestamp and thinking time, result),
so it can be replayed exactly (``examples/replay_game.py``) and analysed.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..agents import GreedyAgent, RandomAgent
from ..core import PHASE_MOVE
from ..env import LabyrinthEnv
from ..observation import ObservationConfig
from ..tiles import HOME, TREASURE_NAMES

_PAGE = Path(__file__).with_name("play.html")


def make_agent(spec: str, seed: int):
    """greedy | random | net:PATH | mcts:PATH[:SIMS]  (the last two need torch)."""
    if spec == "greedy":
        return GreedyAgent(seed)
    if spec == "random":
        return RandomAgent(seed)
    from train.evaluate import make_agent as make_learned
    return make_learned(spec, seed)


class Session:
    def __init__(self, seats: list[dict], agent_spec: str, games_dir: Path, seed: int | None = None,
                 agent_delay: float = 0.8):
        self.lock = threading.RLock()
        self.seats = seats
        self.seed = int(seed if seed is not None else np.random.default_rng().integers(1_000_000))
        self.env = LabyrinthEnv(num_players=len(seats), obs_config=ObservationConfig(), max_turns=10_000)
        self.env.reset(seed=self.seed)
        self.agents = {}
        for p, s in enumerate(seats):
            if s["kind"] != "human":
                spec = agent_spec if s["kind"] == "agent" else s["kind"]
                self.agents[p] = make_agent(spec, self.seed + p)
                s["spec"] = spec
        self.agent_delay = agent_delay
        self.thinking = False
        self.closed = False
        self.turn_started = time.time()
        self.id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        self.path = games_dir / f"{self.id}.json"
        self.record = {"id": self.id, "started": datetime.now().isoformat(timespec="seconds"),
                       "seed": self.seed, "num_players": len(seats), "seats": seats,
                       "initial_state": self.env.state.to_dict(), "events": [], "result": None}
        self._save()
        threading.Thread(target=self._agent_loop, daemon=True).start()

    # ------------------------------------------------------------ actions
    def apply(self, seat: int, action: int, think_s: float):
        with self.lock:
            s = self.env.state
            if s.done:
                raise ValueError("game over")
            if seat != s.current:
                raise ValueError("not your turn")
            if not s.legal_action_mask()[action]:
                raise ValueError("illegal action")
            phase = "shift" if s.phase == 0 else "move"
            self.env.step(action)
            self.record["events"].append({
                "seat": seat, "who": self.seats[seat]["name"], "kind": self.seats[seat]["kind"],
                "phase": phase, "action": int(action), "think_s": round(think_s, 3),
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "found_after": self.env.state.found.astype(int).tolist()})
            if self.env.state.done:
                w = self.env.state.winner
                self.record["result"] = {"winner": w, "winner_name": self.seats[w]["name"],
                                         "winner_kind": self.seats[w]["kind"], "turns": self.env.state.turn,
                                         "found": self.env.state.found.astype(int).tolist(),
                                         "decks": self.env.state.decks,
                                         "ended": datetime.now().isoformat(timespec="seconds")}
            self.turn_started = time.time()
            self._save()

    def _agent_loop(self):
        while True:
            with self.lock:
                s = self.env.state
                if s.done or self.closed:
                    return
                p = s.current
                agent = self.agents.get(p)
                if agent is not None:
                    self.thinking = True
                    obs, snap = self.env.observe(p), s.clone()
            if agent is None:
                time.sleep(0.1)
                continue
            t = time.time()
            a = agent.act(obs, snap, p)
            think = time.time() - t
            time.sleep(max(0.0, self.agent_delay - think))
            self.apply(p, a, think)
            with self.lock:
                self.thinking = False

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.record, indent=1))
        tmp.replace(self.path)

    # --------------------------------------------------------------- view
    def view(self, reveal: int | None):
        with self.lock:
            s = self.env.state
            d = s.to_dict(reveal=False)
            d["seats"] = [{"name": x["name"], "kind": x["kind"]} for x in self.seats]
            d["collected_by"] = [[int(t) for t in s.decks[p][: int(s.found[p])]] for p in range(s.num_players)]
            d["going_home"] = [s.remaining(p) == 0 for p in range(s.num_players)]
            d["thinking"] = self.thinking
            d["game_id"] = self.id
            d["reachable"] = s.reachable().astype(int).tolist() if s.phase == PHASE_MOVE and not s.done else None
            d["my_target"] = None
            human_turn = self.seats[s.current]["kind"] == "human"
            if reveal is not None and reveal == s.current and human_turn and not s.done:
                t = s.target(reveal)
                d["my_target"] = {"id": int(t), "name": "home" if t == HOME else TREASURE_NAMES[t]}
            if s.done:  # reveal everything at the end
                d["decks"] = s.decks
            d["log"] = self.record["events"][-8:]
            return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="greedy", help="spec of the 'IA' seats: greedy | net:PATH | mcts:PATH:SIMS")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--games-dir", default="games")
    ap.add_argument("--agent-delay", type=float, default=0.8, help="minimum seconds per agent decision")
    args = ap.parse_args(argv)
    games_dir = Path(args.games_dir)
    games_dir.mkdir(parents=True, exist_ok=True)
    state = {"session": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                return self._send(200, _PAGE.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/state":
                sess = state["session"]
                if sess is None:
                    return self._send(200, {"game": None, "agent": args.agent})
                reveal = int(q["reveal"][0]) if "reveal" in q else None
                return self._send(200, {"game": sess.view(reveal), "agent": args.agent})
            if url.path == "/api/meta":
                return self._send(200, {"treasure_names": TREASURE_NAMES, "agent": args.agent})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/new":
                    seats = [{"name": str(s.get("name") or f"P{i}")[:24], "kind": s["kind"]}
                             for i, s in enumerate(body["seats"])]
                    if not 2 <= len(seats) <= 4 or any(s["kind"] not in ("human", "agent", "greedy", "random")
                                                      for s in seats):
                        raise ValueError("2-4 seats of kind human/agent/greedy/random")
                    if state["session"] is not None:
                        state["session"].closed = True
                    state["session"] = Session(seats, args.agent, games_dir, body.get("seed"), args.agent_delay)
                    return self._send(200, {"ok": True, "id": state["session"].id})
                if self.path == "/api/action":
                    sess = state["session"]
                    sess.apply(int(body["seat"]), int(body["action"]), time.time() - sess.turn_started)
                    return self._send(200, {"ok": True})
                self._send(404, {"error": "not found"})
            except (ValueError, KeyError, TypeError) as e:
                self._send(400, {"error": str(e)})

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[play] agent = {args.agent}\n[play] open http://{args.host}:{args.port}  (games saved in {games_dir}/)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
