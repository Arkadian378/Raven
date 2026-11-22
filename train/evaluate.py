"""Head-to-head evaluation: one agent (rotating seat) vs. three opponents.

Agent specs:
    greedy | random | net:PATH | mcts:PATH[:SIMS]

    python -m train.evaluate --agent mcts:runs/bc/final.pt:200 --opponent greedy --games 100
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import torch

from labyrinth.agents import GreedyAgent, RandomAgent
from labyrinth.env import LabyrinthEnv
from labyrinth.observation import ObservationConfig

_NETS = {}


def make_agent(spec: str, seed: int):
    kind, *rest = spec.split(":")
    if kind == "greedy":
        return GreedyAgent(seed)
    if kind == "random":
        return RandomAgent(seed)
    from train.mcts import MCTSAgent, NetAgent
    from train.model import load_checkpoint
    path = rest[0]
    if path not in _NETS:
        _NETS[path] = load_checkpoint(path)
    net, args = _NETS[path]
    if kind == "net":
        return NetAgent(net, seed=seed)
    if kind == "mcts":
        sims = int(rest[1]) if len(rest) > 1 else 200
        cfg = ObservationConfig(observability=args.get("observability", "partial"))
        return MCTSAgent(net, sims=sims, obs_config=cfg, seed=seed)
    raise ValueError(f"unknown agent spec {spec!r}")


def obs_config_for(spec: str) -> ObservationConfig:
    kind, *rest = spec.split(":")
    if kind in ("net", "mcts"):
        from train.model import load_checkpoint
        if rest[0] not in _NETS:
            _NETS[rest[0]] = load_checkpoint(rest[0])
        return ObservationConfig(observability=_NETS[rest[0]][1].get("observability", "partial"))
    return ObservationConfig()


def play(job):
    torch.set_num_threads(1)
    g, agent_spec, opp_spec, seed0, max_turns = job
    cfg = obs_config_for(agent_spec)
    env = LabyrinthEnv(obs_config=cfg, max_turns=max_turns)
    seat = g % 4
    agents = [make_agent(agent_spec if p == seat else opp_spec, seed0 + g * 4 + p) for p in range(4)]
    obs, _ = env.reset(seed=seed0 + g)
    think = []
    while True:
        p = env.current_player
        t = time.perf_counter()
        a = agents[p].act(env.observe(p), env.state, p)
        if p == seat:
            think.append(time.perf_counter() - t)
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            s = env.state
            return {"seat": seat, "win": int(s.winner == seat), "cards": s.found[seat] / len(s.decks[seat]),
                    "turns": s.turn, "truncated": bool(trunc), "think_s": float(np.mean(think))}


def run(agent, opponent, games=48, workers=None, seed=123_000, max_turns=400):
    jobs = [(g, agent, opponent, seed, max_turns) for g in range(games)]
    with mp.get_context("spawn").Pool(workers or mp.cpu_count()) as pool:
        res = pool.map(play, jobs)
    w = np.array([r["win"] for r in res], float)
    ci = 1.96 * np.sqrt(w.mean() * (1 - w.mean()) / len(w))
    return {
        "agent": agent, "opponent": opponent, "games": games,
        "win_rate": float(w.mean()), "ci95": float(ci),
        "win_rate_by_seat": [float(np.mean([r["win"] for r in res if r["seat"] == s])) for s in range(4)],
        "cards_frac": float(np.mean([r["cards"] for r in res])),
        "turns": float(np.mean([r["turns"] for r in res])),
        "think_s_per_decision": float(np.mean([r["think_s"] for r in res])),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--opponent", default="greedy")
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123_000)
    a = ap.parse_args()
    print(json.dumps(run(a.agent, a.opponent, a.games, a.workers, a.seed), indent=2))
