"""Stream scripted games to the live viewer (no training needed).

    python examples/watch_live.py            # then open http://127.0.0.1:8765
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.agents import GreedyAgent, RandomAgent
from labyrinth.env import LabyrinthEnv
from labyrinth.viewer.server import LiveViewer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--games", type=int, default=1000)
ap.add_argument("--delay", type=float, default=0.15, help="seconds between moves")
args = ap.parse_args()

viewer = LiveViewer(port=args.port).start()
env = LabyrinthEnv(viewer=viewer, max_turns=400)
agents = [GreedyAgent(0), GreedyAgent(1, epsilon=0.3), RandomAgent(2), GreedyAgent(3, epsilon=0.1)]
for g in range(args.games):
    obs, _ = env.reset(seed=g)
    while True:
        p = env.current_player
        obs, r, term, trunc, info = env.step(agents[p].act(obs, env.state, p))
        time.sleep(args.delay)
        if term or trunc:
            viewer.log_metrics(g, game_length=env.state.turn, winner=env.state.winner)
            break
