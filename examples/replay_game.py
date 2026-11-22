"""Replay a recorded human-vs-agent game in the live viewer (god view).

    python examples/replay_game.py games/20261121-121451-22c60b.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from labyrinth.env import LabyrinthEnv
from labyrinth.viewer.server import LiveViewer

ap = argparse.ArgumentParser()
ap.add_argument("record")
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--delay", type=float, default=0.5)
a = ap.parse_args()
rec = json.load(open(a.record))
viewer = LiveViewer(port=a.port).start()
env = LabyrinthEnv(num_players=rec["num_players"], max_turns=10_000, viewer=viewer)
env.reset(seed=rec["seed"])
assert env.state.to_dict()["board"] == rec["initial_state"]["board"], "engine/seed mismatch"
for e in rec["events"]:
    time.sleep(a.delay)
    env.step(e["action"])
print("result:", rec["result"])
input("press Enter to quit")
