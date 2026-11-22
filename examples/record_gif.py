"""Play one game between greedy agents and save it as a GIF."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from labyrinth.agents import GreedyAgent
from labyrinth.env import LabyrinthEnv

out = sys.argv[1] if len(sys.argv) > 1 else "game.gif"
env = LabyrinthEnv(render_mode="rgb_array")
agents = [GreedyAgent(p) for p in range(4)]
obs, _ = env.reset(seed=0)
frames = [Image.fromarray(env.render())]
while True:
    p = env.current_player
    obs, r, term, trunc, _ = env.step(agents[p].act(obs, env.state, p))
    frames.append(Image.fromarray(env.render()))
    if term or trunc:
        break
frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 10, duration=250, loop=0)
print(f"saved {len(frames)} frames to {out}")
