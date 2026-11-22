import numpy as np
import pytest

torch = pytest.importorskip("torch")

from labyrinth.agents import GreedyAgent, play_game  # noqa: E402
from labyrinth.core import GameState  # noqa: E402
from labyrinth.env import LabyrinthEnv  # noqa: E402
from train.data import SampleBuffer, ego, outcome_vector  # noqa: E402
from train.mcts import MCTSAgent  # noqa: E402
from train.model import build_net  # noqa: E402


def test_greedy_best_actions_are_legal():
    env = LabyrinthEnv(seed=0)
    obs, _ = env.reset()
    g = GreedyAgent(0)
    for _ in range(30):
        best = g.best_actions(env.state)
        assert best and all(obs["action_mask"][a] for a in best)
        obs, *_ = env.step(best[0])


def test_outcome_vector_zero_sum_and_ego():
    s = GameState.new_game(4, rng=np.random.default_rng(0))
    s.winner = 2
    z = outcome_vector(s)
    assert z[2] == 1 and abs(z.sum()) < 1e-6
    assert ego(z, 2, 4)[0] == 1 and ego(z, 3, 4)[3] == 1


def test_sample_buffer_roundtrip(tmp_path):
    env = LabyrinthEnv(seed=0)
    obs, _ = env.reset()
    buf = SampleBuffer(10, env.planes_shape, env.vector_shape[0])
    pi = np.zeros(97, np.float32); pi[3] = 1
    buf.add(obs, pi, np.array([1, -1 / 3, -1 / 3, -1 / 3], np.float32))
    buf.save(tmp_path / "d.npz")
    b2 = SampleBuffer.load(tmp_path / "d.npz")
    planes, vec, mask, p, z = b2.batch(np.array([0]))
    assert np.array_equal(planes[0].numpy(), obs["planes"])
    assert np.array_equal(mask[0].numpy(), obs["action_mask"])


def test_mcts_plays_legal_game():
    net = build_net({}).eval()
    env = LabyrinthEnv(max_turns=20)
    agents = [MCTSAgent(net, sims=8, max_turns=20, seed=0)] + [GreedyAgent(p) for p in range(1, 4)]
    info = play_game(env, agents, seed=0)
    assert info["turn"] > 0
    pi = agents[0].last_pi
    assert abs(pi.sum() - 1) < 1e-5
