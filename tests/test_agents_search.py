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
    buf.add(obs, pi, np.array([1, -1 / 3, -1 / 3, -1 / 3], np.float32), np.array([5, 7, -1], np.int8))
    buf.save(tmp_path / "d.npz")
    b2 = SampleBuffer.load(tmp_path / "d.npz")
    planes, vec, mask, p, z, opp = b2.batch(np.array([0]))
    assert opp[0].tolist() == [5, 7, -1]
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


# --------------------------------------------------------- opponent modelling
def test_trail_follows_shifts_and_resets_on_collect():
    from labyrinth.core import GameState
    from labyrinth.observation import ObservationConfig, encode, plane_names
    rng = np.random.default_rng(1)
    s = GameState.new_game(4, rng=rng)
    for _ in range(40):
        a = int(rng.choice(np.flatnonzero(s.legal_action_mask())))
        p = s.current
        before = s.pos[p].copy()
        found = int(s.found[p])
        s.step(a)
        if a >= 48:
            if s.found[p] > found:
                assert (s.trail[:, p] == -1).all()
            else:
                assert tuple(s.trail[0, p]) == tuple(before)
        assert ((s.trail >= -1) & (s.trail < 7)).all()
        if s.done:
            break
    cfg = ObservationConfig(history=2)
    o = encode(s, s.current, cfg)
    assert o["planes"].shape[0] == len(plane_names(cfg)) == len(plane_names(ObservationConfig())) + 6


def test_determinize_with_belief_respects_information():
    from labyrinth.core import GameState
    s = GameState.new_game(4, rng=np.random.default_rng(0))
    rng = np.random.default_rng(0)
    probs = np.zeros(24); probs[20] = 1.0    # we are sure seat 1 is looking for treasure 20
    if s.target(0) == 20:
        probs[:] = 0; probs[21] = 1.0
    want = int(np.argmax(probs))
    for _ in range(20):
        d = s.determinize(0, rng, {1: probs})
        assert d.target(1) == want
        assert d.decks[0][0] == s.decks[0][0]
        assert [len(x) for x in d.decks] == [len(x) for x in s.decks]
        dealt = [t for deck in d.decks for t in deck]
        assert len(dealt) == len(set(dealt))


def test_checkpoint_upgrade_keeps_behaviour(tmp_path):
    import torch
    from train.model import load_checkpoint, obs_config_from_args
    from labyrinth.observation import encode
    net = build_net({"blocks": 1}).eval()
    torch.save({"model": {k: v for k, v in net.state_dict().items() if not k.startswith("belief_head")},
                "args": {"blocks": 1}}, tmp_path / "old.pt")
    new, args = load_checkpoint(tmp_path / "old.pt", overrides={"history": 2})
    assert not new.belief_trained and args["history"] == 2
    env = LabyrinthEnv(seed=0)
    env.reset()
    o_old = encode(env.state, 0, obs_config_from_args({}))
    o_new = encode(env.state, 0, obs_config_from_args(args))
    t = lambda o, k, dt: torch.as_tensor(o[k][None], dtype=dt)
    with torch.no_grad():
        a = net(t(o_old, "planes", torch.float32), t(o_old, "vector", torch.float32), t(o_old, "action_mask", torch.bool))
        b = new(t(o_new, "planes", torch.float32), t(o_new, "vector", torch.float32), t(o_new, "action_mask", torch.bool))
    assert torch.allclose(a[0], b[0], atol=1e-5) and torch.allclose(a[1], b[1], atol=1e-5)
