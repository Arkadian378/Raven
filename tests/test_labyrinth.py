import numpy as np
import pytest

from labyrinth.agents import GreedyAgent, RandomAgent, play_game
from labyrinth.core import NUM_ACTIONS, PHASE_MOVE, GameState, encode_shift
from labyrinth.env import LabyrinthEnv, RewardConfig
from labyrinth.observation import ObservationConfig, plane_names, vector_names
from labyrinth.tiles import BOARD_SIZE, FIXED_TILES, NUM_TREASURES, opposite_slot, rotate, shape_of


def tile_multiset(s: GameState):
    """Rotation-invariant multiset of (shape, treasure) over board + spare."""
    out = [(shape_of(int(m)), int(t)) for m, t in zip(s.board.ravel(), s.treasure.ravel())]
    out.append((shape_of(s.spare), s.spare_treasure))
    return sorted(out)


def test_initial_setup():
    s = GameState.new_game(4, rng=np.random.default_rng(0))
    shapes = [x[0] for x in tile_multiset(s)]
    assert len(shapes) == 50
    assert shapes.count("I") == 12 and shapes.count("L") == 20 and shapes.count("T") == 18
    treasures = [t for _, t in tile_multiset(s) if t >= 0]
    assert sorted(treasures) == list(range(NUM_TREASURES))
    for rc, m in FIXED_TILES.items():
        assert s.board[rc] == m
    assert sorted(sum(s.decks, [])) == list(range(NUM_TREASURES))
    assert all(len(d) == 6 for d in s.decks)


def test_rotation():
    assert rotate(1) == 2 and rotate(8) == 1 and rotate(0b0011, 4) == 0b0011


@pytest.mark.parametrize("slot", range(12))
def test_shift_conserves_tiles_and_wraps_pawns(slot):
    rng = np.random.default_rng(slot)
    s = GameState.new_game(4, rng=rng)
    before = tile_multiset(s)
    # put a pawn on the tile that will be ejected
    side, line = slot // 3, [1, 3, 5][slot % 3]
    eject = {0: (6, line), 1: (line, 0), 2: (0, line), 3: (line, 6)}[side]
    inserted = {0: (0, line), 1: (line, 6), 2: (6, line), 3: (line, 0)}[side]
    s.pos[1] = eject
    s.apply_shift(slot, 1)
    assert tile_multiset(s) == before
    assert tuple(s.pos[1]) == inserted
    assert s.phase == PHASE_MOVE
    s.apply_move(*s.pos[0])
    assert opposite_slot(slot) not in s.legal_slots()
    with pytest.raises(ValueError):
        s.apply_shift(opposite_slot(slot), 0)


def test_reachability_is_symmetric():
    s = GameState.new_game(4, rng=np.random.default_rng(3))
    from labyrinth.core import reachable_from
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            reach = reachable_from(s.board, (r, c))
            assert reach[r, c]
            for rr, cc in np.argwhere(reach):
                assert reachable_from(s.board, (int(rr), int(cc)))[r, c]


def test_collect_and_win():
    s = GameState.new_game(2, cards_per_player=1, rng=np.random.default_rng(1))
    t = s.target(0)
    # teleport the treasure next to the pawn by editing the state
    loc = s.target_location(0)
    if loc is None:
        s.spare_treasure, s.treasure[0, 0] = s.treasure[0, 0], t
    else:
        s.treasure[loc], s.treasure[0, 0] = s.treasure[0, 0], t
    s.phase = PHASE_MOVE
    s.apply_move(0, 0)
    assert s.found[0] == 1 and s.target(0) == -1
    s.current, s.phase = 0, PHASE_MOVE
    s.pos[0] = (0, 0)
    s.apply_move(0, 0)
    assert s.winner == 0 and s.done


def test_determinize_keeps_information_set():
    rng = np.random.default_rng(0)
    env = LabyrinthEnv(seed=0)
    env.reset()
    agents = [GreedyAgent(i) for i in range(4)]
    for _ in range(40):
        p = env.current_player
        env.step(agents[p].act(env.observe(p), env.state, p))
    s = env.state
    d = s.determinize(0, rng)
    assert d.target(0) == s.target(0)
    assert d.collected_treasures() == s.collected_treasures()
    assert [len(x) for x in d.decks] == [len(x) for x in s.decks]
    assert sorted(sum(d.decks, [])) == sorted(sum(s.decks, []))
    assert np.array_equal(d.board, s.board)
    obs_s = env.observe(0)
    env.state = d
    obs_d = env.observe(0)
    for k in obs_s:  # the observer cannot tell the two states apart
        assert np.array_equal(obs_s[k], obs_d[k])


def test_partial_observation_hides_opponent_targets():
    env = LabyrinthEnv(seed=0)
    obs, _ = env.reset()
    names = plane_names(env.obs_config)
    for i in range(1, 4):
        assert obs["planes"][names.index(f"opp_target_rel{i}")].sum() == 0
    full = LabyrinthEnv(seed=0, obs_config=ObservationConfig(observability="full"))
    obs_f, _ = full.reset()
    assert obs_f["planes"].shape == obs["planes"].shape
    vn = vector_names(full.obs_config)
    seen = sum(obs_f["planes"][names.index(f"opp_target_rel{i}")].sum()
               + obs_f["vector"][vn.index(f"spare_is_opp_target_rel{i}")] for i in range(1, 4))
    assert seen == 3


def test_fog_of_war():
    env = LabyrinthEnv(seed=0, obs_config=ObservationConfig(view_radius=1))
    obs, _ = env.reset()
    names = plane_names(env.obs_config)
    assert obs["planes"][names.index("visible")].sum() == 4  # corner pawn, radius 1
    assert obs["planes"][names.index("open_N")][4:, 4:].sum() == 0


@pytest.mark.parametrize("n", [2, 3, 4])
def test_random_games_run_and_are_zero_sum(n):
    env = LabyrinthEnv(num_players=n, max_turns=200, reward_config=RewardConfig(treasure=0.0))
    for seed in range(3):
        info = play_game(env, [RandomAgent(seed + p) for p in range(n)], seed=seed)
        assert abs(info["returns"].sum()) < 1e-5


def test_greedy_beats_random():
    env = LabyrinthEnv(max_turns=400)
    wins = sum(play_game(env, [GreedyAgent(g)] + [RandomAgent(g + p) for p in range(1, 4)], seed=g)["winner"] == 0
               for g in range(10))
    assert wins >= 8


def test_masks_and_obs_shapes():
    env = LabyrinthEnv(seed=1)
    obs, _ = env.reset()
    ag = RandomAgent(0)
    for _ in range(200):
        assert obs["planes"].shape == env.planes_shape
        assert obs["vector"].shape == env.vector_shape
        assert obs["action_mask"].shape == (NUM_ACTIONS,) and obs["action_mask"].any()
        obs, r, term, trunc, _ = env.step(ag.act(obs))
        if term or trunc:
            obs, _ = env.reset()


def test_illegal_action_raises():
    env = LabyrinthEnv(seed=0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(48)  # a move during the shift phase
    env.step(encode_shift(0, 0))
    with pytest.raises(ValueError):
        env.step(0)  # a shift during the move phase


def test_pettingzoo_api():
    pz = pytest.importorskip("pettingzoo")
    from pettingzoo.test import api_test
    from labyrinth.wrappers import LabyrinthAEC
    api_test(LabyrinthAEC(max_turns=60), num_cycles=200)


def test_gymnasium_single_agent():
    pytest.importorskip("gymnasium")
    from labyrinth.wrappers import SingleAgentEnv
    env = SingleAgentEnv([GreedyAgent(1), GreedyAgent(2), GreedyAgent(3)], seat=None, max_turns=100)
    obs, _ = env.reset(seed=0)
    ag = RandomAgent(0)
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(ag.act(obs))
        if term or trunc:
            obs, _ = env.reset()
