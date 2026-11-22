"""Multi-agent, turn-based RL environment for Labyrinth.

The core API is a minimal turn-based (AEC-style) interface that is easy to
drive from a self-play trainer::

    env = LabyrinthEnv(num_players=4)
    obs, info = env.reset(seed=0)
    while True:
        p = env.current_player
        action = policy[p](obs)                    # obs is from p's viewpoint
        obs, rewards, terminated, truncated, info = env.step(action)
        # rewards: np.ndarray (num_players,) produced by this transition
        if terminated or truncated:
            break

Wrappers for PettingZoo (AEC) and Gymnasium (single agent vs fixed
opponents) live in :mod:`labyrinth.wrappers`.

Each decision is one *phase* of a turn (SHIFT then MOVE), and actions are a flat
``Discrete(97)`` with a legality mask: indices 0..47 = (slot*4 + rotation),
48..96 = 48 + row*7 + col.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import NUM_ACTIONS, GameState
from .observation import ObservationConfig, encode, observation_shapes
from .tiles import HOME


@dataclass(frozen=True)
class RewardConfig:
    win: float = 1.0
    zero_sum: bool = True             # losers share -win (sum of rewards = 0)
    treasure: float = 0.0             # bonus when a treasure card is collected
    distance_shaping: float = 0.0     # potential-based shaping coefficient
    truncation: str = "progress"      # "draw" | "progress" (rank by cards found)


class LabyrinthEnv:
    metadata = {"render_modes": ["rgb_array", "ansi"], "name": "labyrinth_v0"}

    def __init__(
        self,
        num_players: int = 4,
        cards_per_player: int | None = None,
        obs_config: ObservationConfig = ObservationConfig(),
        reward_config: RewardConfig = RewardConfig(),
        max_turns: int = 400,
        render_mode: str | None = None,
        viewer=None,
        seed: int | None = None,
    ):
        self.num_players = num_players
        self.cards_per_player = cards_per_player
        self.obs_config = obs_config
        self.reward_config = reward_config
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.viewer = viewer
        self.rng = np.random.default_rng(seed)
        self.planes_shape, self.vector_shape = observation_shapes(obs_config)
        self.num_actions = NUM_ACTIONS
        self.state: GameState | None = None
        self.episode = 0
        self._renderer = None

    # ----------------------------------------------------------------- API
    @property
    def current_player(self) -> int:
        return self.state.current

    def reset(self, seed: int | None = None, state: GameState | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = state.clone() if state is not None else GameState.new_game(
            self.num_players, self.cards_per_player, self.rng
        )
        self.episode += 1
        self._notify_viewer(reset=True)
        return self.observe(self.current_player), self._info()

    def observe(self, player: int) -> dict:
        return encode(self.state, player, self.obs_config, self.max_turns)

    def action_mask(self) -> np.ndarray:
        return self.state.legal_action_mask()

    def step(self, action: int):
        s = self.state
        if s.done:
            raise RuntimeError("episode is over, call reset()")
        action = int(action)
        if not s.legal_action_mask()[action]:
            raise ValueError(f"illegal action {action} in phase {s.phase}")
        cfg = self.reward_config
        phi_before = self._potentials() if cfg.distance_shaping else None
        events = s.step(action)

        rewards = np.zeros(self.num_players, dtype=np.float32)
        if events["collected"] is not None and cfg.treasure:
            rewards[events["player"]] += cfg.treasure
        if cfg.distance_shaping:
            rewards += cfg.distance_shaping * (self._potentials() - phi_before)

        terminated = s.done
        truncated = (not terminated) and s.turn >= self.max_turns
        if terminated:
            rewards += self._outcome(winner_scores=np.eye(self.num_players)[s.winner])
        elif truncated:
            if cfg.truncation == "progress":
                rewards += self._outcome(winner_scores=self._progress_scores())
        self._notify_viewer(rewards=rewards, terminated=terminated, truncated=truncated)
        obs = self.observe(s.current)
        info = self._info()
        info["events"] = events
        return obs, rewards, terminated, truncated, info

    # ------------------------------------------------------------- helpers
    def _outcome(self, winner_scores: np.ndarray) -> np.ndarray:
        """Map per-player scores in [0,1] (summing to 1) to terminal rewards."""
        cfg = self.reward_config
        w = winner_scores.astype(np.float32)
        if cfg.zero_sum:  # winner +win, losers -win/(P-1); sums to 0
            return cfg.win * (w - (1.0 - w) / (self.num_players - 1))
        return cfg.win * w

    def _progress_scores(self) -> np.ndarray:
        s = self.state
        prog = np.array([s.found[p] / max(len(s.decks[p]), 1) for p in range(self.num_players)])
        best = prog == prog.max()
        return best / best.sum()  # leaders share the "win"

    def _potentials(self) -> np.ndarray:
        """Phi_p = cards found - normalised Manhattan distance to target."""
        s = self.state
        phi = np.zeros(self.num_players, dtype=np.float32)
        for p in range(self.num_players):
            loc = s.target_location(p)
            d = 12 if loc is None else abs(loc[0] - s.pos[p][0]) + abs(loc[1] - s.pos[p][1])
            phi[p] = s.found[p] + (s.target(p) == HOME) - d / 12.0
        return phi

    def _info(self) -> dict:
        s = self.state
        return {
            "current_player": s.current,
            "phase": s.phase,
            "turn": s.turn,
            "found": s.found.copy(),
            "winner": s.winner,
        }

    def _notify_viewer(self, **extra) -> None:
        if self.viewer is not None:
            self.viewer.push(self.state, episode=self.episode, **{
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in extra.items()
            })

    # -------------------------------------------------------------- render
    def render(self):
        if self.render_mode == "ansi":
            from .render import render_ansi
            return render_ansi(self.state)
        if self.render_mode == "rgb_array":
            from .render import render_rgb
            return render_rgb(self.state)
        return None

    def clone_state(self) -> GameState:
        return self.state.clone()
