"""Scripted baseline agents.

All agents implement ``act(obs, state, player) -> action``. ``obs`` is the
player's (partial) observation; ``state`` is passed for convenience, but the
scripted agents below only read information the player legitimately has (the
public board and its own current target), so they are valid POMDP policies.
"""
from __future__ import annotations

import numpy as np

from .core import NUM_SHIFT_ACTIONS, PHASE_SHIFT, GameState, encode_move, encode_shift
from .tiles import BOARD_SIZE, canonical_rotations, rotate


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, obs: dict, state: GameState | None = None, player: int | None = None) -> int:
        return int(self.rng.choice(np.flatnonzero(obs["action_mask"])))


class GreedyAgent:
    """One-ply lookahead: try every (slot, rotation), then move as close as
    possible to the current target (Manhattan distance). Picks the shift that
    lets it reach the target, otherwise the one minimising the distance.
    Ties are broken at random. Ignores opponents entirely."""

    name = "greedy"

    def __init__(self, seed: int | None = None, epsilon: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.epsilon = epsilon

    def act(self, obs: dict, state: GameState, player: int | None = None) -> int:
        if self.epsilon and self.rng.random() < self.epsilon:
            return int(self.rng.choice(np.flatnonzero(obs["action_mask"])))
        best = self.best_actions(state, player)
        return int(best[int(self.rng.integers(len(best)))])

    def best_actions(self, state: GameState, player: int | None = None) -> list[int]:
        """All flat actions tied for the best greedy score (used as soft
        imitation targets). Shift actions include equivalent rotations."""
        player = state.current if player is None else player
        if state.phase == PHASE_SHIFT:
            best, best_score = [], -np.inf
            for slot in state.legal_slots():
                for rot in canonical_rotations(state.spare):
                    s2 = state.clone()
                    s2.apply_shift(slot, rot)
                    score = self._move_scores(s2, player)[0].max()
                    if score > best_score + 1e-9:
                        best, best_score = [(slot, rot)], score
                    elif abs(score - best_score) <= 1e-9:
                        best.append((slot, rot))
            same = {r: [k for k in range(4) if rotate(state.spare, k) == rotate(state.spare, r)] for r in range(4)}
            return sorted({encode_shift(slot, k) for slot, rot in best for k in same[rot]})
        scores, cells = self._move_scores(state, player)
        top = np.flatnonzero(scores >= scores.max() - 1e-9)
        return [encode_move(int(cells[k][0]), int(cells[k][1])) for k in top]

    def _move_scores(self, s: GameState, player: int):
        reach = s.reachable(player)
        cells = np.argwhere(reach)
        tgt = s.target_location(player)
        if tgt is None:  # target sits on the spare tile: just stay mobile
            scores = reach.sum() / 49.0 - 20 + 0.0 * cells[:, 0]
        else:
            d = np.abs(cells[:, 0] - tgt[0]) + np.abs(cells[:, 1] - tgt[1])
            scores = np.where(d == 0, 100.0, -d.astype(float)) + reach.sum() / 1000.0
        return scores, cells


def play_game(env, agents, seed: int | None = None, render_every: int = 0):
    """Play one game with a list of agents (one per seat). Returns info."""
    obs, info = env.reset(seed=seed)
    total = np.zeros(env.num_players)
    while True:
        p = env.current_player
        a = agents[p].act(obs, env.state, p)
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            info["returns"] = total
            info["truncated"] = trunc
            return info


__all__ = ["RandomAgent", "GreedyAgent", "play_game", "NUM_SHIFT_ACTIONS", "BOARD_SIZE"]
