"""Observation encoding: from the god-view GameState to a player's observation.

Labyrinth is naturally a POMDP: the board is public, but each player only
knows the top card of its *own* treasure deck. Opponents' targets are hidden.

The encoder is configurable so the same environment covers a spectrum of
observability regimes:

* ``"full"``     -- MDP: every player's current target is visible (oracle /
                    privileged information, useful for teacher networks and
                    for "oracle guiding" curricula as in Suphx).
* ``"partial"``  -- the official rules (default).
* ``view_radius`` -- additional fog of war around the own pawn (Chebyshev
                    radius); cells outside are blanked, and opponents outside
                    the view are invisible. This forces memory (recurrent
                    policies), as in AlphaStar's camera-limited setting.

The tensor layout is *identical* across regimes: privileged channels are just
zeroed when not observable. This makes it possible to distil a full-info
teacher into a partial-info student with the same architecture.

All tensors are **ego-centric**: player slot 0 is always the observer, then the
opponents in turn order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import NUM_ACTIONS, PHASE_SHIFT, GameState
from .tiles import BOARD_SIZE, HOME, NUM_SLOTS, NUM_TREASURES, SIDES, opposite_slot

MAX_PLAYERS = 4


@dataclass(frozen=True)
class ObservationConfig:
    observability: str = "partial"      # "partial" | "full"
    view_radius: int | None = None      # None = whole board visible
    treasure_id_planes: bool = True     # one plane per treasure identity
    reachability_plane: bool = True     # derived feature: reachable cells

    def __post_init__(self):
        if self.observability not in ("partial", "full"):
            raise ValueError("observability must be 'partial' or 'full'")


def plane_names(cfg: ObservationConfig) -> list[str]:
    names = [f"open_{s}" for s in "NESW"]
    names += ["has_treasure", "treasure_collected", "own_target", "own_home"]
    names += [f"pawn_rel{i}" for i in range(MAX_PLAYERS)]
    names += [f"home_rel{i}" for i in range(MAX_PLAYERS)]
    if cfg.reachability_plane:
        names += ["reachable"]
    names += ["visible"]
    names += [f"opp_target_rel{i}" for i in range(1, MAX_PLAYERS)]  # privileged
    if cfg.treasure_id_planes:
        names += [f"treasure_{t}" for t in range(NUM_TREASURES)]
    return names


def vector_names(cfg: ObservationConfig) -> list[str]:
    names = ["phase_shift", "phase_move"]
    names += [f"spare_open_{s}" for s in "NESW"]
    names += ["spare_has_treasure", "spare_is_own_target", "spare_collected"]
    names += [f"forbidden_slot_{s}" for s in range(NUM_SLOTS)]
    names += [f"found_frac_rel{i}" for i in range(MAX_PLAYERS)]
    names += [f"remaining_frac_rel{i}" for i in range(MAX_PLAYERS)]
    names += [f"going_home_rel{i}" for i in range(MAX_PLAYERS)]
    names += [f"active_rel{i}" for i in range(MAX_PLAYERS)]
    names += [f"own_target_{t}" for t in range(NUM_TREASURES)] + ["own_target_home"]
    names += ["turn_frac"]
    names += [f"spare_is_opp_target_rel{i}" for i in range(1, MAX_PLAYERS)]  # privileged
    if cfg.treasure_id_planes:
        names += [f"spare_treasure_{t}" for t in range(NUM_TREASURES)]
    return names


def observation_shapes(cfg: ObservationConfig) -> tuple[tuple[int, int, int], tuple[int]]:
    return (len(plane_names(cfg)), BOARD_SIZE, BOARD_SIZE), (len(vector_names(cfg)),)


def encode(state: GameState, player: int, cfg: ObservationConfig, max_turns: int = 400) -> dict:
    """Return {"planes", "vector", "action_mask"} as seen by ``player``."""
    P = state.num_players
    rel = [(player + i) % P for i in range(P)]  # ego-centric seat order
    planes = np.zeros(observation_shapes(cfg)[0], dtype=np.float32)
    vec = np.zeros(observation_shapes(cfg)[1], dtype=np.float32)
    pn = {n: i for i, n in enumerate(plane_names(cfg))}
    vn = {n: i for i, n in enumerate(vector_names(cfg))}

    # ---- visibility (fog of war)
    if cfg.view_radius is None:
        visible = np.ones((BOARD_SIZE, BOARD_SIZE), dtype=bool)
    else:
        r0, c0 = state.pos[player]
        rr, cc = np.mgrid[0:BOARD_SIZE, 0:BOARD_SIZE]
        visible = np.maximum(abs(rr - r0), abs(cc - c0)) <= cfg.view_radius
    planes[pn["visible"]] = visible

    # ---- public board
    for k, bit in enumerate(SIDES):
        planes[k] = (state.board & bit) > 0
    has_t = state.treasure >= 0
    planes[pn["has_treasure"]] = has_t
    collected = state.collected_treasures()
    if collected:
        planes[pn["treasure_collected"]] = np.isin(state.treasure, list(collected))
    if cfg.treasure_id_planes:
        base = pn["treasure_0"]
        rs, cs = np.nonzero(has_t)
        planes[base + state.treasure[rs, cs].astype(int), rs, cs] = 1.0

    # ---- own information
    own_t = state.target(player)
    loc = state.target_location(player)
    if loc is not None:
        planes[pn["own_target"]][loc] = 1.0
    planes[pn["own_home"]][state.home(player)] = 1.0
    if cfg.reachability_plane and state.current == player:
        planes[pn["reachable"]] = state.reachable(player)

    # ---- seats (relative order)
    for i, p in enumerate(rel):
        r, c = state.pos[p]
        if p == player or visible[r, c]:
            planes[pn[f"pawn_rel{i}"], r, c] = 1.0
        planes[pn[f"home_rel{i}"]][state.home(p)] = 1.0
        n_cards = max(len(state.decks[p]), 1)
        vec[vn[f"found_frac_rel{i}"]] = state.found[p] / n_cards
        vec[vn[f"remaining_frac_rel{i}"]] = state.remaining(p) / n_cards
        vec[vn[f"going_home_rel{i}"]] = float(state.remaining(p) == 0)
        vec[vn[f"active_rel{i}"]] = 1.0

    # ---- privileged information (only in the fully observable regime)
    if cfg.observability == "full":
        for i, p in enumerate(rel[1:], start=1):
            l2 = state.target_location(p)
            if l2 is not None:
                planes[pn[f"opp_target_rel{i}"]][l2] = 1.0
            else:
                vec[vn[f"spare_is_opp_target_rel{i}"]] = 1.0

    # fog: blank every spatial channel outside the view (except own home and
    # the visibility mask itself, which the agent always knows)
    if cfg.view_radius is not None:
        keep = {pn["visible"], pn["own_home"]} | {pn[f"home_rel{i}"] for i in range(MAX_PLAYERS)}
        keep.add(pn["pawn_rel0"])
        for k in range(planes.shape[0]):
            if k not in keep:
                planes[k] *= visible

    # ---- vector features
    vec[vn["phase_shift"] if state.phase == PHASE_SHIFT else vn["phase_move"]] = 1.0
    for k, bit in enumerate(SIDES):
        vec[vn[f"spare_open_{'NESW'[k]}"]] = float(state.spare & bit > 0)
    if state.spare_treasure >= 0:
        vec[vn["spare_has_treasure"]] = 1.0
        vec[vn["spare_is_own_target"]] = float(state.spare_treasure == own_t)
        vec[vn["spare_collected"]] = float(state.spare_treasure in collected)
        if cfg.treasure_id_planes:
            vec[vn[f"spare_treasure_{state.spare_treasure}"]] = 1.0
    if state.last_slot >= 0:
        vec[vn[f"forbidden_slot_{opposite_slot(state.last_slot)}"]] = 1.0
    vec[vn["own_target_home"] if own_t == HOME else vn[f"own_target_{own_t}"]] = 1.0
    vec[vn["turn_frac"]] = min(state.turn / max_turns, 1.0)

    mask = state.legal_action_mask() if state.current == player else np.zeros(NUM_ACTIONS, bool)
    return {"planes": planes, "vector": vec, "action_mask": mask}
