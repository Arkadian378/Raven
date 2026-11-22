"""Rules engine for Ravensburger's *Labyrinth* (Das verrückte Labyrinth).

The engine is deliberately independent of any RL library: it only knows the
rules. A turn of the active player has two phases:

1. SHIFT: insert the spare tile (choose one of 12 slots and one of 4
   rotations); the tile pushed off the opposite edge becomes the new spare.
   Pawns standing on the ejected tile wrap around onto the inserted tile.
   The slot that would directly undo the previous insertion is forbidden.
2. MOVE: move the pawn to any cell reachable through connected corridors
   (staying in place is always allowed).

When a pawn ends its move on the tile holding its current treasure card, the
card is collected and the next card in its deck becomes the target. After the
last card the target is the player's starting corner; the first player to get
back there wins.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .tiles import (
    BOARD_SIZE,
    DIRECTIONS,
    FIXED_TILES,
    FIXED_TREASURE_CELLS,
    HOME,
    MOVABLE_TILE_SPEC,
    NO_TREASURE,
    NUM_SLOTS,
    NUM_TREASURES,
    ODD_LINES,
    START_CORNERS,
    opposite_slot,
    rotate,
)

PHASE_SHIFT = 0
PHASE_MOVE = 1

NUM_SHIFT_ACTIONS = NUM_SLOTS * 4  # 48
NUM_MOVE_ACTIONS = BOARD_SIZE * BOARD_SIZE  # 49
NUM_ACTIONS = NUM_SHIFT_ACTIONS + NUM_MOVE_ACTIONS  # 97 (flat, phase-masked)


def encode_shift(slot: int, rotation: int) -> int:
    return slot * 4 + rotation


def decode_shift(action: int) -> tuple[int, int]:
    return divmod(action, 4)


def encode_move(row: int, col: int) -> int:
    return NUM_SHIFT_ACTIONS + row * BOARD_SIZE + col


def decode_move(action: int) -> tuple[int, int]:
    return divmod(action - NUM_SHIFT_ACTIONS, BOARD_SIZE)


@dataclass
class GameState:
    """Complete (god-view) state of a game. Mutable; use :meth:`clone`."""

    board: np.ndarray            # (7,7) uint8 opening masks
    treasure: np.ndarray         # (7,7) int8 treasure id or -1
    spare: int                   # opening mask of the spare tile
    spare_treasure: int          # treasure id on the spare tile or -1
    pos: np.ndarray              # (P,2) int pawn positions
    decks: list[list[int]]       # each player's (secret) ordered treasure deck
    found: np.ndarray            # (P,) number of cards collected
    current: int = 0
    phase: int = PHASE_SHIFT
    last_slot: int = -1          # slot used by the previous insertion
    turn: int = 0                # number of completed player turns
    winner: int = -1
    history: list = field(default_factory=list)

    # ------------------------------------------------------------------ setup
    @classmethod
    def new_game(
        cls,
        num_players: int = 4,
        cards_per_player: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "GameState":
        if not 2 <= num_players <= 4:
            raise ValueError("Labyrinth is played by 2 to 4 players")
        rng = rng if rng is not None else np.random.default_rng()
        if cards_per_player is None:
            cards_per_player = NUM_TREASURES // num_players
        if cards_per_player * num_players > NUM_TREASURES:
            raise ValueError("not enough treasure cards")

        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.uint8)
        treasure = np.full((BOARD_SIZE, BOARD_SIZE), NO_TREASURE, dtype=np.int8)
        for rc, mask in FIXED_TILES.items():
            board[rc] = mask
        for i, rc in enumerate(FIXED_TREASURE_CELLS):
            treasure[rc] = i

        tiles = list(MOVABLE_TILE_SPEC)
        order = rng.permutation(len(tiles))
        free_cells = [
            (r, c)
            for r in range(BOARD_SIZE)
            for c in range(BOARD_SIZE)
            if (r, c) not in FIXED_TILES
        ]
        assert len(free_cells) == len(tiles) - 1
        for cell, idx in zip(free_cells, order[:-1]):
            mask, t = tiles[idx]
            board[cell] = rotate(mask, int(rng.integers(4)))
            treasure[cell] = t
        spare_mask, spare_t = tiles[order[-1]]

        cards = rng.permutation(NUM_TREASURES)
        decks = [
            [int(x) for x in cards[p * cards_per_player:(p + 1) * cards_per_player]]
            for p in range(num_players)
        ]
        pos = np.array(START_CORNERS[:num_players], dtype=np.int64)
        if num_players == 2:  # opposite corners, as in the official rules
            pos = np.array([START_CORNERS[0], START_CORNERS[2]], dtype=np.int64)
        return cls(
            board=board,
            treasure=treasure,
            spare=int(rotate(spare_mask, int(rng.integers(4)))),
            spare_treasure=int(spare_t),
            pos=pos,
            decks=decks,
            found=np.zeros(num_players, dtype=np.int64),
        )

    def clone(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            treasure=self.treasure.copy(),
            spare=self.spare,
            spare_treasure=self.spare_treasure,
            pos=self.pos.copy(),
            decks=[list(d) for d in self.decks],
            found=self.found.copy(),
            current=self.current,
            phase=self.phase,
            last_slot=self.last_slot,
            turn=self.turn,
            winner=self.winner,
            history=list(self.history),
        )

    # -------------------------------------------------------------- queries
    @property
    def num_players(self) -> int:
        return len(self.decks)

    @property
    def done(self) -> bool:
        return self.winner >= 0

    def home(self, player: int) -> tuple[int, int]:
        return int(self.pos_home[player][0]), int(self.pos_home[player][1])

    @property
    def pos_home(self) -> list[tuple[int, int]]:
        if self.num_players == 2:
            return [START_CORNERS[0], START_CORNERS[2]]
        return list(START_CORNERS[: self.num_players])

    def target(self, player: int) -> int:
        """Current target treasure id, or HOME when the deck is exhausted."""
        k = int(self.found[player])
        deck = self.decks[player]
        return deck[k] if k < len(deck) else HOME

    def target_location(self, player: int) -> tuple[int, int] | None:
        """Board cell of the current target, or None if it is on the spare."""
        t = self.target(player)
        if t == HOME:
            return self.home(player)
        hits = np.argwhere(self.treasure == t)
        if len(hits) == 0:
            return None
        return int(hits[0][0]), int(hits[0][1])

    def remaining(self, player: int) -> int:
        return len(self.decks[player]) - int(self.found[player])

    def collected_treasures(self) -> set[int]:
        """Treasures already found by anyone (public information)."""
        out = set()
        for p, deck in enumerate(self.decks):
            out.update(deck[: int(self.found[p])])
        return out

    def legal_slots(self) -> list[int]:
        forbidden = opposite_slot(self.last_slot) if self.last_slot >= 0 else -1
        return [s for s in range(NUM_SLOTS) if s != forbidden]

    def reachable(self, player: int | None = None) -> np.ndarray:
        """Boolean (7,7) map of cells reachable from the player's pawn."""
        player = self.current if player is None else player
        return reachable_from(self.board, tuple(int(x) for x in self.pos[player]))

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        if self.done:
            return mask
        if self.phase == PHASE_SHIFT:
            for s in self.legal_slots():
                mask[s * 4:(s + 1) * 4] = True
        else:
            mask[NUM_SHIFT_ACTIONS:] = self.reachable().reshape(-1)
        return mask

    # ------------------------------------------------------------ dynamics
    def apply_shift(self, slot: int, rotation: int) -> None:
        if self.phase != PHASE_SHIFT or self.done:
            raise RuntimeError("not in shift phase")
        if slot not in self.legal_slots():
            raise ValueError(f"slot {slot} is forbidden (undoes previous push)")
        tile = rotate(self.spare, rotation)
        t_in = self.spare_treasure
        b, tr = self.board, self.treasure
        line = ODD_LINES[slot % 3]
        side = slot // 3
        n = BOARD_SIZE - 1
        if side == 0:  # top: push down column `line`
            ej, ej_t = int(b[n, line]), int(tr[n, line])
            b[1:, line] = b[:-1, line].copy(); tr[1:, line] = tr[:-1, line].copy()
            b[0, line], tr[0, line] = tile, t_in
            self._move_pawns(axis=1, line=line, delta=+1)
        elif side == 1:  # right: push left row `line`
            ej, ej_t = int(b[line, 0]), int(tr[line, 0])
            b[line, :-1] = b[line, 1:].copy(); tr[line, :-1] = tr[line, 1:].copy()
            b[line, n], tr[line, n] = tile, t_in
            self._move_pawns(axis=0, line=line, delta=-1)
        elif side == 2:  # bottom: push up column `line`
            ej, ej_t = int(b[0, line]), int(tr[0, line])
            b[:-1, line] = b[1:, line].copy(); tr[:-1, line] = tr[1:, line].copy()
            b[n, line], tr[n, line] = tile, t_in
            self._move_pawns(axis=1, line=line, delta=-1)
        else:  # left: push right row `line`
            ej, ej_t = int(b[line, n]), int(tr[line, n])
            b[line, 1:] = b[line, :-1].copy(); tr[line, 1:] = tr[line, :-1].copy()
            b[line, 0], tr[line, 0] = tile, t_in
            self._move_pawns(axis=0, line=line, delta=+1)
        self.spare, self.spare_treasure = ej, ej_t
        self.last_slot = slot
        self.phase = PHASE_MOVE
        self.history.append(("shift", self.current, slot, rotation))

    def _move_pawns(self, axis: int, line: int, delta: int) -> None:
        """Shift pawns on a pushed row (axis=0) / column (axis=1), wrapping."""
        for p in range(self.num_players):
            r, c = self.pos[p]
            if axis == 1 and c == line:
                self.pos[p, 0] = (r + delta) % BOARD_SIZE
            elif axis == 0 and r == line:
                self.pos[p, 1] = (c + delta) % BOARD_SIZE

    def apply_move(self, row: int, col: int) -> dict:
        """Move the current pawn; returns a dict of events for reward shaping."""
        if self.phase != PHASE_MOVE or self.done:
            raise RuntimeError("not in move phase")
        p = self.current
        if not self.reachable(p)[row, col]:
            raise ValueError(f"cell {(row, col)} not reachable")
        self.pos[p] = (row, col)
        events = {"player": p, "collected": None, "won": False}
        t = self.target(p)
        if t != HOME and int(self.treasure[row, col]) == t:
            self.found[p] += 1
            events["collected"] = t
        elif t == HOME and (row, col) == self.home(p):
            self.winner = p
            events["won"] = True
        self.history.append(("move", p, row, col))
        self.phase = PHASE_SHIFT
        self.turn += 1
        if not self.done:
            self.current = (p + 1) % self.num_players
        return events

    def step(self, action: int) -> dict:
        """Apply a flat action in [0, 97). Returns shaping events."""
        if action < NUM_SHIFT_ACTIONS:
            self.apply_shift(*decode_shift(action))
            return {"player": self.current, "collected": None, "won": False}
        return self.apply_move(*decode_move(action))

    # -------------------------------------------------------- partial info
    def determinize(self, player: int, rng: np.random.Generator) -> "GameState":
        """Sample a full state consistent with ``player``'s information set.

        Hidden information in Labyrinth is the content of the treasure decks:
        a player only sees the top card of its own deck and the cards already
        collected by everybody. All other cards (opponents' current targets,
        future cards of every deck, including its own) are reshuffled uniformly
        among the unseen treasures. Used for determinized / information-set
        MCTS (e.g. PIMC, IS-MCTS) on top of the POMDP.
        """
        s = self.clone()
        known = set(s.collected_treasures())
        own_t = s.target(player)
        if own_t != HOME:
            known.add(own_t)
        pool = [t for t in range(NUM_TREASURES) if t not in known]
        # also treasures never dealt are unseen; keep deck sizes unchanged
        pool = list(rng.permutation(pool))
        for p in range(s.num_players):
            k = int(s.found[p])
            keep = k + 1 if p == player else k
            need = len(s.decks[p]) - keep
            if need > 0:
                s.decks[p] = s.decks[p][:keep] + [int(x) for x in pool[:need]]
                pool = pool[need:]
        return s

    # ----------------------------------------------------------- export
    def to_dict(self, reveal: bool = True) -> dict:
        """JSON-serialisable snapshot (used by the live viewer)."""
        return {
            "board": self.board.astype(int).tolist(),
            "treasure": self.treasure.astype(int).tolist(),
            "spare": int(self.spare),
            "spare_treasure": int(self.spare_treasure),
            "pos": self.pos.astype(int).tolist(),
            "home": [list(h) for h in self.pos_home],
            "found": self.found.astype(int).tolist(),
            "deck_sizes": [len(d) for d in self.decks],
            "collected": sorted(int(t) for t in self.collected_treasures()),
            "targets": [int(self.target(p)) for p in range(self.num_players)] if reveal else None,
            "current": int(self.current),
            "phase": int(self.phase),
            "last_slot": int(self.last_slot),
            "turn": int(self.turn),
            "winner": int(self.winner),
            "last_action": list(self.history[-1]) if self.history else None,
        }


def reachable_from(board: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    seen = np.zeros(board.shape, dtype=bool)
    seen[start] = True
    q = deque([start])
    n = board.shape[0]
    while q:
        r, c = q.popleft()
        m = board[r, c]
        for bit, dr, dc, opp in DIRECTIONS:
            if not m & bit:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and not seen[rr, cc] and board[rr, cc] & opp:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen
