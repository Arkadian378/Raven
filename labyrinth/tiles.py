"""Tile geometry for the Labyrinth board.

A tile is encoded as a 4-bit mask of its open sides: N=1, E=2, S=4, W=8.
Rotating a tile by 90 degrees clockwise is a cyclic left shift of that mask.
"""
from __future__ import annotations

N, E, S, W = 1, 2, 4, 8
SIDES = (N, E, S, W)
SIDE_NAMES = ("N", "E", "S", "W")

# (side bit, d_row, d_col, opposite side bit)
DIRECTIONS = ((N, -1, 0, S), (E, 0, 1, W), (S, 1, 0, N), (W, 0, -1, E))

STRAIGHT = N | S
CORNER = N | E
TJUNCTION = N | E | S

BOARD_SIZE = 7
NUM_TREASURES = 24
HOME = -1  # pseudo-target once all treasure cards have been found
NO_TREASURE = -1


def rotate(mask: int, k: int = 1) -> int:
    """Rotate an opening mask clockwise by k quarter turns."""
    k %= 4
    for _ in range(k):
        mask = ((mask << 1) | (mask >> 3)) & 0xF
    return mask


def shape_of(mask: int) -> str:
    n = bin(mask).count("1")
    if n == 3:
        return "T"
    if mask in (N | S, E | W):
        return "I"
    return "L"


def canonical_rotations(mask: int) -> list[int]:
    """Rotation counts producing distinct tiles (I has 2, L and T have 4)."""
    seen, out = set(), []
    for k in range(4):
        m = rotate(mask, k)
        if m not in seen:
            seen.add(m)
            out.append(k)
    return out


# --------------------------------------------------------------------------
# Fixed tiles (the 16 tiles glued to the board at even coordinates), copied
# from the Ravensburger board (2006 edition). Edge T-junctions have their
# closed side facing outward; the layout is invariant under 90-degree
# rotations of the board, so the four starting corners are equivalent.
# --------------------------------------------------------------------------
FIXED_TILES: dict[tuple[int, int], int] = {
    # starting corners
    (0, 0): E | S,
    (0, 6): S | W,
    (6, 6): N | W,
    (6, 0): N | E,
    # edges (closed side facing outward)
    (0, 2): E | S | W, (0, 4): E | S | W,
    (2, 6): N | S | W, (4, 6): N | S | W,
    (6, 2): N | E | W, (6, 4): N | E | W,
    (2, 0): N | E | S, (4, 0): N | E | S,
    # inner ring
    (2, 2): N | E | S,
    (2, 4): E | S | W,
    (4, 4): N | S | W,
    (4, 2): N | E | W,
}

START_CORNERS = ((0, 0), (0, 6), (6, 6), (6, 0))  # clockwise: yellow, red, blue, green

# Fixed T-junctions (all except the 4 corners) carry treasures 0..11.
FIXED_TREASURE_CELLS = [rc for rc in sorted(FIXED_TILES) if rc not in START_CORNERS]

# Movable tiles: 34 in total (33 on the board + 1 spare), as in the box:
#   12 straight, 16 corners (6 with an animal), 6 T-junctions (6 fantasy beings)
# Treasures 12..17 sit on movable T-junctions, 18..23 on movable corners.
MOVABLE_TILE_SPEC: list[tuple[int, int]] = (
    [(STRAIGHT, NO_TREASURE)] * 12
    + [(CORNER, NO_TREASURE)] * 10
    + [(CORNER, 18 + i) for i in range(6)]
    + [(TJUNCTION, 12 + i) for i in range(6)]
)
assert len(MOVABLE_TILE_SPEC) == 34

TREASURE_NAMES = [
    # fixed tiles, in the order of FIXED_TREASURE_CELLS:
    # (0,2) (0,4) (2,0) (2,2) (2,4) (2,6) (4,0) (4,2) (4,4) (4,6) (6,2) (6,4)
    "book", "gold purse", "map", "crown", "keys", "skull",
    "ring", "chest", "emerald", "sword", "candlestick", "helmet",
    # movable T-junctions
    "genie", "bat", "dragon", "ghost", "sorceress", "dwarf",
    # movable corners
    "owl", "rat", "moth", "spider", "lizard", "beetle",
]
assert len(TREASURE_NAMES) == NUM_TREASURES

# --------------------------------------------------------------------------
# Insertion slots. 12 slots, three per side, at odd rows / columns.
#   0-2  top    (columns 1,3,5) push down
#   3-5  right  (rows 1,3,5)    push left
#   6-8  bottom (columns 1,3,5) push up
#   9-11 left   (rows 1,3,5)    push right
# The slot opposite to s is (s + 6) % 12; inserting there would undo s.
# --------------------------------------------------------------------------
NUM_SLOTS = 12
ODD_LINES = (1, 3, 5)


def slot_info(slot: int) -> tuple[str, int]:
    side = ("top", "right", "bottom", "left")[slot // 3]
    return side, ODD_LINES[slot % 3]


def opposite_slot(slot: int) -> int:
    return (slot + 6) % NUM_SLOTS
