"""Offline renderers: ANSI text and RGB arrays (PIL), e.g. for GIFs/videos."""
from __future__ import annotations

import numpy as np

from .core import PHASE_SHIFT, GameState
from .tiles import BOARD_SIZE, E, HOME, N, ODD_LINES, S, TREASURE_NAMES, W

PLAYER_COLORS = [(230, 180, 20), (220, 50, 47), (38, 110, 210), (40, 160, 70)]  # seat order = START_CORNERS
PLAYER_NAMES = ["yellow", "red", "blue", "green"]


def render_ansi(state: GameState) -> str:
    """3x3 characters per tile: '#' walls, ' ' corridors, digits = pawns."""
    rows = []
    for r in range(BOARD_SIZE):
        lines = ["", "", ""]
        for c in range(BOARD_SIZE):
            m = int(state.board[r, c])
            centre = "*" if state.treasure[r, c] >= 0 else " "
            for p in range(state.num_players):
                if tuple(state.pos[p]) == (r, c):
                    centre = str(p)
            lines[0] += "#" + (" " if m & N else "#") + "#"
            lines[1] += (" " if m & W else "#") + centre + (" " if m & E else "#")
            lines[2] += "#" + (" " if m & S else "#") + "#"
        rows.extend(lines)
    p = state.current
    head = (
        f"turn {state.turn} | player {p} ({PLAYER_NAMES[p]}) | "
        f"{'SHIFT' if state.phase == PHASE_SHIFT else 'MOVE'} | found {state.found.tolist()}"
    )
    return head + "\n" + "\n".join(rows)


def render_rgb(state: GameState, tile_px: int = 64, reveal: bool = True) -> np.ndarray:
    from PIL import Image, ImageDraw

    margin = tile_px // 2
    board_px = BOARD_SIZE * tile_px
    panel = 220
    img = Image.new("RGB", (board_px + 2 * margin + panel, board_px + 2 * margin), (32, 30, 38))
    d = ImageDraw.Draw(img)

    def draw_tile(x0, y0, mask, treasure, collected, size):
        wall, floor = (88, 70, 52), (236, 224, 196)
        d.rectangle([x0, y0, x0 + size - 1, y0 + size - 1], fill=wall)
        w = size // 3
        cx0, cy0 = x0 + w, y0 + w
        d.rectangle([cx0, cy0, cx0 + w - 1, cy0 + w - 1], fill=floor)
        if mask & N: d.rectangle([cx0, y0, cx0 + w - 1, cy0], fill=floor)
        if mask & S: d.rectangle([cx0, cy0 + w - 1, cx0 + w - 1, y0 + size - 1], fill=floor)
        if mask & W: d.rectangle([x0, cy0, cx0, cy0 + w - 1], fill=floor)
        if mask & E: d.rectangle([cx0 + w - 1, cy0, x0 + size - 1, cy0 + w - 1], fill=floor)
        if treasure >= 0:
            col = (150, 150, 150) if collected else (200, 120, 230)
            d.ellipse([cx0 + 3, cy0 + 3, cx0 + w - 4, cy0 + w - 4], outline=col, width=2)
            d.text((x0 + 2, y0 + 1), str(treasure), fill=(255, 255, 255))

    collected = state.collected_treasures()
    targets = {state.target(p): p for p in range(state.num_players)} if reveal else {}
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            x0, y0 = margin + c * tile_px, margin + r * tile_px
            t = int(state.treasure[r, c])
            draw_tile(x0, y0, int(state.board[r, c]), t, t in collected, tile_px)
            if t in targets and t != HOME:
                d.rectangle([x0, y0, x0 + tile_px - 1, y0 + tile_px - 1],
                            outline=PLAYER_COLORS[targets[t]], width=3)
    # homes
    for p, (r, c) in enumerate(state.pos_home):
        x0, y0 = margin + c * tile_px, margin + r * tile_px
        d.rectangle([x0 + 2, y0 + 2, x0 + tile_px - 3, y0 + tile_px - 3],
                    outline=PLAYER_COLORS[p], width=2)
    # pawns (offset when sharing a tile)
    offs = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    rad = tile_px // 7
    for p in range(state.num_players):
        r, c = state.pos[p]
        cx = margin + c * tile_px + tile_px // 2 + offs[p][0] * tile_px // 6
        cy = margin + r * tile_px + tile_px // 2 + offs[p][1] * tile_px // 6
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=PLAYER_COLORS[p], outline=(0, 0, 0))
    # insertion arrows
    forb = (state.last_slot + 6) % 12 if state.last_slot >= 0 else -1
    for slot in range(12):
        side, line = slot // 3, ODD_LINES[slot % 3]
        col = (90, 90, 90) if slot == forb else (240, 240, 240)
        mid = margin + line * tile_px + tile_px // 2
        pts = {0: (mid, margin // 2), 1: (board_px + margin + margin // 2, mid),
               2: (mid, board_px + margin + margin // 2), 3: (margin // 2, mid)}[side]
        d.ellipse([pts[0] - 4, pts[1] - 4, pts[0] + 4, pts[1] + 4], fill=col)
    # side panel
    px = board_px + 2 * margin + 10
    d.text((px, margin), f"turn {state.turn}", fill=(255, 255, 255))
    d.text((px, margin + 14), f"{'SHIFT' if state.phase == PHASE_SHIFT else 'MOVE'} - P{state.current}",
           fill=PLAYER_COLORS[state.current])
    d.text((px, margin + 34), "spare:", fill=(255, 255, 255))
    draw_tile(px, margin + 50, state.spare, state.spare_treasure,
              state.spare_treasure in collected, tile_px)
    y = margin + 60 + tile_px
    for p in range(state.num_players):
        t = state.target(p)
        tname = "HOME" if t == HOME else f"{t} {TREASURE_NAMES[t]}"
        txt = f"P{p} {state.found[p]}/{len(state.decks[p])}" + (f"  -> {tname}" if reveal else "")
        d.text((px, y), txt, fill=PLAYER_COLORS[p])
        y += 16
    if state.winner >= 0:
        d.text((px, y + 10), f"WINNER: P{state.winner}", fill=PLAYER_COLORS[state.winner])
    return np.asarray(img)
