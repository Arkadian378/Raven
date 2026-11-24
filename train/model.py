"""Policy/value network for Labyrinth (AlphaZero-style residual tower).

Inputs are the ego-centric planes (C x 7 x 7) and the vector features. The
policy has two heads matching the factorised action space:

* shift head : MLP on the global embedding  -> 48 logits (slot x rotation)
* move head  : 1x1 conv on the spatial map  -> 49 logits (one per cell)

Both are concatenated into the flat 97-way action space and masked with the
legality mask, so a single categorical distribution covers both phases.

The value head predicts the return of *every* seat, in ego-centric order
(index 0 = the player to act). PPO only uses index 0; multi-player MCTS
(max^n backups) needs the whole vector.

The belief head is an opponent model: for each opponent (ego order 1..3) it
predicts which treasure that opponent is currently looking for. It is spatial,
like the move head: a 1x1 conv scores every cell for every opponent, and the
score of a treasure is the score of the cell it lies on (read from the
treasure-identity planes), or a separate score if it is on the spare tile.
Treasures that cannot be the answer (already collected, or the observer's own
target) are masked. It is trained as an auxiliary task and used by the search
to sample determinisations.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SHIFT, NUM_MOVE = 48, 49
MAX_PLAYERS = 4
NUM_TREASURES = 24


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(x + self.b2(self.c2(y)))


class LabyrinthNet(nn.Module):
    def __init__(self, in_planes: int, vec_dim: int, channels: int = 64, blocks: int = 4, hidden: int = 256,
                 belief_index: dict | None = None):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.vec_in = nn.Linear(vec_dim, channels)  # broadcast-added to every cell (FiLM-lite)
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.move_head = nn.Conv2d(channels, 1, 1)
        self.trunk = nn.Sequential(nn.Linear(channels * 49 + vec_dim, hidden), nn.ReLU())
        self.shift_head = nn.Linear(hidden, NUM_SHIFT)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, MAX_PLAYERS))
        # auxiliary: final fraction of cards found by every seat (ego order), in [0, 1]
        self.score_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, MAX_PLAYERS))
        self.belief_index = belief_index  # positions of treasure features in the input (see build_net)
        self.belief_head = nn.ModuleDict({
            "cell": nn.Conv2d(channels, MAX_PLAYERS - 1, 1),         # where is each opponent heading
            "spare": nn.Linear(hidden, MAX_PLAYERS - 1),             # ... or is it the spare tile's treasure
            "prior": nn.Linear(hidden, (MAX_PLAYERS - 1) * NUM_TREASURES),
        })
        self.belief_trained = False  # set by load_checkpoint; the search ignores an untrained head

    def forward(self, planes, vector, mask):
        """Returns masked logits (B,97) and the acting player's value (B,)."""
        logits, values = self.forward_all(planes, vector, mask)
        return logits, values[:, 0]

    def forward_all(self, planes, vector, mask):
        """Returns masked logits (B,97) and ego-centric values (B,4)."""
        logits, values, _ = self.forward_full(planes, vector, mask)
        return logits, values

    def forward_full(self, planes, vector, mask, with_score: bool = False):
        """Also returns the opponent-target belief logits (B,3,24) and, if
        asked, the predicted final card fractions (B,4)."""
        x = self.stem(planes) + self.vec_in(vector)[:, :, None, None]
        x = self.tower(x)
        move_logits = self.move_head(x).flatten(1)                     # (B,49)
        h = self.trunk(torch.cat([x.flatten(1), vector], dim=1))
        logits = torch.cat([self.shift_head(h), move_logits], dim=1)   # (B,97)
        logits = logits.masked_fill(~mask, -1e9)
        out = (logits, self.value_head(h), self._belief(x, h, planes, vector))
        if with_score:
            out = out + (torch.sigmoid(self.score_head(h)),)
        return out

    def _belief(self, x, h, planes, vector):
        bh = self.belief_head
        out = bh["prior"](h).view(-1, MAX_PLAYERS - 1, NUM_TREASURES)
        bi = self.belief_index
        if bi is None:
            return out
        cell = bh["cell"](x).flatten(2)                                              # (B,3,49)
        where = planes[:, bi["treasure"]: bi["treasure"] + NUM_TREASURES].flatten(2)  # (B,24,49) one-hot
        spare = vector[:, bi["spare"]: bi["spare"] + NUM_TREASURES]                 # (B,24)
        out = out + torch.einsum("bic,btc->bit", cell, where) + bh["spare"](h)[:, :, None] * spare[:, None, :]
        collected = (where * planes[:, bi["collected"]].flatten(1)[:, None, :]).sum(2) > 0
        collected = collected | ((spare * vector[:, bi["spare_collected"], None]) > 0)
        impossible = collected | (vector[:, bi["own"]: bi["own"] + NUM_TREASURES] > 0)
        return out.masked_fill(impossible[:, None, :], -30.0)


def obs_config_from_args(args: dict):
    from labyrinth.observation import ObservationConfig
    return ObservationConfig(observability=args.get("observability", "partial"),
                             view_radius=args.get("view_radius"), history=args.get("history", 0))


def build_net(args: dict) -> LabyrinthNet:
    from labyrinth.env import LabyrinthEnv

    from labyrinth.observation import plane_names, vector_names

    cfg = obs_config_from_args(args)
    probe = LabyrinthEnv(obs_config=cfg)
    pn, vn = plane_names(cfg), vector_names(cfg)
    index = None
    if cfg.treasure_id_planes:
        index = {"treasure": pn.index("treasure_0"), "collected": pn.index("treasure_collected"),
                 "spare": vn.index("spare_treasure_0"), "spare_collected": vn.index("spare_collected"),
                 "own": vn.index("own_target_0")}
    return LabyrinthNet(probe.planes_shape[0], probe.vector_shape[0],
                        args.get("channels", 64), args.get("blocks", 4), belief_index=index)


def load_checkpoint(path, device="cpu", overrides: dict | None = None):
    """Load a checkpoint saved by any trainer in this repo.

    Older checkpoints are upgraded so they keep playing exactly as before:
    * a single-output value head becomes the ego value (opponents untrained);
    * a missing belief head starts untrained (``net.belief_trained = False``);
    * ``overrides={"history": k}`` widens the input with the trail planes,
      whose stem weights start at zero (they are appended last).
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    args = {**ck["args"], **(overrides or {})}
    net = build_net(args).to(device)
    state = ck["model"]
    own = net.state_dict()
    for k, v in list(state.items()):
        if k in own and own[k].shape != v.shape:
            fixed = torch.zeros_like(own[k]) if k == "stem.0.weight" else own[k].clone()
            if k.startswith("value_head"):
                fixed[: v.shape[0]] = v
            elif k == "stem.0.weight" and v.shape[1] < fixed.shape[1]:
                fixed[:, : v.shape[1]] = v
            else:
                raise ValueError(f"{path}: incompatible parameter {k} {tuple(v.shape)} vs {tuple(own[k].shape)}")
            state[k] = fixed
    missing, unexpected = net.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith(("belief_head", "score_head")) for k in missing):
        raise ValueError(f"{path}: missing {missing}, unexpected {unexpected}")
    net.belief_trained = bool(args.get("belief_trained", False)) and not any(k.startswith("belief_head") for k in missing)
    net.eval()
    return net, args
