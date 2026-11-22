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
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_SHIFT, NUM_MOVE = 48, 49
MAX_PLAYERS = 4


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
    def __init__(self, in_planes: int, vec_dim: int, channels: int = 64, blocks: int = 4, hidden: int = 256):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.vec_in = nn.Linear(vec_dim, channels)  # broadcast-added to every cell (FiLM-lite)
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.move_head = nn.Conv2d(channels, 1, 1)
        self.trunk = nn.Sequential(nn.Linear(channels * 49 + vec_dim, hidden), nn.ReLU())
        self.shift_head = nn.Linear(hidden, NUM_SHIFT)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, MAX_PLAYERS))

    def forward(self, planes, vector, mask):
        """Returns masked logits (B,97) and the acting player's value (B,)."""
        logits, values = self.forward_all(planes, vector, mask)
        return logits, values[:, 0]

    def forward_all(self, planes, vector, mask):
        """Returns masked logits (B,97) and ego-centric values (B,4)."""
        x = self.stem(planes) + self.vec_in(vector)[:, :, None, None]
        x = self.tower(x)
        move_logits = self.move_head(x).flatten(1)                     # (B,49)
        h = self.trunk(torch.cat([x.flatten(1), vector], dim=1))
        logits = torch.cat([self.shift_head(h), move_logits], dim=1)   # (B,97)
        logits = logits.masked_fill(~mask, -1e9)
        return logits, self.value_head(h)


def build_net(args: dict) -> LabyrinthNet:
    from labyrinth.env import LabyrinthEnv
    from labyrinth.observation import ObservationConfig

    probe = LabyrinthEnv(obs_config=ObservationConfig(
        observability=args.get("observability", "partial"), view_radius=args.get("view_radius")))
    return LabyrinthNet(probe.planes_shape[0], probe.vector_shape[0],
                        args.get("channels", 64), args.get("blocks", 4))


def load_checkpoint(path, device="cpu"):
    """Load a checkpoint saved by any trainer in this repo.

    Older checkpoints with a single-output value head are upgraded: the old
    output becomes the ego value, the opponents' outputs start untrained.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    net = build_net(ck["args"]).to(device)
    state = ck["model"]
    own = net.state_dict()
    for k, v in state.items():
        if k in own and own[k].shape != v.shape and k.startswith("value_head"):
            fixed = own[k].clone()
            fixed[: v.shape[0]] = v
            state[k] = fixed
    net.load_state_dict(state)
    net.eval()
    return net, ck["args"]
