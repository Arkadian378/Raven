"""Compact storage of training samples (observation, policy target, value target).

Planes are binary, so they are bit-packed; a 45x7x7 observation takes 276 bytes.
"""
from __future__ import annotations

import numpy as np
import torch

from labyrinth.observation import MAX_PLAYERS


class SampleBuffer:
    """Ring buffer of (planes, vector, mask, pi, z) samples."""

    def __init__(self, capacity: int, planes_shape, vec_dim: int, num_actions: int = 97):
        self.capacity, self.planes_shape = capacity, tuple(planes_shape)
        n_bits = int(np.prod(planes_shape))
        self.planes = np.zeros((capacity, (n_bits + 7) // 8), np.uint8)
        self.vector = np.zeros((capacity, vec_dim), np.float16)
        self.mask = np.zeros((capacity, (num_actions + 7) // 8), np.uint8)
        self.pi = np.zeros((capacity, num_actions), np.float16)
        self.z = np.zeros((capacity, MAX_PLAYERS), np.float32)
        self.num_actions = num_actions
        self.size = self.ptr = 0

    def add(self, obs: dict, pi: np.ndarray, z: np.ndarray) -> None:
        i = self.ptr
        self.planes[i] = np.packbits(obs["planes"].astype(bool).ravel())
        self.vector[i] = obs["vector"]
        self.mask[i] = np.packbits(obs["action_mask"].astype(bool))
        self.pi[i] = pi
        self.z[i] = z
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, samples) -> None:
        for obs, pi, z in samples:
            self.add(obs, pi, z)

    def batch(self, idx, device="cpu"):
        n_bits = int(np.prod(self.planes_shape))
        planes = np.unpackbits(self.planes[idx], axis=1)[:, :n_bits].reshape(len(idx), *self.planes_shape)
        mask = np.unpackbits(self.mask[idx], axis=1)[:, :self.num_actions].astype(bool)
        t = lambda x, dt: torch.as_tensor(x, device=device, dtype=dt)
        return (t(planes, torch.float32), t(self.vector[idx], torch.float32), t(mask, torch.bool),
                t(self.pi[idx], torch.float32), t(self.z[idx], torch.float32))

    def save(self, path) -> None:
        n = self.size
        np.savez_compressed(path, planes=self.planes[:n], vector=self.vector[:n], mask=self.mask[:n],
                            pi=self.pi[:n], z=self.z[:n], planes_shape=np.array(self.planes_shape))

    @classmethod
    def load(cls, path) -> "SampleBuffer":
        d = np.load(path)
        n = len(d["z"])
        buf = cls(n, tuple(d["planes_shape"]), d["vector"].shape[1], d["pi"].shape[1])
        buf.planes[:], buf.vector[:], buf.mask[:], buf.pi[:], buf.z[:] = d["planes"], d["vector"], d["mask"], d["pi"], d["z"]
        buf.size, buf.ptr = n, 0
        return buf


def outcome_vector(state, max_players: int = MAX_PLAYERS) -> np.ndarray:
    """Zero-sum final score of every seat: +1 winner, -1/(N-1) others.
    Truncated games: the leaders by cards found share the win."""
    n = state.num_players
    if state.winner >= 0:
        w = np.eye(n)[state.winner]
    else:
        prog = state.found / np.array([max(len(d), 1) for d in state.decks])
        w = (prog == prog.max()).astype(float)
        w /= w.sum()
    z = np.zeros(max_players, np.float32)
    z[:n] = w - (1 - w) / (n - 1)
    return z


def ego(z_abs: np.ndarray, player: int, n: int) -> np.ndarray:
    """Absolute seat values -> ego-centric order (player first)."""
    out = np.zeros_like(z_abs)
    for i in range(n):
        out[i] = z_abs[(player + i) % n]
    return out


def train_steps(net, opt, buf: SampleBuffer, steps: int, batch: int, device, value_coef: float = 1.0, rng=None):
    """Policy cross-entropy (soft targets) + value MSE over all seats."""
    import torch.nn.functional as F
    rng = rng or np.random.default_rng()
    net.train()
    logs = []
    for _ in range(steps):
        idx = rng.integers(0, buf.size, size=min(batch, buf.size))
        planes, vec, mask, pi, z = buf.batch(idx, device)
        logits, values = net.forward_all(planes, vec, mask)
        logp = F.log_softmax(logits, dim=1)
        ploss = -(pi * logp.clamp(min=-1e4)).sum(1).mean()
        vloss = F.mse_loss(values, z)
        loss = ploss + value_coef * vloss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        acc = (pi.gather(1, logits.argmax(1, keepdim=True)) > 0).float().mean()
        logs.append((ploss.item(), vloss.item(), acc.item()))
    net.eval()
    return np.mean(logs, axis=0)
