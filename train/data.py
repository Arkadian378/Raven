"""Compact storage of training samples (observation, policy target, value target).

Planes are binary, so they are bit-packed; a 45x7x7 observation takes 276 bytes.
"""
from __future__ import annotations

import numpy as np
import torch

from labyrinth.observation import MAX_PLAYERS


class SampleBuffer:
    """Ring buffer of (planes, vector, mask, pi, z, opp) samples.

    ``opp`` holds, for each opponent in ego order, the treasure it was looking
    for at that moment (-1 = none / unknown); it trains the belief head.
    ``score`` is the final fraction of cards found by every seat (ego order,
    -1 = unknown); it trains the auxiliary score head.
    A sample whose ``pi`` is all zeros only trains the value-type heads
    (positions played with a fast search, see playout cap randomisation)."""

    def __init__(self, capacity: int, planes_shape, vec_dim: int, num_actions: int = 97):
        self.capacity, self.planes_shape = capacity, tuple(planes_shape)
        n_bits = int(np.prod(planes_shape))
        self.planes = np.zeros((capacity, (n_bits + 7) // 8), np.uint8)
        self.vector = np.zeros((capacity, vec_dim), np.float16)
        self.mask = np.zeros((capacity, (num_actions + 7) // 8), np.uint8)
        self.pi = np.zeros((capacity, num_actions), np.float16)
        self.z = np.zeros((capacity, MAX_PLAYERS), np.float32)
        self.opp = np.full((capacity, MAX_PLAYERS - 1), -1, np.int8)
        self.score = np.full((capacity, MAX_PLAYERS), -1, np.float16)
        self.num_actions = num_actions
        self.size = self.ptr = 0

    def add(self, obs: dict, pi, z: np.ndarray, opp=None, score=None) -> None:
        i = self.ptr
        self.planes[i] = np.packbits(obs["planes"].astype(bool).ravel())
        self.vector[i] = obs["vector"]
        self.mask[i] = np.packbits(obs["action_mask"].astype(bool))
        self.pi[i] = 0 if pi is None else pi
        self.z[i] = z
        self.opp[i] = -1 if opp is None else opp
        self.score[i] = -1 if score is None else score
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def extend(self, samples) -> None:
        for sample in samples:
            self.add(*sample)

    def batch(self, idx, device="cpu"):
        n_bits = int(np.prod(self.planes_shape))
        planes = np.unpackbits(self.planes[idx], axis=1)[:, :n_bits].reshape(len(idx), *self.planes_shape)
        mask = np.unpackbits(self.mask[idx], axis=1)[:, :self.num_actions].astype(bool)
        t = lambda x, dt: torch.as_tensor(x, device=device, dtype=dt)
        return (t(planes, torch.float32), t(self.vector[idx], torch.float32), t(mask, torch.bool),
                t(self.pi[idx], torch.float32), t(self.z[idx], torch.float32), t(self.opp[idx], torch.long),
                t(self.score[idx], torch.float32))

    def save(self, path) -> None:
        n = self.size
        np.savez_compressed(path, planes=self.planes[:n], vector=self.vector[:n], mask=self.mask[:n],
                            pi=self.pi[:n], z=self.z[:n], opp=self.opp[:n], score=self.score[:n], planes_shape=np.array(self.planes_shape))

    @classmethod
    def load(cls, path) -> "SampleBuffer":
        d = np.load(path)
        n = len(d["z"])
        buf = cls(n, tuple(d["planes_shape"]), d["vector"].shape[1], d["pi"].shape[1])
        buf.planes[:], buf.vector[:], buf.mask[:], buf.pi[:], buf.z[:] = d["planes"], d["vector"], d["mask"], d["pi"], d["z"]
        if "opp" in d:
            buf.opp[:] = d["opp"]
        if "score" in d:
            buf.score[:] = d["score"]
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


def opponent_targets(state, player: int, max_players: int = MAX_PLAYERS) -> np.ndarray:
    """Label for the belief head: each opponent's current target (ego order 1..),
    -1 if it has none (going home) or the seat is empty."""
    n = state.num_players
    out = np.full(max_players - 1, -1, np.int8)
    for i in range(1, n):
        t = state.target((player + i) % n)
        out[i - 1] = t if 0 <= t < 24 else -1
    return out


def score_vector(state, max_players: int = MAX_PLAYERS) -> np.ndarray:
    """Final fraction of its cards found by every seat (absolute order), -1 for empty seats."""
    out = np.full(max_players, -1, np.float32)
    for p in range(state.num_players):
        out[p] = state.found[p] / max(len(state.decks[p]), 1)
    return out


def ego(z_abs: np.ndarray, player: int, n: int, empty: float = 0.0) -> np.ndarray:
    """Absolute seat values -> ego-centric order (player first)."""
    out = np.full_like(z_abs, empty)
    for i in range(n):
        out[i] = z_abs[(player + i) % n]
    return out


def train_steps(net, opt, buf: SampleBuffer, steps: int, batch: int, device, value_coef: float = 1.0,
                belief_coef: float = 0.5, score_coef: float = 1.0, rng=None) -> dict:
    """Policy cross-entropy (soft targets) + value MSE over all seats
    + belief cross-entropy on the opponents' hidden targets
    + score MSE on the final fraction of cards of every seat."""
    import torch.nn.functional as F
    rng = rng or np.random.default_rng()
    net.train()
    logs = []
    for _ in range(steps):
        idx = rng.integers(0, buf.size, size=min(batch, buf.size))
        planes, vec, mask, pi, z, opp, score = buf.batch(idx, device)
        logits, values, belief, pred_score = net.forward_full(planes, vec, mask, with_score=True)
        logp = F.log_softmax(logits, dim=1)
        has_pi = pi.sum(1) > 0
        if not has_pi.any():
            has_pi[0] = True  # degenerate batch: keep the graph well defined
        ploss = -(pi[has_pi] * logp[has_pi].clamp(min=-1e4)).sum(1).mean()
        vloss = F.mse_loss(values, z)
        known = score >= 0
        sloss = F.mse_loss(pred_score[known], score[known]) if known.any() else torch.zeros((), device=planes.device)
        has = opp >= 0
        if has.any():
            bl = F.cross_entropy(belief[has], opp[has])
            bacc = (belief[has].argmax(1) == opp[has]).float().mean().item()
        else:
            bl, bacc = torch.zeros((), device=planes.device), float("nan")
        loss = ploss + value_coef * vloss + belief_coef * bl + score_coef * sloss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        pi, logits = pi[has_pi], logits[has_pi]
        top = pi.gather(1, logits.argmax(1, keepdim=True)).squeeze(1)
        acc = (top > 0).float().mean()                                  # net's choice is in the target's support
        exact = (top >= pi.max(1).values - 1e-6).float().mean()          # net's choice is the target's best action
        ent = -(pi * pi.clamp(min=1e-8).log()).sum(1).mean()
        logs.append((ploss.item(), vloss.item(), acc.item(), bl.item(), bacc, exact.item(), (ploss - ent).item(),
                     sloss.item()))
    net.eval()
    m = np.nanmean(np.array(logs, float), axis=0)
    return {"policy_loss": m[0], "value_loss": m[1], "top1": m[2], "belief_loss": m[3], "belief_top1": m[4],
            "top1_exact": m[5], "kl_target_net": m[6], "score_loss": m[7]}
