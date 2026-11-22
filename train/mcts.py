"""AlphaZero-style search for 2-4 player Labyrinth with hidden information.

* **Information sets.** Hidden information (the treasure decks) never affects
  which actions are legal -- legality depends only on the public board -- so we
  can grow a single tree over *action sequences* (single-observer IS-MCTS,
  Cowling et al. 2012). Every simulation first samples a determinisation of
  the true state consistent with the searching player's information
  (``GameState.determinize``) and replays the tree path on it.
* **Multi-player values.** The network predicts a value for every seat; nodes
  store a value *vector* and each node's player maximises its own component
  (max^n backups). Terminal states use the true zero-sum outcome.
* **Shift expansion.** When a node where a tile must be inserted is expanded,
  all its insertions are simulated exactly and evaluated in one batched
  forward pass (the move-phase observation contains the exact reachable set,
  so the value of "after this insertion" is easy to judge). Every insertion
  thus starts with a value estimate instead of a blind first-play urgency.
* **PUCT** selection with the network policy as prior, Dirichlet noise at the
  root for self-play exploration, and visit-count based action selection.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from labyrinth.core import NUM_ACTIONS, GameState
from labyrinth.observation import ObservationConfig, encode
from train.data import outcome_vector


class Node:
    __slots__ = ("player", "prior", "children", "N", "W", "N0")

    def __init__(self, prior: float = 0.0):
        self.player = -1           # seat to act at this node (set on expansion)
        self.prior = prior
        self.children: dict[int, Node] | None = None
        self.N = 0
        self.N0 = 0                # visits given at creation (batched expansion), not real search visits
        self.W: np.ndarray | None = None  # value sums, absolute seat order

    def q(self, player: int) -> float:
        return float(self.W[player] / self.N)


class MCTS:
    def __init__(self, net, obs_config: ObservationConfig = ObservationConfig(), sims: int = 200,
                 c_puct: float = 1.5, dirichlet_alpha: float = 0.3, noise_frac: float = 0.25,
                 max_turns: int = 400, device="cpu", seed: int | None = None, determinize: bool = True,
                 expand_shifts: bool = True):
        self.net, self.cfg, self.sims = net, obs_config, sims
        self.c_puct, self.alpha, self.noise_frac = c_puct, dirichlet_alpha, noise_frac
        self.max_turns, self.device = max_turns, device
        self.rng = np.random.default_rng(seed)
        self.determinize = determinize and obs_config.observability != "full"
        self.expand_shifts = expand_shifts

    # ------------------------------------------------------------ network
    def evaluate(self, s: GameState):
        """Prior over legal actions and value vector in absolute seat order."""
        return self.evaluate_batch([s])[0]

    @torch.no_grad()
    def evaluate_batch(self, states: list[GameState]):
        obs = [encode(s, s.current, self.cfg, self.max_turns) for s in states]
        t = lambda k, dt: torch.as_tensor(np.stack([o[k] for o in obs]), dtype=dt, device=self.device)
        logits, values = self.net.forward_all(t("planes", torch.float32), t("vector", torch.float32),
                                              t("action_mask", torch.bool))
        P = torch.softmax(logits, 1).cpu().numpy()
        V = values.cpu().numpy()
        out = []
        for s, o, p, v_ego in zip(states, obs, P, V):
            n = s.num_players
            v = np.zeros(n, np.float32)
            for i in range(n):
                v[(s.current + i) % n] = v_ego[i]
            out.append((o["action_mask"], p, v))
        return out

    def _expand(self, node: Node, s: GameState, mask, prior, noise: bool = False):
        node.player = s.current
        legal = np.flatnonzero(mask)
        pr = prior[legal]
        pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(legal), 1 / len(legal))
        if noise:
            pr = (1 - self.noise_frac) * pr + self.noise_frac * self.rng.dirichlet([self.alpha] * len(legal))
        node.children = {int(a): Node(float(p)) for a, p in zip(legal, pr)}

    # -------------------------------------------------------------- search
    def search(self, state: GameState, add_noise: bool = False) -> np.ndarray:
        """Run the search from ``state`` (as seen by the player to act) and
        return the visit distribution over the 97 actions."""
        observer = state.current
        root = Node()
        base = state.clone()
        base.history = []
        for sim in range(self.sims):
            s = base.determinize(observer, self.rng) if self.determinize else base.clone()
            node, path = root, [root]
            while node.children is not None and not s.done and s.turn < self.max_turns:
                a, node = self._select(node)
                s.step(a)
                path.append(node)
            if s.done or s.turn >= self.max_turns:
                value = outcome_vector(s)[: s.num_players]
            else:
                mask, prior, value = self.evaluate(s)
                self._expand(node, s, mask, prior, noise=node is root and add_noise)
                if self.expand_shifts and s.phase == 0:
                    value = self._expand_shift_children(node, s)
            for n in path:
                n.N += 1
                n.W = value.astype(np.float64) if n.W is None else n.W + value
        visits = np.zeros(NUM_ACTIONS, np.float32)
        for a, child in root.children.items():
            visits[a] = child.N - child.N0
        if visits.sum() <= 0:  # too few simulations: fall back to the initial estimates
            for a, child in root.children.items():
                visits[a] = child.N
        return visits / max(visits.sum(), 1)

    def _expand_shift_children(self, node: Node, s: GameState) -> np.ndarray:
        """Simulate every insertion, evaluate all of them in one batch, and
        return the value of the best one for the player to act."""
        acts = list(node.children)
        nxt = []
        for a in acts:
            s2 = s.clone()
            s2.step(a)
            nxt.append(s2)
        evals = self.evaluate_batch(nxt)
        best = None
        for a, s2, (mask, prior, v) in zip(acts, nxt, evals):
            child = node.children[a]
            self._expand(child, s2, mask, prior)
            child.N = child.N0 = 1
            child.W = v.astype(np.float64)
            if best is None or v[s.current] > best[s.current]:
                best = v
        return best

    def _select(self, node: Node):
        sqrt_n = math.sqrt(node.N)
        parent_q = node.q(node.player) if node.N else 0.0
        best, best_score = None, -1e18
        for a, child in node.children.items():
            # first-play urgency: unvisited children inherit a slightly pessimistic parent value
            q = child.q(node.player) if child.N else parent_q - 0.2
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.N)
            if score > best_score:
                best, best_score = (a, child), score
        return best


class MCTSAgent:
    """Agent interface (``act(obs, state, player)``) on top of :class:`MCTS`."""

    def __init__(self, net, sims: int = 200, temperature: float = 0.0, **kw):
        self.mcts = MCTS(net, sims=sims, **kw)
        self.temperature = temperature
        self.name = f"mcts{sims}"
        self.last_pi = None

    def act(self, obs, state: GameState, player: int | None = None, add_noise: bool = False) -> int:
        legal = np.flatnonzero(obs["action_mask"])
        if len(legal) == 1:
            self.last_pi = np.eye(NUM_ACTIONS, dtype=np.float32)[legal[0]]
            return int(legal[0])
        pi = self.mcts.search(state, add_noise=add_noise)
        self.last_pi = pi
        if self.temperature > 0:
            p = pi ** (1 / self.temperature)
            return int(self.mcts.rng.choice(NUM_ACTIONS, p=p / p.sum()))
        top = np.flatnonzero(pi == pi.max())
        return int(self.mcts.rng.choice(top))


class NetAgent:
    """Raw policy network (no search): argmax or sampling."""

    def __init__(self, net, sample: bool = False, seed=None):
        self.net, self.sample, self.name = net, sample, "net"
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def act(self, obs, state=None, player=None) -> int:
        t = lambda k, dt: torch.as_tensor(obs[k][None], dtype=dt)
        logits, _ = self.net(t("planes", torch.float32), t("vector", torch.float32), t("action_mask", torch.bool))
        if self.sample:
            p = torch.softmax(logits[0], 0).numpy()
            return int(self.rng.choice(NUM_ACTIONS, p=p / p.sum()))
        return int(logits.argmax())
