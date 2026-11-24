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
* **Opponent model.** If the network has a trained belief head, each
  determinisation samples the opponents' current targets from the network's
  belief (mixed with a little uniform noise) instead of uniformly, so the
  search plans against what the opponents are most likely trying to reach.
* **Tree reuse.** After an insertion is chosen, the subtree below it (which
  already holds the batched evaluation of every move and the visits spent
  on them) becomes the root of the following move decision.
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
                 expand_shifts: bool = True, belief: bool = True, belief_eps: float = 0.1):
        self.net, self.cfg, self.sims = net, obs_config, sims
        self.c_puct, self.alpha, self.noise_frac = c_puct, dirichlet_alpha, noise_frac
        self.max_turns, self.device = max_turns, device
        self.rng = np.random.default_rng(seed)
        self.determinize = determinize and obs_config.observability != "full"
        self.expand_shifts = expand_shifts
        self.belief = belief and getattr(net, "belief_trained", False)
        self.belief_eps = belief_eps

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

    @torch.no_grad()
    def opponent_beliefs(self, s: GameState, player: int) -> dict[int, np.ndarray]:
        """Network belief over each opponent's current target, absolute seats."""
        o = encode(s, player, self.cfg, self.max_turns)
        t = lambda k, dt: torch.as_tensor(o[k][None], dtype=dt, device=self.device)
        _, _, b = self.net.forward_full(t("planes", torch.float32), t("vector", torch.float32),
                                        t("action_mask", torch.bool))
        b = torch.softmax(b[0], -1).cpu().numpy().astype(np.float64)
        b = (1 - self.belief_eps) * b + self.belief_eps / b.shape[1]
        n = s.num_players
        return {(player + i) % n: b[i - 1] for i in range(1, n)}

    def _expand(self, node: Node, s: GameState, mask, prior, noise: bool = False):
        node.player = s.current
        legal = np.flatnonzero(mask)
        pr = prior[legal]
        pr = pr / pr.sum() if pr.sum() > 0 else np.full(len(legal), 1 / len(legal))
        if noise:
            pr = (1 - self.noise_frac) * pr + self.noise_frac * self.rng.dirichlet([self.alpha] * len(legal))
        node.children = {int(a): Node(float(p)) for a, p in zip(legal, pr)}

    # -------------------------------------------------------------- search
    def search(self, state: GameState, add_noise: bool = False, root: Node | None = None) -> np.ndarray:
        """Run the search from ``state`` (as seen by the player to act) and
        return the visit distribution over the 97 actions. ``root`` may be a
        subtree kept from the previous decision of the same player."""
        observer = state.current
        if root is None or root.children is None or root.player != observer:
            root = Node()
        elif add_noise:
            legal = list(root.children)
            noise = self.rng.dirichlet([self.alpha] * len(legal))
            for a, eta in zip(legal, noise):
                c = root.children[a]
                c.prior = (1 - self.noise_frac) * c.prior + self.noise_frac * float(eta)
        self.root = root
        base = state.clone()
        base.history = []
        beliefs = self.opponent_beliefs(base, observer) if self.determinize and self.belief else None
        self.last_beliefs = beliefs
        for sim in range(self.sims):
            s = base.determinize(observer, self.rng, beliefs) if self.determinize else base.clone()
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
        if root.children is None:  # zero simulations: expand once
            mask, prior, _ = self.evaluate(base)
            self._expand(root, base, mask, prior)
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

    def __init__(self, net, sims: int = 200, temperature: float = 0.0, reuse_tree: bool = True, **kw):
        self.mcts = MCTS(net, sims=sims, **kw)
        self.temperature = temperature
        self.reuse_tree = reuse_tree
        self.name = f"mcts{sims}"
        self.last_pi = None
        self._kept = None  # (state key, subtree) kept from the insertion just chosen

    @staticmethod
    def _key(s: GameState):
        return (s.turn, s.current, s.phase, s.board.tobytes(), s.pos.tobytes(), s.spare, s.last_slot)

    def act(self, obs, state: GameState, player: int | None = None, add_noise: bool = False) -> int:
        legal = np.flatnonzero(obs["action_mask"])
        kept, self._kept = self._kept, None
        root = kept[1] if kept is not None and kept[0] == self._key(state) else None
        if len(legal) == 1:
            self.last_pi = np.eye(NUM_ACTIONS, dtype=np.float32)[legal[0]]
            return int(legal[0])
        pi = self.mcts.search(state, add_noise=add_noise, root=root)
        self.last_pi = pi
        if self.temperature > 0:
            p = pi ** (1 / self.temperature)
            a = int(self.mcts.rng.choice(NUM_ACTIONS, p=p / p.sum()))
        else:
            top = np.flatnonzero(pi == pi.max())
            a = int(self.mcts.rng.choice(top))
        if self.reuse_tree and state.phase == 0:  # keep the subtree for this player's move decision
            child = self.mcts.root.children.get(a)
            if child is not None and child.children is not None:
                nxt = state.clone()
                nxt.step(a)
                self._kept = (self._key(nxt), child)
        return a


class NetAgent:
    """Raw policy network (no search): argmax or sampling."""

    def __init__(self, net, sample: bool = False, seed=None, obs_config: ObservationConfig | None = None,
                 max_turns: int = 400):
        self.net, self.sample, self.name = net, sample, "net"
        self.rng = np.random.default_rng(seed)
        self.cfg, self.max_turns = obs_config, max_turns

    @torch.no_grad()
    def act(self, obs, state=None, player=None) -> int:
        if self.cfg is not None and state is not None:  # encode with the network's own observation settings
            obs = encode(state, state.current if player is None else player, self.cfg, self.max_turns)
        t = lambda k, dt: torch.as_tensor(obs[k][None], dtype=dt)
        logits, _ = self.net(t("planes", torch.float32), t("vector", torch.float32), t("action_mask", torch.bool))
        if self.sample:
            p = torch.softmax(logits[0], 0).numpy()
            return int(self.rng.choice(NUM_ACTIONS, p=p / p.sum()))
        return int(logits.argmax())
