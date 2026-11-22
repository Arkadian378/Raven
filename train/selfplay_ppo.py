"""Self-play PPO with a small AlphaStar-style league.

* The learner shares its parameters across every seat it occupies.
* Other seats are filled from a league: frozen snapshots of past learners
  sampled with Prioritised Fictitious Self-Play (PFSP, weight (1-p_win)^2),
  plus scripted exploiters (greedy, random) as fixed members.
* Only the learner's decisions become training data. Rewards produced by
  other players' moves are credited to the learner's last pending decision,
  so each seat sees a proper single-agent trajectory of its own decisions.
* Periodic evaluation vs. greedy baselines; everything is streamed to the
  live viewer (``--live``) so games can be watched while training.

Run:  python -m train.selfplay_ppo --live --updates 200
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from labyrinth.agents import GreedyAgent, RandomAgent
from labyrinth.env import LabyrinthEnv, RewardConfig
from labyrinth.observation import ObservationConfig
from train.model import LabyrinthNet, load_checkpoint


# ---------------------------------------------------------------- policies
class NetPolicy:
    """Wraps a (frozen) network so it can play a seat."""

    def __init__(self, net, name, device):
        self.net, self.name, self.device = net, name, device

    @torch.no_grad()
    def act_batch(self, obs_list, greedy=False):
        planes, vec, mask = stack(obs_list, self.device)
        logits, value = self.net(planes, vec, mask)
        dist = torch.distributions.Categorical(logits=logits)
        a = logits.argmax(-1) if greedy else dist.sample()
        return a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), value.cpu().numpy()


class ScriptedPolicy:
    def __init__(self, agent):
        self.agent, self.name = agent, agent.name


def stack(obs_list, device):
    planes = torch.as_tensor(np.stack([o["planes"] for o in obs_list]), device=device)
    vec = torch.as_tensor(np.stack([o["vector"] for o in obs_list]), device=device)
    mask = torch.as_tensor(np.stack([o["action_mask"] for o in obs_list]), device=device)
    return planes, vec, mask


# ------------------------------------------------------------------ league
class League:
    def __init__(self, max_snapshots=20, rng=None):
        self.members: list[dict] = []  # {"policy", "games", "learner_wins"}
        self.max_snapshots = max_snapshots
        self.rng = rng or np.random.default_rng()

    def add(self, policy):
        self.members.append({"policy": policy, "games": 0, "score": 0.0})
        snaps = [m for m in self.members if isinstance(m["policy"], NetPolicy)]
        if len(snaps) > self.max_snapshots:
            self.members.remove(snaps[0])

    def sample(self):
        """PFSP: favour opponents the learner does not beat yet."""
        p_win = np.array([(m["score"] + 1) / (m["games"] + 2) for m in self.members])
        w = (1 - p_win) ** 2 + 1e-3
        return self.members[self.rng.choice(len(self.members), p=w / w.sum())]

    def record(self, member, learner_score):
        member["games"] += 1
        member["score"] += learner_score


# --------------------------------------------------------------- rollouts
@dataclass
class Pending:
    obs: dict
    action: int
    logp: float
    value: float
    reward: float = 0.0


class SelfPlayRunner:
    def __init__(self, args, net, league, device, viewer=None):
        self.args, self.net, self.league, self.device = args, net, league, device
        self.rng = np.random.default_rng(args.seed)
        obs_cfg = ObservationConfig(observability=args.observability,
                                    view_radius=args.view_radius)
        rew_cfg = RewardConfig(treasure=args.treasure_reward, distance_shaping=args.shaping)
        self.envs = [LabyrinthEnv(num_players=args.players, obs_config=obs_cfg, reward_config=rew_cfg,
                                  max_turns=args.max_turns, viewer=viewer if i == 0 else None,
                                  seed=args.seed + i) for i in range(args.num_envs)]
        self.learner = NetPolicy(net, "learner", device)
        self.obs = [None] * len(self.envs)
        self.seats = [None] * len(self.envs)
        self.pending = [dict() for _ in self.envs]
        self.trajs = {}  # (env, seat, episode-id) -> list of finished transitions
        self.ep_ids = [0] * len(self.envs)
        self.stats = {"episodes": 0, "learner_wins": 0, "learner_seats": 0, "turns": []}
        for i in range(len(self.envs)):
            self._reset(i)

    def _reset(self, i):
        env = self.envs[i]
        self.obs[i], _ = env.reset()
        self.ep_ids[i] += 1
        n = env.num_players
        if self.rng.random() < self.args.self_play_prob or not self.league.members:
            seats = [("learner", None)] * n  # pure self-play
        else:
            k = int(self.rng.integers(n))
            seats = []
            for p in range(n):
                if p == k:
                    seats.append(("learner", None))
                else:
                    seats.append(("league", self.league.sample()))
        self.seats[i] = seats
        self.pending[i] = {}

    def collect(self, n_steps):
        """Step all envs until n_steps learner transitions are finished."""
        out = []
        while len(out) < n_steps:
            # group envs by who must act
            learner_idx, others = [], []
            for i, env in enumerate(self.envs):
                p = env.current_player
                (learner_idx if self.seats[i][p][0] == "learner" else others).append(i)
            actions = {}
            if learner_idx:
                a, lp, v = self.learner.act_batch([self.obs[i] for i in learner_idx])
                for j, i in enumerate(learner_idx):
                    p = self.envs[i].current_player
                    prev = self.pending[i].pop(p, None)
                    if prev is not None:
                        out.append((prev, False, (i, p, self.ep_ids[i])))
                    self.pending[i][p] = Pending(self.obs[i], int(a[j]), float(lp[j]), float(v[j]))
                    actions[i] = int(a[j])
            for i in others:
                p = self.envs[i].current_player
                pol = self.seats[i][p][1]["policy"]
                if isinstance(pol, ScriptedPolicy):
                    actions[i] = pol.agent.act(self.obs[i], self.envs[i].state, p)
                else:
                    actions[i] = int(pol.act_batch([self.obs[i]])[0][0])
            for i, env in enumerate(self.envs):
                self.obs[i], r, term, trunc, info = env.step(actions[i])
                for p, pend in self.pending[i].items():
                    pend.reward += float(r[p])
                if term or trunc:
                    for p, pend in self.pending[i].items():
                        out.append((pend, True, (i, p, self.ep_ids[i])))
                    self._episode_stats(i, env, r, trunc)
                    self._reset(i)
        # value of the next (still pending) decision, to bootstrap cut trajectories
        boot = {(i, p, self.ep_ids[i]): pend.value
                for i in range(len(self.envs)) for p, pend in self.pending[i].items()}
        return out, boot

    def _episode_stats(self, i, env, final_r, trunc):
        seats = self.seats[i]
        s = env.state
        self.stats["episodes"] += 1
        self.stats["turns"].append(s.turn)
        learner_seats = [p for p in range(len(seats)) if seats[p][0] == "learner"]
        if len(learner_seats) < len(seats):  # learner vs league: update PFSP stats
            p = learner_seats[0]
            score = 1.0 if s.winner == p else 0.0
            if trunc:
                score = float(s.found[p] == s.found.max())
            for q, (kind, member) in enumerate(seats):
                if kind == "league":
                    self.league.record(member, score)


# ------------------------------------------------------------------- PPO
def compute_gae(transitions, boot, gamma, lam):
    """Group finished transitions by trajectory and compute GAE in order."""
    by_traj = {}
    for k, (pend, done, key) in enumerate(transitions):
        by_traj.setdefault(key, []).append((k, pend, done))
    adv = np.zeros(len(transitions), np.float32)
    ret = np.zeros(len(transitions), np.float32)
    for key, items in by_traj.items():
        gae = 0.0
        for t in reversed(range(len(items))):
            k, pend, done = items[t]
            # trajectory may continue in the next batch: bootstrap with V(s_{t+1})
            next_v = 0.0 if done else (items[t + 1][1].value if t + 1 < len(items) else boot.get(key, 0.0))
            nonterminal = 0.0 if done else 1.0
            delta = pend.reward + gamma * next_v * nonterminal - pend.value
            gae = delta + gamma * lam * nonterminal * gae
            adv[k] = gae
            ret[k] = gae + pend.value
    return adv, ret


def ppo_update(net, opt, transitions, boot, args, device):
    adv, ret = compute_gae(transitions, boot, args.gamma, args.lam)
    obs = [t[0].obs for t in transitions]
    planes, vec, mask = stack(obs, device)
    acts = torch.as_tensor([t[0].action for t in transitions], device=device)
    old_lp = torch.as_tensor([t[0].logp for t in transitions], device=device)
    adv_t = torch.as_tensor(adv, device=device)
    ret_t = torch.as_tensor(ret, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    n = len(transitions)
    logs = []
    net.train()
    for _ in range(args.epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, args.minibatch):
            idx = perm[s:s + args.minibatch]
            logits, value = net(planes[idx], vec[idx], mask[idx])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(acts[idx])
            ratio = (lp - old_lp[idx]).exp()
            pg = -torch.min(ratio * adv_t[idx], ratio.clamp(1 - args.clip, 1 + args.clip) * adv_t[idx]).mean()
            vloss = F.mse_loss(value, ret_t[idx])
            ent = dist.entropy().mean()
            loss = pg + args.vf_coef * vloss - args.ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            logs.append((pg.item(), vloss.item(), ent.item()))
    net.eval()
    return np.mean(logs, axis=0)


# -------------------------------------------------------------- evaluation
def evaluate(net, args, device, n_games=8, opponent="greedy", sample=False):
    """Learner in one (rotating) seat vs. scripted opponents."""
    pol = NetPolicy(net, "learner", device)
    env = LabyrinthEnv(num_players=args.players, max_turns=args.max_turns,
                       obs_config=ObservationConfig(observability=args.observability, view_radius=args.view_radius))
    wins, cards = 0.0, 0.0
    for g in range(n_games):
        obs, _ = env.reset(seed=10_000 + g)
        seat = g % env.num_players
        opp = {p: (GreedyAgent(seed=g * 7 + p) if opponent == "greedy" else RandomAgent(seed=p))
               for p in range(env.num_players)}
        while True:
            p = env.current_player
            a = pol.act_batch([obs], greedy=not sample)[0][0] if p == seat else opp[p].act(obs, env.state, p)
            obs, r, term, trunc, _ = env.step(int(a))
            if term or trunc:
                wins += 1.0 if env.state.winner == seat else 0.0
                cards += env.state.found[seat] / len(env.state.decks[seat])
                break
    return wins / n_games, cards / n_games


# -------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--observability", default="partial", choices=["partial", "full"])
    ap.add_argument("--view-radius", type=int, default=None)
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--steps-per-update", type=int, default=2048)
    ap.add_argument("--updates", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--treasure-reward", type=float, default=0.1)
    ap.add_argument("--shaping", type=float, default=0.05)
    ap.add_argument("--self-play-prob", type=float, default=0.5)
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=8)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from a checkpoint (e.g. behaviour cloning)")
    ap.add_argument("--resume", default=None, help="continue a run from its latest.pt (league is rebuilt)")
    ap.add_argument("--live", action="store_true", help="start the live web viewer")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="runs/selfplay")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    viewer = None
    if args.live:
        from labyrinth.viewer.server import LiveViewer
        viewer = LiveViewer(host=args.host, port=args.port).start()
        viewer.set_meta(run=vars(args))

    probe = LabyrinthEnv(num_players=args.players,
                         obs_config=ObservationConfig(observability=args.observability, view_radius=args.view_radius))
    start, total = 1, 0
    if args.init or args.resume:
        net, ck_args = load_checkpoint(args.init or args.resume, device)
        args.channels, args.blocks = ck_args.get("channels", 64), ck_args.get("blocks", 4)
    else:
        net = LabyrinthNet(probe.planes_shape[0], probe.vector_shape[0], args.channels, args.blocks).to(device)
    net.eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start, total = ck.get("update", 0) + 1, ck.get("steps", 0)
    league = League(rng=np.random.default_rng(args.seed + 1))
    league.add(ScriptedPolicy(GreedyAgent(seed=args.seed, epsilon=0.1)))
    league.add(ScriptedPolicy(RandomAgent(seed=args.seed)))
    runner = SelfPlayRunner(args, net, league, device, viewer)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for u in range(start, args.updates + 1):
        batch, boot = runner.collect(args.steps_per_update)
        total += len(batch)
        pg, vl, ent = ppo_update(net, opt, batch, boot, args, device)
        turns = runner.stats["turns"][-50:]
        metrics = {"policy_loss": pg, "value_loss": vl, "entropy": ent,
                   "mean_step_reward": float(np.mean([t[0].reward for t in batch])),
                   "game_length_turns": float(np.mean(turns)) if turns else 0.0}
        if u % args.snapshot_every == 0:
            snap = copy.deepcopy(net).eval()
            league.add(NetPolicy(snap, f"snapshot_{u}", device))
            torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "args": vars(args),
                        "update": u, "steps": total}, out / "latest.pt")
        if u % args.eval_every == 0:
            for opp in ("greedy", "random"):
                wr, cf = evaluate(net, args, device, args.eval_games, opp)
                metrics[f"win_rate_vs_{opp}"] = wr
                metrics[f"cards_frac_vs_{opp}"] = cf
        if viewer:
            viewer.log_metrics(total, **metrics)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"update": u, "steps": total, "time": time.time() - t0, **metrics}) + "\n")
        print(f"[{u:4d}] steps={total} eps={runner.stats['episodes']} "
              + " ".join(f"{k}={v:.3f}" for k, v in metrics.items())
              + f" | {time.time() - t0:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "args": vars(args)}, out / "final.pt")


if __name__ == "__main__":
    main()
