"""Stage 1 (as AlphaGo's supervised policy): imitate the greedy expert.

Generates games where every seat is a greedy agent (some with a little noise
for state diversity), records each decision from the acting player's
partial observation, and trains the policy on the greedy's set of tied-best
actions (soft target) and the value head on the final outcome of every seat.

    python -m train.behavior_cloning --games 6000 --out runs/bc
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import signal
import time
from pathlib import Path

import numpy as np
import torch

from labyrinth.agents import GreedyAgent
from labyrinth.core import NUM_ACTIONS
from labyrinth.env import LabyrinthEnv
from labyrinth.observation import ObservationConfig
from train.data import SampleBuffer, ego, opponent_targets, outcome_vector, score_vector, train_steps
from train.model import build_net


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process


def play_expert_game(job):
    seed, cfg, noise = job
    rng = np.random.default_rng(seed)
    env = LabyrinthEnv(obs_config=ObservationConfig(**cfg), max_turns=200)
    obs, _ = env.reset(seed=seed)
    experts = [GreedyAgent(seed * 4 + p) for p in range(4)]
    eps = [float(rng.choice(noise)) for _ in range(4)]
    records = []
    while True:
        p = env.current_player
        best = experts[p].best_actions(env.state, p)
        pi = np.zeros(NUM_ACTIONS, np.float32)
        pi[best] = 1.0 / len(best)
        if rng.random() < eps[p]:  # exploratory move: keep state, no policy label
            a = int(rng.choice(np.flatnonzero(obs["action_mask"])))
            records.append((obs, None, p, None))
        else:
            a = int(rng.choice(best))
            records.append((obs, pi, p, opponent_targets(env.state, p)))
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    z = outcome_vector(env.state)
    n = env.num_players
    sc = score_vector(env.state)
    return [(o, pi, ego(z, p, n), opp, ego(sc, p, n, empty=-1.0))
            for o, pi, p, opp in records if pi is not None], env.state.turn


def evaluate_policy(net, cfg, games=24, opponent="greedy", seed=50_000):
    """Raw network (argmax) in a rotating seat vs. 3 scripted opponents."""
    from labyrinth.agents import RandomAgent
    env = LabyrinthEnv(obs_config=ObservationConfig(**cfg), max_turns=400)
    wins = cards = 0.0
    for g in range(games):
        obs, _ = env.reset(seed=seed + g)
        seat = g % 4
        opp = [GreedyAgent(seed + g * 4 + p) if opponent == "greedy" else RandomAgent(g * 4 + p) for p in range(4)]
        while True:
            p = env.current_player
            if p == seat:
                with torch.no_grad():
                    t = lambda k, dt: torch.as_tensor(obs[k][None], dtype=dt)
                    logits, _ = net(t("planes", torch.float32), t("vector", torch.float32), t("action_mask", torch.bool))
                a = int(logits.argmax())
            else:
                a = opp[p].act(obs, env.state, p)
            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                wins += env.state.winner == seat
                cards += env.state.found[seat] / len(env.state.decks[seat])
                break
    return wins / games, cards / games


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--observability", default="partial", choices=["partial", "full"])
    ap.add_argument("--history", type=int, default=2, help="opponents' last moves given to the network")
    ap.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.0, 0.05, 0.15],
                    help="per-seat probability of a random move, sampled per game")
    ap.add_argument("--data", default=None, help="reuse a saved dataset (.npz)")
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--out", default="runs/bc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {"observability": args.observability, "history": args.history}
    probe = LabyrinthEnv(obs_config=ObservationConfig(**cfg))
    t0 = time.time()

    if args.data:
        buf = SampleBuffer.load(args.data)
        if buf.planes_shape != tuple(probe.planes_shape):
            raise SystemExit(f"{args.data} was generated with other observation settings "
                             f"(planes {buf.planes_shape}, now {tuple(probe.planes_shape)}): "
                             "regenerate it or pass the matching --history")
    else:
        jobs = [(args.seed * 1_000_000 + g, cfg, args.noise) for g in range(args.games)]
        buf = SampleBuffer(args.games * 80, probe.planes_shape, probe.vector_shape[0])
        lengths = []
        with mp.Pool(args.workers, initializer=_ignore_sigint) as pool:
            for k, (samples, turns) in enumerate(pool.imap_unordered(play_expert_game, jobs, chunksize=8)):
                buf.extend(samples)
                lengths.append(turns)
                if (k + 1) % 500 == 0:
                    print(f"generated {k + 1}/{args.games} games, {buf.size} samples, {time.time() - t0:.0f}s", flush=True)
        buf.save(out / "dataset.npz")
        print(f"dataset: {buf.size} samples, mean game length {np.mean(lengths):.1f} turns")

    device = torch.device(args.device)
    net = build_net({**vars(args), "view_radius": None}).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, buf.size // args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * args.epochs)

    class _Opt:  # step optimiser and scheduler together
        def zero_grad(self): opt.zero_grad()
        def step(self): opt.step(); sched.step()

    rng = np.random.default_rng(args.seed)
    ck_args = {**vars(args), "observability": args.observability, "view_radius": None, "history": args.history,
               "belief_trained": bool((buf.opp[:buf.size] >= 0).any())}
    for ep in range(1, args.epochs + 1):
        m = train_steps(net, _Opt(), buf, steps_per_epoch, args.batch, device, rng=rng)
        wr, cf = evaluate_policy(net.cpu(), cfg, args.eval_games)
        net.to(device)
        row = {"epoch": ep, "policy_loss": m["policy_loss"], "value_loss": m["value_loss"], "score_loss": m["score_loss"],
               "top1_in_expert_set": m["top1"], "belief_loss": m["belief_loss"], "belief_top1": m["belief_top1"],
               "win_rate_vs_greedy": wr, "cards_frac_vs_greedy": cf, "time": time.time() - t0}
        print(json.dumps(row), flush=True)
        with open(out / "metrics.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
        torch.save({"model": net.state_dict(), "args": ck_args}, out / "final.pt")
    wr, cf = evaluate_policy(net.cpu(), cfg, args.eval_games, opponent="random")
    print(f"vs random: win {wr:.2f} cards {cf:.2f}")


if __name__ == "__main__":
    main()
