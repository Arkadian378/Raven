"""Stage 3: AlphaZero-style expert iteration with information-set MCTS.

Each iteration:
  1. workers play self-play games where every MCTS seat searches with the
     current network (plus, optionally, some greedy seats to keep the agent
     honest against a fixed style of play);
  2. each MCTS decision yields a sample (observation, visit distribution,
     final outcome of every seat, the opponents' hidden targets);
  3. the network is trained on a replay buffer of recent samples (policy,
     value, and the belief head that models the opponents' targets; once
     trained, the search uses it to sample determinisations);
  4. periodically the search agent and the raw network are evaluated against
     three greedy opponents.

    python -m train.alphazero --init runs/bc/final.pt --live
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
from labyrinth.env import LabyrinthEnv
from labyrinth.observation import ObservationConfig
from train.data import SampleBuffer, ego, opponent_targets, outcome_vector, train_steps
from train.model import load_checkpoint, obs_config_from_args

_W = {"version": None, "net": None, "args": None}


def _worker_init():
    # workers ignore Ctrl+C: the main process catches it and terminates the pool
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)


def selfplay_game(job):
    from train.mcts import MCTSAgent
    seed, ckpt, version, a = job
    if _W["version"] != version:
        _W["net"], _W["args"] = load_checkpoint(ckpt)
        _W["version"] = version
    rng = np.random.default_rng(seed)
    cfg = obs_config_from_args(_W["args"])
    env = LabyrinthEnv(obs_config=cfg, max_turns=a["max_turns"])
    obs, _ = env.reset(seed=seed)
    n = env.num_players
    kinds = ["greedy" if rng.random() < a["greedy_prob"] else "mcts" for _ in range(n)]
    if "mcts" not in kinds:
        kinds[int(rng.integers(n))] = "mcts"
    agents = [MCTSAgent(_W["net"], sims=a["sims"], obs_config=cfg, max_turns=a["max_turns"], seed=seed + p,
                        c_puct=a["c_puct"], belief_eps=a["belief_eps"])
              if k == "mcts" else GreedyAgent(seed + p) for p, k in enumerate(kinds)]
    records, actions, decisions = [], [], 0
    while True:
        p = env.current_player
        if kinds[p] == "mcts":
            ag = agents[p]
            ag.temperature = 1.0 if decisions < a["temp_decisions"] else 0.0
            act = ag.act(obs, env.state, p, add_noise=True)
            records.append((obs, ag.last_pi, p, opponent_targets(env.state, p)))
            decisions += 1
        else:
            act = agents[p].act(obs, env.state, p)
        actions.append(int(act))
        obs, r, term, trunc, _ = env.step(act)
        if term or trunc:
            break
    z = outcome_vector(env.state)
    samples = [(o, pi, ego(z, p, n), opp) for o, pi, p, opp in records]
    s = env.state
    summary = {"seed": seed, "actions": actions, "kinds": kinds, "winner": int(s.winner),
               "winner_kind": kinds[s.winner] if s.winner >= 0 else "none", "turns": int(s.turn)}
    return samples, summary


def replay_into_viewer(viewer, summary, episode):
    env = LabyrinthEnv(viewer=viewer)
    env.episode = episode - 1
    env.reset(seed=summary["seed"])
    for a in summary["actions"]:
        env.step(a)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, help="starting checkpoint (behaviour cloning or PPO)")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--games-per-iter", type=int, default=32)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--history", type=int, default=2, help="opponents' last moves given to the network (trail planes)")
    ap.add_argument("--belief-coef", type=float, default=0.5, help="weight of the opponent-model loss")
    ap.add_argument("--belief-eps", type=float, default=0.1, help="uniform mixing of the belief used by the search")
    ap.add_argument("--greedy-prob", type=float, default=0.25, help="probability that a seat is a greedy opponent")
    ap.add_argument("--temp-decisions", type=int, default=8, help="decisions sampled with temperature 1")
    ap.add_argument("--max-turns", type=int, default=200)
    ap.add_argument("--buffer", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reuse", type=float, default=4.0, help="times each new sample is seen in training")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count()))
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=24)
    ap.add_argument("--eval-sims", type=int, default=100)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="runs/alphazero")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    net, ck_args = load_checkpoint(args.init, device, overrides={"history": args.history})
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    probe = LabyrinthEnv(obs_config=obs_config_from_args(ck_args))
    buf = SampleBuffer(args.buffer, probe.planes_shape, probe.vector_shape[0])
    ckpt = out / "latest.pt"
    def save(path, it):  # write then rename, so an interruption never leaves a broken file
        tmp = Path(str(path) + ".tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "args": ck_args, "iteration": it}, tmp)
        tmp.replace(path)
    save(ckpt, 0)

    viewer = None
    if args.live:
        from labyrinth.viewer.server import LiveViewer
        viewer = LiveViewer(host=args.host, port=args.port).start()

    from train import evaluate as ev
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=_worker_init)
    rng = np.random.default_rng(args.seed)
    t0, best = time.time(), -1.0
    a = {k: getattr(args, k) for k in ("sims", "greedy_prob", "temp_decisions", "max_turns", "c_puct", "belief_eps")}
    try:
        for it in range(1, args.iterations + 1):
            jobs = [(int(rng.integers(2**31)), str(ckpt), it, a) for _ in range(args.games_per_iter)]
            summaries, new = [], 0
            for samples, summ in pool.imap_unordered(selfplay_game, jobs):
                buf.extend(samples)
                new += len(samples)
                summaries.append(summ)
            steps = max(1, int(new * args.reuse / args.batch))
            net.to(device)
            m = train_steps(net, opt, buf, steps, args.batch, device, belief_coef=args.belief_coef, rng=rng)
            net.cpu()
            if it >= 3:  # from now on the search samples opponents' targets from the belief head
                ck_args["belief_trained"] = True
            save(ckpt, it)
            mixed = [s for s in summaries if "greedy" in s["kinds"]]
            row = {"iteration": it, "time": time.time() - t0, "samples": new, "buffer": buf.size, "train_steps": steps,
                   "policy_loss": m["policy_loss"], "value_loss": m["value_loss"], "net_best_eq_search_best": m["top1_exact"],
                   "kl_search_vs_net": m["kl_target_net"],
                   "belief_loss": m["belief_loss"], "belief_top1": m["belief_top1"],
                   "belief_used": bool(ck_args.get("belief_trained")),
                   "game_length_turns": float(np.mean([s["turns"] for s in summaries])),
                   "selfplay_mcts_win_share_vs_greedy_seats": float(np.mean(
                       [s["winner_kind"] == "mcts" for s in mixed])) if mixed else float("nan")}
            if it % args.eval_every == 0:
                for spec, key in ((f"mcts:{ckpt}:{args.eval_sims}", "mcts"), (f"net:{ckpt}", "net")):
                    r = ev.run(spec, "greedy", args.eval_games, args.workers, max_turns=400)
                    row[f"{key}_win_rate_vs_greedy"] = r["win_rate"]
                    row[f"{key}_cards_frac_vs_greedy"] = r["cards_frac"]
                if row["mcts_win_rate_vs_greedy"] >= best:
                    best = row["mcts_win_rate_vs_greedy"]
                    save(out / "best.pt", it)
            print(json.dumps(row), flush=True)
            with open(out / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            if viewer:
                viewer.log_metrics(it, **{k: v for k, v in row.items() if isinstance(v, (int, float)) and v == v})
                replay_into_viewer(viewer, summaries[0], it)
    except KeyboardInterrupt:
        print("\n[alphazero] interrupted: checkpoints are in", out, flush=True)
        pool.terminate()
    else:
        pool.close()
    pool.join()


if __name__ == "__main__":
    main()
