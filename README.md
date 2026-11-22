# LabyrinthRL

Ambiente di Reinforcement Learning multi-agente per **Labirinto (Das verrückte Labyrinth, Ravensburger)**, pensato per una ricerca ispirata ad AlphaGo / AlphaZero / AlphaStar: 2–4 giocatori, self-play, informazione parziale (POMDP), visualizzazione live durante il training.

## Struttura

```
labyrinth/
  tiles.py         geometria delle tessere, layout fisso, slot di inserimento
  core.py          motore di regole (GameState): spinta, movimento, raccolta, vittoria, determinizzazione
  observation.py   codifica osservazioni ego-centriche (full / partial / fog of war)
  env.py           ambiente turn-based multi-agente + configurazione ricompense
  wrappers.py      PettingZoo AEC e Gymnasium (singolo agente vs avversari fissi)
  agents.py        baseline: RandomAgent, GreedyAgent (lookahead a 1 mossa)
  render.py        render ANSI e RGB (PIL) per GIF/video
  viewer/          visualizzatore web live (solo libreria standard)
train/
  model.py         rete policy/value residua (stile AlphaZero) con testa shift + testa move
  selfplay_ppo.py  PPO in self-play con league PFSP (stile AlphaStar)
scripts/env_stats.py  statistiche dell'ambiente usate nel paper
tests/             test unitari (regole, POMDP, API PettingZoo/Gymnasium)
examples/          watch_live.py, record_gif.py
paper/             paper LaTeX che descrive l'ambiente
```

## Installazione

```bash
pip install -e ".[rl,dev]"
pytest -q
```

## Uso rapido

```python
from labyrinth import LabyrinthEnv, ObservationConfig
from labyrinth.agents import GreedyAgent

env = LabyrinthEnv(num_players=4, obs_config=ObservationConfig(observability="partial"))
obs, info = env.reset(seed=0)
agents = [GreedyAgent(p) for p in range(4)]
while True:
    p = env.current_player
    obs, rewards, terminated, truncated, info = env.step(agents[p].act(obs, env.state, p))
    if terminated or truncated:
        break
```

* **Azioni**: `Discrete(97)` con maschera di legalità. 0–47 = spinta (`slot*4 + rotazione`), 48–96 = movimento (`48 + riga*7 + colonna`). Ogni turno = 2 decisioni (SHIFT poi MOVE).
* **Osservazioni**: `{"planes": (45,7,7), "vector": (90,), "action_mask": (97,)}`, ego-centriche (il giocatore che osserva è sempre lo slot 0).
* **Osservabilità** (`ObservationConfig`):
  * `observability="partial"` (regole ufficiali): si vede solo la propria carta tesoro corrente.
  * `observability="full"`: anche gli obiettivi avversari (informazione privilegiata, per teacher/oracle).
  * `view_radius=k`: nebbia di guerra attorno alla propria pedina (richiede memoria → policy ricorrenti).
* **Ricompense** (`RewardConfig`): terminale a somma zero (+1 vincitore, −1/(N−1) agli altri), bonus tesoro opzionale, shaping potential-based opzionale, troncamento con classifica per progresso.
* **Ricerca ad albero**: `env.state.clone()` e `state.determinize(player, rng)` campionano uno stato completo coerente con l'information set del giocatore (per IS-MCTS / PIMC stile AlphaZero).

## Visualizzazione live

```bash
python examples/watch_live.py                # partite fra baseline, nessun training
python -m train.selfplay_ppo --live          # training self-play + viewer
# poi apri http://127.0.0.1:8765  (da remoto: ssh -L 8765:localhost:8765 server)
```

Il viewer mostra la partita del primo ambiente di training (selettore "vista" onnisciente oppure dal punto di vista POMDP di un singolo giocatore) e le curve delle metriche (loss, entropia, win rate vs greedy/random).

Per una GIF: `python examples/record_gif.py game.gif`.

## Training self-play

```bash
python -m train.selfplay_ppo --updates 500 --num-envs 16 --live
python -m train.selfplay_ppo --observability full        # teacher con informazione completa
python -m train.selfplay_ppo --view-radius 2             # POMDP più duro (nebbia)
```

## Pipeline per un agente forte (stile AlphaGo)

```bash
# 1) imitazione del greedy (come la policy supervisionata di AlphaGo)
python -m train.behavior_cloning --games 6000 --out runs/bc
# 2) (opzionale) PPO in self-play partendo dalla rete BC
python -m train.selfplay_ppo --init runs/bc/final.pt --out runs/ppo_bc --live
# 3) AlphaZero: self-play con IS-MCTS + training sulla distribuzione delle visite
python -m train.alphazero --init runs/bc/final.pt --out runs/az --live
# valutazione testa a testa (posto a rotazione, 3 avversari)
python -m train.evaluate --agent mcts:runs/az/best.pt:200 --opponent greedy --games 100
```

Specifiche agente: `greedy`, `random`, `net:PATH` (rete senza ricerca), `mcts:PATH:SIMS` (rete + ricerca).

## Giocare contro l'agente (e registrare le partite)

```bash
python -m labyrinth.play.server --agent mcts:runs/az/best.pt:200
# apri http://127.0.0.1:8000  (--host 0.0.0.0 per giocare da altri dispositivi in rete locale)
```

Si sceglie per ogni posto: umano, IA, greedy o random. Gli umani giocano a turno sullo stesso schermo e vedono la propria carta solo premendo "Mostra la mia carta". Ogni partita è salvata dopo ogni mossa in `games/<id>.json` (seed, sequenza di azioni, tempi di riflessione, vincitore) e si può rivedere con `python examples/replay_game.py games/<id>.json`. 
Modelli https://huggingface.co/Gab378/Raven.

## Paper

`paper/main.tex` (compilare con `latexmk -pdf main.tex`).
