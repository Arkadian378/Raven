# Piano di esperimenti: dall'imitazione ad AlphaZero

Tutti i comandi vanno lanciati dalla cartella `labyrinth-rl`. Ogni esperimento scrive nella propria cartella `runs/...`; non riusare mai la stessa cartella.

I tempi sono stime per **4 core CPU**: con più core scalano circa in proporzione, con una GPU il training è molto più veloce. Ogni comando di training ha `--live`, se vuoi guardare il training su http://127.0.0.1:8765.

---

## Passo 0: controllo (2 minuti)
```bash
pytest -q                                   # 31 test passati
python scripts/env_stats.py --games 200 > runs/env_stats.json
```
**Cosa aspettarsi**: il greedy vince ~100% contro 3 random e chi gioca per primo vince ~40% tra greedy. Il secondo numero è il vantaggio del primo giocatore, da citare nelle conclusioni.

---

## Passo 1: imitazione del greedy (~40 minuti)
```bash
python -m train.behavior_cloning --games 6000 --epochs 8 --out runs/bc
```
**Output**: `runs/bc/final.pt`, `runs/bc/metrics.jsonl`, `runs/bc/dataset.npz`.

**Cosa aspettarsi** (dalla run sul tabellone vecchio):
- `top1_in_expert_set` sale a circa 65–70%;
- contro 3 random vince ~100%;
- contro 3 greedy la rete da sola vince quasi mai.

---

## Passo 2: PPO in self-play partendo dalla rete del passo 1 (~1 ora)
Serve a capire se il solo RL, senza ricerca, migliora l'imitazione.
```bash
python -m train.selfplay_ppo --init runs/bc/final.pt --updates 400 --lr 1e-4 --ent-coef 0.003 \
    --eval-every 10 --eval-games 32 --out runs/ppo_from_bc
```
Il learning rate e il coefficiente di entropia sono bassi apposta, per non "dimenticare" subito l'imitazione.

**Da guardare**: `win_rate_vs_greedy` e `cards_frac_vs_greedy` in `metrics.jsonl`.

**Ipotesi**: miglioramento piccolo o nullo, come nella run da zero che si era fermata su un plateau.

---

## Passo 3: AlphaZero partendo dalla rete del passo 1 (6–12 ore, la run principale)
```bash
python -m train.alphazero --init runs/bc/final.pt --iterations 60 --games-per-iter 32 --sims 32 \
    --eval-every 5 --eval-games 32 --eval-sims 64 --out runs/az
```
**Output**: `runs/az/latest.pt`, `runs/az/best.pt` (il migliore per win rate contro il greedy) e `runs/az/metrics.jsonl`.

**Da guardare**:
- `mcts_win_rate_vs_greedy`: rete + ricerca;
- `net_win_rate_vs_greedy`: la rete da sola;
- `selfplay_mcts_win_share_vs_greedy_seats`: nelle partite di self-play con posti greedy, quanto spesso vince l'agente.

**Ipotesi**: si parte da circa 25% (alla pari con il greedy) e l'obiettivo è salire chiaramente sopra, cioè 40% o più. Anche la rete da sola dovrebbe migliorare, perché impara dalla ricerca.

Se si interrompe, puoi ripartire con `--init runs/az/latest.pt` e una nuova `--out`.

---

## Passo 3b: AlphaZero completo (la run principale per il modello finale)
Rispetto al passo 3 ci sono queste aggiunte, tutte attive di default:
- **memoria**: la rete vede dove ogni avversario ha iniziato le sue ultime 2 mosse (`--history 2`). Le scie si azzerano quando l'avversario raccoglie un tesoro;
- **modello degli avversari**: una testa della rete ("belief") predice quale tesoro sta cercando ogni avversario, e la ricerca la usa per immaginare le carte nascoste invece di sceglierle a caso;
- **ricerca più profonda**: 128 simulazioni per le decisioni "piene";
- **simulazioni a budget variabile** (KataGo): solo il 25% delle decisioni usa la ricerca piena e diventa esempio per la policy (`--full-prob 0.25`). Le altre usano 32 simulazioni (`--fast-sims 32`) e allenano solo valore, punteggio e belief. Le partite sono circa 3 volte più veloci;
- **riuso dell'albero**: dopo l'inserimento, la ricerca per il movimento riparte dal sottoalbero già esplorato;
- **obiettivo ausiliario sul punteggio**: la rete predice anche la frazione di carte che ogni giocatore raccoglierà (`--score-coef 1.0`). È un segnale più ricco di vinto/perso.

Prima rifai il passo 1 con il codice nuovo (il `dataset.npz` vecchio non è compatibile con le scie), poi:
```bash
python -m train.behavior_cloning --games 20000 --epochs 8 --out runs/bc2
python -m train.alphazero --init runs/bc2/final.pt --iterations 40 --games-per-iter 64 --sims 128 --fast-sims 32 \
    --full-prob 0.25 --reuse 8 --eval-every 5 --eval-games 48 --eval-sims 128 --out runs/az2
```
Un checkpoint BC vecchio funziona comunque: le parti nuove della rete vengono aggiunte da sole.

**Da guardare**, oltre alle metriche del passo 3:
- `belief_top1`: quante volte la rete indovina l'obiettivo di un avversario. A caso si indovina circa l'8%, una regola semplice ("si è avvicinato a quel tesoro") arriva al 19%;
- `net_best_eq_search_best`: quante volte la mossa preferita dalla rete è anche la più visitata dalla ricerca; `kl_search_vs_net`: quanto la ricerca si allontana dalla rete. Se il primo sale e il secondo scende nel tempo, la rete sta assorbendo ciò che la ricerca scopre. (Il vecchio `top1_match_search` contava solo se la mossa della rete riceveva almeno una visita, quindi era sempre vicino al 99% e non diceva nulla.)
- `value_loss` e `score_loss`: devono scendere. Un predittore costante ha `value_loss` di circa 0,33.

**Ablazione** (quanto conta il modello degli avversari), a parità di rete:
```bash
python -m train.evaluate --agent mcts:runs/az2/best.pt:128 --opponent greedy --games 100
python -m train.evaluate --agent mcts:runs/az2/best.pt:128:nobelief --opponent greedy --games 100
```

---

## Passo 4: valutazione finale, la tabella delle conclusioni (~2–3 ore)
Ogni riga gioca 100 partite: l'agente occupa un posto a rotazione contro 3 avversari. Contro 3 greedy, il 25% significa "forte quanto il greedy".
```bash
mkdir -p runs/eval
for A in greedy net:runs/bc/final.pt mcts:runs/bc/final.pt:64 net:runs/ppo_from_bc/final.pt \
         net:runs/az/best.pt mcts:runs/az/best.pt:64; do
  python -m train.evaluate --agent $A --opponent greedy --games 100 >> runs/eval/vs_greedy.txt
done
```
**Scaling della ricerca**: quanto conta pensare di più.
```bash
for S in 16 64 200; do
  python -m train.evaluate --agent mcts:runs/az/best.pt:$S --opponent greedy --games 100 >> runs/eval/sims.txt
done
```
**Facoltativo**: AlphaZero contro l'agente del passo 1, cioè quanto ha guadagnato il self-play. È lento perché tutti e 4 i posti cercano.
```bash
python -m train.evaluate --agent mcts:runs/az/best.pt:64 --opponent mcts:runs/bc/final.pt:64 --games 48 >> runs/eval/az_vs_bc.txt
```

---

## Passo 5: partite contro gli umani
```bash
python -m labyrinth.play.server --agent mcts:runs/az/best.pt:200 --host 0.0.0.0
```
Le partite finiscono in `games/`. Si rivedono con `python examples/replay_game.py games/<id>.json`.

---

## Conclusioni da trarre
1. **Imitazione contro RL puro**: confronto tra la PPO da zero (la run da 2000 update) e la rete del passo 1.
2. **Valore della ricerca**: `net:` contro `mcts:` a parità di rete, nel passo 4.
3. **Valore del self-play AlphaZero**: `mcts:bc` contro `mcts:az`, nel passo 4 e nel confronto facoltativo.
4. **Scaling**: win rate al variare delle simulazioni (16 / 64 / 200).
5. **Contro gli umani**: win rate dell'agente sulle partite registrate.

Quando hai i file `metrics.jsonl` e `runs/eval/*.txt`, mandameli e ti preparo grafici e tabelle.
