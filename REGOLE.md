# Regole del Labirinto implementate in LabyrinthRL

Questo documento descrive **esattamente** le regole del codice (`labyrinth/tiles.py` e `labyrinth/core.py`). Dove il codice si discosta dal gioco in scatola Ravensburger, è segnalato con **⚠ Differenza**.

---

## 1. Componenti

### 1.1 Tabellone e coordinate
- Griglia **7×7**. Una casella si indica con `(riga, colonna)`: `(0,0)` è in alto a sinistra e `(6,6)` in basso a destra. La riga cresce verso il basso, la colonna verso destra.
- Ogni tessera è un pezzo di labirinto con corridoi aperti su alcuni dei 4 lati: **N** (su), **E** (destra), **S** (giù), **O** (sinistra).
- Nel codice una tessera è una maschera di 4 bit: N=1, E=2, S=4, O=8.

### 1.2 Forme delle tessere
| Forma | Lati aperti | Rotazioni distinte |
|---|---|---|
| **I** (dritta) | 2 opposti (N-S oppure E-O) | 2 |
| **L** (curva) | 2 adiacenti | 4 |
| **T** (incrocio a T) | 3 | 4 |

### 1.3 Le 50 tessere
| | I | L | T | Totale |
|---|---|---|---|---|
| Fisse (incollate al tabellone) | 0 | 4 (angoli di partenza) | 12 (tutte con tesoro) | 16 |
| Mobili | 12 | 16 (6 con tesoro) | 6 (tutte con tesoro) | 34 |
| **Totale** | 12 | 20 | 18 | **50** |

- **24 tesori** in totale, ognuno presente su una sola tessera: 12 sulle T fisse, 6 sulle T mobili e 6 sulle L mobili.
- Delle 34 tessere mobili, 33 riempiono il tabellone e 1 resta fuori: è la **tessera libera** (spare).

### 1.4 Tessere fisse
Le 16 tessere fisse stanno nelle caselle con riga **e** colonna pari (0, 2, 4, 6).

| Casella | Forma, lati aperti | Tesoro (id) |
|---|---|---|
| (0,0) | L, E-S | partenza Giallo |
| (0,6) | L, S-O | partenza Rosso |
| (6,6) | L, N-O | partenza Blu |
| (6,0) | L, N-E | partenza Verde |
| (0,2) | T, E-S-O | 0 libro |
| (0,4) | T, E-S-O | 1 borsa di monete |
| (2,0) | T, N-E-S | 2 mappa |
| (2,2) | T, N-E-S | 3 corona |
| (2,4) | T, E-S-O | 4 chiavi |
| (2,6) | T, N-S-O | 5 teschio |
| (4,0) | T, N-E-S | 6 anello |
| (4,2) | T, N-E-O | 7 forziere |
| (4,4) | T, N-S-O | 8 smeraldo |
| (4,6) | T, N-S-O | 9 spada |
| (6,2) | T, N-E-O | 10 candelabro |
| (6,4) | T, N-E-O | 11 elmo |

- Le T sul bordo hanno il lato chiuso rivolto verso l'esterno del tabellone.
- La disposizione è copiata dal tabellone Ravensburger (edizione 2006). Il tabellone reale è simmetrico per rotazioni di 90°, quindi i 4 posti di partenza sono equivalenti.

**Tesori mobili** (come nella scatola):
- 12–17 sono sulle **T** mobili e sono esseri fantastici: genio, pipistrello, drago, fantasma, maga, nano.
- 18–23 sono sulle **curve** mobili e sono animali: gufo, topo, falena, ragno, lucertola, scarabeo.

---

## 2. Preparazione (`GameState.new_game`)
1. Le 16 tessere fisse vengono messe al loro posto.
2. Le 34 tessere mobili vengono mescolate a caso. Le prime 33 riempiono, in ordine di lettura, le caselle con riga o colonna dispari; ognuna riceve una rotazione casuale tra le 4 possibili. L'ultima diventa la tessera libera, anch'essa con rotazione casuale.
3. Le **24 carte tesoro** vengono mescolate e distribuite coperte. Ogni giocatore riceve un mazzo ordinato di `24 // N` carte: 12 carte con 2 giocatori, 8 con 3, 6 con 4. Il parametro `cards_per_player` permette di usarne meno.
4. **Giocatori e partenze**:
   - con 4 giocatori: Giallo (0,0), Rosso (0,6), Blu (6,6), Verde (6,0);
   - con 3 giocatori: i primi tre;
   - con 2 giocatori: gli angoli opposti (0,0) e (6,6), come nel regolamento.
5. Inizia il giocatore 0. Si gioca in senso orario: 0 → 1 → 2 → 3 → 0…

Tutta la casualità della partita sta in questa preparazione (tessere, rotazioni, mazzi). Una volta definita, con lo stesso seed e le stesse azioni la partita si ripete identica.

---

## 3. Il turno
Ogni turno ha **due fasi obbligatorie**, sempre dello stesso giocatore. Nell'ambiente RL ogni fase è una decisione separata, quindi un turno corrisponde a 2 azioni.

### 3.1 Fase 1: SPINTA (inserire la tessera libera)
1. Il giocatore sceglie una **rotazione** della tessera libera: `rot` ∈ {0, 1, 2, 3}, cioè il numero di quarti di giro in senso orario rispetto all'orientamento attuale.
2. Sceglie uno dei **12 slot** di inserimento. Gli slot sono 3 per lato, allineati alle righe e colonne dispari 1, 3, 5:

| Slot | Lato | Linea | Effetto |
|---|---|---|---|
| 0, 1, 2 | alto | colonna 1, 3, 5 | la colonna scende di una casella |
| 3, 4, 5 | destra | riga 1, 3, 5 | la riga scorre a sinistra |
| 6, 7, 8 | basso | colonna 1, 3, 5 | la colonna sale di una casella |
| 9, 10, 11 | sinistra | riga 1, 3, 5 | la riga scorre a destra |

3. La tessera entra dallo slot e spinge tutta la riga o colonna di una casella. La tessera che esce dal lato opposto **diventa la nuova tessera libera**, con l'orientamento che aveva sul tabellone e con l'eventuale tesoro.
4. **Pedine trascinate**: ogni pedina sulla riga o colonna spinta si sposta insieme alla sua tessera. Una pedina che si trova sulla tessera espulsa **rientra dal lato opposto**, sulla tessera appena inserita. Questo vale per le pedine di tutti i giocatori, non solo di chi spinge.
5. **Divieto di annullare**: non si può inserire nello slot opposto a quello usato dalla spinta precedente, da chiunque sia stata fatta. Lo slot opposto a `s` è `(s + 6) mod 12`: per esempio dopo lo slot 0 (alto, colonna 1) è vietato lo slot 6 (basso, colonna 1). Al primo turno della partita non c'è nessun divieto, quindi gli slot legali sono 12; in tutti gli altri turni sono 11.
6. Le tessere fisse non si muovono mai.

**Codifica dell'azione**: `azione = slot × 4 + rot`, nell'intervallo 0…47. Per le tessere dritte le rotazioni `r` e `r+2` danno la stessa tessera: sono azioni diverse ma con lo stesso effetto.

### 3.2 Fase 2: MOVIMENTO
1. Il giocatore muove la propria pedina in **qualsiasi casella raggiungibile** attraverso corridoi collegati. Il percorso può essere lungo quanto si vuole.
2. Due caselle adiacenti sono collegate **solo se entrambe** hanno un'apertura sul lato in comune (per esempio E per la casella a sinistra e O per quella a destra).
3. **Restare fermi è sempre permesso**: la propria casella è sempre raggiungibile.
4. Le pedine non si bloccano tra loro: si può attraversare una casella occupata e anche fermarcisi.
5. Non si esce dal tabellone muovendosi; si esce solo venendo spinti, come al punto 3.1.4.

**Codifica dell'azione**: `azione = 48 + riga × 7 + colonna`, nell'intervallo 48…96.

### 3.3 Fine del turno
Dopo il movimento si controlla la raccolta (sezione 4), poi tocca al giocatore successivo, che inizia di nuovo dalla fase SPINTA.

---

## 4. Obiettivi, raccolta e vittoria
- Ogni giocatore conosce **solo la prima carta** del proprio mazzo: è il suo **obiettivo corrente**.
- **Raccolta**: se **al termine del proprio movimento** la pedina si trova sulla tessera con il tesoro dell'obiettivo corrente, la carta viene raccolta. La carta raccolta diventa pubblica e l'obiettivo passa alla carta successiva del mazzo.
  - Si raccoglie **solo alla fine del proprio movimento**. Passare sopra il tesoro durante il percorso non conta.
  - Se un avversario ti spinge sopra il tuo tesoro, non lo raccogli subito. Lo raccogli al tuo turno, se a fine movimento sei ancora lì; puoi anche scegliere di restare fermo.
  - Si raccoglie al massimo **una carta per turno**.
  - Fermarsi sul tesoro di un altro giocatore non ha nessun effetto.
- Un tesoro già raccolto resta disegnato sulla tessera, ma non è più l'obiettivo di nessuno.
- Se il tesoro obiettivo è sulla **tessera libera**, in quel momento non si può raggiungere: bisogna prima reinserire quella tessera nel tabellone, per esempio con la propria spinta.
- **Ritorno a casa**: quando hai raccolto tutte le carte, il tuo obiettivo diventa la tua **casella di partenza**.
- **Vittoria**: vince il **primo** giocatore che termina il movimento sulla propria casella di partenza dopo aver raccolto tutte le carte. La partita finisce subito.

### 4.1 Limite di turni (solo nel training)
- Se nessuno vince entro `max_turns` turni (default 400 nell'ambiente, 200 nel self-play AlphaZero, 10.000 nella pagina di gioco), la partita viene **troncata**.
- Nell'impostazione `"progress"` (il default) si dividono la vittoria i giocatori con la frazione più alta di carte raccolte. Nell'impostazione `"draw"` la partita finisce in pareggio.

⚠ **Differenza**: il limite di turni non esiste nel gioco reale; serve solo per evitare partite infinite tra agenti deboli.

---

## 5. Informazione nascosta (POMDP)
Ogni giocatore **vede**:
- tutto il tabellone, la tessera libera e tutte le pedine;
- per ogni giocatore, quante carte ha raccolto e quante gliene mancano;
- tutte le carte già raccolte, perché vengono rivelate;
- la propria carta corrente.

Ogni giocatore **non vede**:
- l'obiettivo corrente degli avversari;
- le carte future di tutti, comprese le proprie.

Le modalità alternative, utili solo per la ricerca, sono:
- `observability="full"`: si vedono anche gli obiettivi correnti degli avversari;
- `view_radius=k`: nebbia di guerra intorno alla propria pedina. Non esiste nel gioco reale.

---

## 6. Ricompense (solo RL)
- **Fine partita, a somma zero**: +1 al vincitore e −1/(N−1) a ciascuno degli altri giocatori.
- **Opzionali, per facilitare l'apprendimento**: un bonus per ogni carta raccolta (`treasure`) e uno shaping basato sulla distanza dal proprio obiettivo (`distance_shaping`). In `RewardConfig` valgono 0 di default; lo script PPO li attiva con i valori 0.1 e 0.05.

---

## 7. Riepilogo delle differenze dal gioco in scatola
1. Esiste un limite di turni, usato solo nel training.
2. Con meno di 4 giocatori si usano comunque tutte le 24 carte, divise in parti uguali, come nel regolamento base.
3. Il primo a giocare è sempre il giocatore 0 (Giallo), mentre nel regolamento inizia il più giovane.
4. Tutto il resto segue il regolamento ufficiale e il materiale della scatola: tabellone, tessere, tesori, spinta, pedine che rientrano dal lato opposto, divieto di annullare la spinta precedente, raccolta solo a fine movimento, ritorno a casa.
