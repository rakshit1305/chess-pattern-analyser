# ♟️ Chess Win Pattern Analyzer 

Predict chess game outcomes (white win / draw / black win) from everything knowable after move 10: ratings, opening, time control,
the board (material, development, pawn structure, king safety, mobility ...), **engine evaluation**, **clock usage** and **player history**.
Data: Lichess 2013-2026 (months [1, 7]) + OTB championships via Lichess broadcasts (79,661 games, 93,972 players, 2012-12-31 to 2026-11-20).

**Live app:** _add your Streamlit URL here_

## Results (chronological hold-out: games from 2025-01-01, n = 12,038)
| Metric | Value | 95% bootstrap CI |
|---|---|---|
| Accuracy | 0.583 | 0.574 - 0.592 |
| Macro F1 | 0.480 | 0.466 - 0.493 |
| Log loss | 0.805 | 0.796 - 0.816 |
| ROC-AUC (macro OvR) | 0.712 | |
| Baseline "higher Elo wins" accuracy | 0.526 | |
| Baseline Elo-only logistic regression log loss | 0.879 | |

## Method
1. Stream Lichess monthly dumps 2013-2026 (first N games per month) + official Lichess broadcasts of OTB championships; parse with python-chess.
2. 74 features from the position after 10 moves, incl. Stockfish / Lichess evaluation, ACPL, clocks and strictly-past player history.
3. Chronological train/test split; models compared with month-grouped 5-fold CV only; ablation by feature group; class-weight and player-disjoint experiments.
4. Randomized hyper-parameter search (log loss), bootstrap CIs, calibration, McNemar tests, grouped permutation importance, SHAP.
5. Streamlit app: explore, players, live prediction (sliders, pasted PGN with clocks/evals, or a Lichess game URL), model dashboard.

## Run locally
```
pip install -r requirements.txt   # and install stockfish for live engine evals (apt-get install stockfish)
streamlit run app.py
```

## Limitations
Lichess games are sampled from the start of each month (time-of-day bias); player history is short for online players; FIDE and Lichess
ratings are different scales (the `source` feature lets the model learn separate behaviour); the engine depth is shallow for speed.
