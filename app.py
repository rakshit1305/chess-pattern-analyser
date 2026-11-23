import io, json, math, os, re, shutil
import numpy as np, pandas as pd, joblib, requests
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
import chess, chess.pgn
from features import (game_features, board_from_moves, normalise_name, title_rank, time_category,
                      BASIC_BOARD, EXTRA_BOARD, CLOCK, ENGINE, HISTORY, CATEGORICAL)

BASE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="Chess Win Pattern Analyzer v2", page_icon="♟️", layout="wide")
COLORS = {"white": "#cbd5e1", "draw": "#f59e0b", "black": "#334155"}
SRC_COLORS = {"lichess": "#0ea5e9", "otb": "#f59e0b"}


# ----------------------------------------------------------------------------- loading
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE, "chess_model.joblib"))


@st.cache_data
def load_csv(name, **kw):
    p = os.path.join(BASE, name)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


@st.cache_data
def load_games():
    d = pd.read_csv(os.path.join(BASE, "games_sample.csv"), parse_dates=["date"])
    otb = load_csv("games_otb.csv", parse_dates=["date"])
    if otb is not None:                              # keep every championship game, de-duplicated against the sample
        d = pd.concat([d, otb], ignore_index=True).drop_duplicates(subset=["date", "white", "black", "moves_san"]).reset_index(drop=True)
    d["elo_diff"] = d.white_elo - d.black_elo
    d["avg_elo"] = (d.white_elo + d.black_elo) / 2
    return d


@st.cache_data
def load_json(name):
    with open(os.path.join(BASE, name)) as f:
        return json.load(f)


@st.cache_resource
def find_engine():
    for c in ("stockfish", "/usr/games/stockfish", "/usr/bin/stockfish"):
        p = shutil.which(c) if not os.path.isabs(c) else (c if os.path.exists(c) else None)
        if p:
            return p
    return None


def show_fig(name, caption=None):
    p = os.path.join(BASE, "figures", f"{name}.png")
    if os.path.exists(p):
        st.image(p, caption=caption)


bundle, M = load_model(), load_json("metrics.json")
model, CLASSES, NUM, CAT, GROUPS = bundle["pipeline"], bundle["classes"], bundle["num"], bundle["cat"], bundle["groups"]
ALL = NUM + CAT
df = load_games()
players = load_csv("player_table.csv")
medians = load_json("feature_medians.json")
ENGINE_PATH = find_engine()

st.title("♟️ Chess Win Pattern Analyzer v2")
st.caption(f"{M['n_games']:,} games · {M['n_players']:,} players · {M['date_range'][0]} → {M['date_range'][1]} · "
           f"online (Lichess) + over-the-board championships · {M['n_features']} features incl. engine evaluation, clocks and player history")

# ----------------------------------------------------------------------------- sidebar filters
st.sidebar.header("Explore filters")
src_sel = st.sidebar.multiselect("Source", ["lichess", "otb"], default=["lichess", "otb"],
                                 format_func=lambda s: {"lichess": "Lichess (online)", "otb": "Over the board"}[s])
yr = st.sidebar.slider("Year", int(df.year.min()), int(df.year.max()), (int(df.year.min()), int(df.year.max())))
tcs_all = sorted(df.time_cat.dropna().unique())
tcs = st.sidebar.multiselect("Time control", tcs_all, default=tcs_all)
lo, hi = int(df.avg_elo.min()), int(df.avg_elo.max())
elo_range = st.sidebar.slider("Average rating", lo, hi, (lo, hi))
fd = df[df.source.isin(src_sel) & df.year.between(*yr) & df.time_cat.isin(tcs) & df.avg_elo.between(*elo_range)]
st.sidebar.write(f"{len(fd):,} games match")

tab_over, tab_exp, tab_players, tab_pred, tab_perf, tab_about = st.tabs(
    ["🏠 Overview", "📊 Explore", "👤 Players", "🔮 Predict", "📈 Model performance", "ℹ️ About"])

# ============================================================================= OVERVIEW
with tab_over:
    t, ci = M["test"], M["test_ci95"]
    c = st.columns(5)
    c[0].metric("Games", f"{M['n_games']:,}", f"{M['games_by_source'].get('otb', 0):,} over-the-board")
    c[1].metric("Test accuracy (2025-26)", f"{t['accuracy']:.1%}", f"{t['accuracy'] - M['baseline_rule_accuracy']:+.1%} vs 'higher Elo wins'")
    c[2].metric("Log loss", f"{t['log_loss']:.3f}", f"{t['log_loss'] - M['baseline_elo_logreg']['log_loss']:+.3f} vs Elo-only", delta_color="inverse")
    c[3].metric("ROC-AUC (macro)", f"{t['roc_auc_ovr']:.3f}")
    c[4].metric("Macro F1", f"{t['f1_macro']:.3f}", f"CI {ci['f1_macro'][0]:.3f}-{ci['f1_macro'][1]:.3f}")
    abl = pd.DataFrame(M["ablation_nested"])
    gain = abl.log_loss.iloc[0] - abl.log_loss.iloc[-1]
    st.markdown(f"""
**Key findings**
- The rating gap is the single strongest predictor, in online and over-the-board chess alike.
- Everything else together lowers cross-validated log loss from **{abl.log_loss.iloc[0]:.3f}** (rating only) to **{abl.log_loss.iloc[-1]:.3f}** ({gain:+.3f}); see the ablation chart for which groups matter.
- Over-the-board championship games are ~{(df[df.source=='otb'].label=='draw').mean():.0%} draws, online games ~{(df[df.source=='lichess'].label=='draw').mean():.0%}: the model needs to know *where* a game was played.
- White's first-move advantage is statistically significant in both worlds ({M['stats']['white_share_decisive_lichess']:.1%} / {M['stats']['white_share_decisive_otb']:.1%} of decisive games).
- The test set is the **future** (games from {M['test_from']}), so these numbers measure generalisation to today's players.
""")
    show_fig("ablation", "Ablation: what each feature group adds (cross-validation inside the training period)")
    show_fig("eda_overview")

# ============================================================================= EXPLORE
with tab_exp:
    st.subheader("Rating difference and outcome")
    c1, c2 = st.columns(2)
    b = fd.assign(bin=pd.cut(fd.elo_diff, np.arange(-400, 401, 50))).dropna(subset=["bin"])
    b = b.groupby(["bin", "source"], observed=True).label.value_counts(normalize=True).rename("share").reset_index()
    b["mid"] = b["bin"].apply(lambda i: i.mid)
    c1.plotly_chart(px.line(b, x="mid", y="share", color="label", line_dash="source", markers=True, color_discrete_map=COLORS,
                            title="Outcome by rating difference (White - Black), by source"))
    br = fd.groupby(["source"]).label.value_counts(normalize=True).rename("share").reset_index()
    c2.plotly_chart(px.bar(br, x="source", y="share", color="label", color_discrete_map=COLORS, title="Outcome distribution by source"))

    st.subheader("Openings")
    c1, c2 = st.columns([1, 3])
    outcome = c1.radio("Outcome", ["white", "draw", "black"])
    min_games = c1.slider("Minimum games per opening", 5, 300, 30)
    n_show = c1.slider("Openings to show", 5, 30, 15)
    g = fd.groupby("opening").agg(games=("label", "size"), hits=("label", lambda s: (s == outcome).sum()))
    g["rate"] = g.hits / g.games; g["ci"] = 1.96 * np.sqrt(g.rate * (1 - g.rate) / g.games)
    g = g[g.games >= min_games].sort_values("rate", ascending=False).head(n_show).reset_index()
    if len(g):
        c2.plotly_chart(px.bar(g.iloc[::-1], x="rate", y="opening", orientation="h", error_x="ci", hover_data=["games"], labels={"rate": f"{outcome} rate"}))
    else:
        c2.info("No openings meet the minimum-games threshold with these filters.")

    st.subheader("New signals: engine evaluation, clocks, player history")
    c1, c2, c3 = st.columns(3)
    e = fd[fd.has_eval == 1]
    if len(e):
        e = e.assign(eb=pd.cut(e.eval_cp, [-1001, -300, -150, -75, -25, 25, 75, 150, 300, 1001]).astype(str))
        e = e.groupby("eb").label.value_counts(normalize=True).rename("share").reset_index()
        c1.plotly_chart(px.bar(e, x="eb", y="share", color="label", color_discrete_map=COLORS, title="Outcome vs engine eval after move 10 (cp)"))
    k = fd[fd.has_clock == 1]
    if len(k):
        k = k.assign(d=pd.cut(k.clock_frac_diff, [-1, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 1]).astype(str))
        k = k.groupby("d").label.apply(lambda s: (s == "white").mean()).rename("white_win_rate").reset_index()
        c2.plotly_chart(px.bar(k, x="d", y="white_win_rate", title="White win rate vs clock lead after move 10"))
    h = fd[fd.hist_score_diff.notna()]
    if len(h):
        h = h.assign(d=pd.cut(h.hist_score_diff, [-1, -0.2, -0.1, -0.03, 0.03, 0.1, 0.2, 1]).astype(str))
        h = h.groupby("d").label.apply(lambda s: (s == "white").mean()).rename("white_win_rate").reset_index()
        c3.plotly_chart(px.bar(h, x="d", y="white_win_rate", title="White win rate vs prior-score difference"))

    st.subheader("Championship events")
    ev = fd[fd.source == "otb"]
    if len(ev):
        evs = ev.groupby("event").agg(games=("label", "size"), draws=("label", lambda s: (s == "draw").mean()),
                                      white=("label", lambda s: (s == "white").mean()), avg_elo=("avg_elo", "mean"),
                                      first=("date", "min")).sort_values("games", ascending=False)
        evs["first"] = evs["first"].dt.date
        st.dataframe(evs.head(40).round({"draws": 3, "white": 3, "avg_elo": 0}))
    else:
        st.info("Select 'Over the board' in the sidebar to see events.")

    st.subheader("Time control and game length (game length is for exploration only, not a model feature)")
    c1, c2 = st.columns(2)
    tc = fd.groupby("time_cat").label.value_counts(normalize=True).rename("share").reset_index()
    c1.plotly_chart(px.bar(tc, x="time_cat", y="share", color="label", barmode="group", color_discrete_map=COLORS, title="Outcome by time control"))
    c2.plotly_chart(px.histogram(fd[fd.num_moves <= 120], x="num_moves", color="label", barmode="overlay", opacity=0.55,
                                 histnorm="probability density", color_discrete_map=COLORS, title="Game length (moves) by outcome"))

# ============================================================================= PLAYERS
with tab_players:
    st.subheader("Look up a player")
    q = st.text_input("Name (Lichess username or OTB name, e.g. 'Carlsen', 'Gukesh', 'DrNykterstein')", "")
    if q and players is not None:
        hits = players[players.name.str.contains(q, case=False, na=False)].sort_values("games", ascending=False).head(15)
        if len(hits):
            st.dataframe(hits[["name", "source", "games", "score", "draw_rate", "score_as_white", "score_as_black", "first_elo", "last_elo", "last_date"]].round(3))
            pick = st.selectbox("Show games of", hits.name.tolist())
            key = players.loc[players.name == pick, "player_key"].iloc[0]
            src = key.split(":")[0]
            pg = df[((df.white.map(lambda n: normalise_name(n, src)) == key) | (df.black.map(lambda n: normalise_name(n, src)) == key))].sort_values("date")
            if len(pg):
                pg = pg.assign(player_elo=np.where(pg.white.map(lambda n: normalise_name(n, src)) == key, pg.white_elo, pg.black_elo),
                               colour=np.where(pg.white.map(lambda n: normalise_name(n, src)) == key, "white", "black"))
                pg["score"] = np.where(pg.label == "draw", 0.5, np.where(pg.label == pg.colour, 1.0, 0.0))
                c1, c2 = st.columns(2)
                c1.plotly_chart(px.line(pg, x="date", y="player_elo", markers=True, title=f"Rating over time ({len(pg)} games in our data)"))
                c2.plotly_chart(px.bar(pg.groupby("colour").score.mean().reset_index(), x="colour", y="score", title="Average score as White / as Black", range_y=[0, 1]))
                st.dataframe(pg[["date", "event", "white", "black", "white_elo", "black_elo", "result", "opening", "time_cat", "moves_san"]].tail(50))
            else:
                st.info("This player is in the player table, but none of their games are in the app's game sample.")
        else:
            st.warning("No player with that name in the data.")
    st.subheader("Most active players")
    if players is not None:
        c1, c2 = st.columns(2)
        c1.dataframe(players[players.source == "otb"].sort_values("games", ascending=False).head(20)[["name", "games", "score", "draw_rate", "last_elo"]].round(3))
        c2.dataframe(players[players.source == "lichess"].sort_values("games", ascending=False).head(20)[["name", "games", "score", "draw_rate", "last_elo"]].round(3))
    st.subheader("Head to head (over-the-board)")
    otb_names = players[players.source == "otb"].sort_values("games", ascending=False).name.tolist() if players is not None else []
    if len(otb_names) >= 2:
        c1, c2 = st.columns(2)
        a = c1.selectbox("Player A", otb_names, index=0); bb = c2.selectbox("Player B", otb_names, index=1)
        ka, kb = normalise_name(a, "otb"), normalise_name(bb, "otb")
        wk, bk = df.white.map(lambda n: normalise_name(n, "otb")), df.black.map(lambda n: normalise_name(n, "otb"))
        hh = df[((wk == ka) & (bk == kb)) | ((wk == kb) & (bk == ka))].sort_values("date")
        if len(hh):
            a_is_white = (hh.white.map(lambda n: normalise_name(n, "otb")) == ka).to_numpy()
            lab = hh.label.to_numpy()
            sa = np.where(lab == "draw", 0.5, np.where((lab == "white") == a_is_white, 1.0, 0.0)).mean()
            st.write(f"{len(hh)} games · score of **{a}**: {sa:.2f}")
            st.dataframe(hh[["date", "event", "white", "black", "result", "opening", "moves_san"]])
        else:
            st.info("No games between these two players in the data.")


# ============================================================================= PREDICT
def history_feats(name, source, colour):
    """Player-history features for the app from the final player table (NaN when the player is unknown)."""
    nan = float("nan"); pre = colour
    out = {f"{pre}_hist_games": 0, f"{pre}_hist_score": nan, f"{pre}_hist_colour_score": nan, f"{pre}_hist_draw_rate": nan,
           f"{pre}_rating_trend": nan, f"{pre}_eco_familiarity": nan, f"{pre}_days_since_last": nan}
    if not name or players is None:
        return out, None
    key = normalise_name(name, source)
    row = players[players.player_key == key]
    if row.empty:
        return out, None
    r = row.iloc[0]; k = 5; n = r.games
    cs = r.score_as_white if colour == "w" else r.score_as_black
    out.update({f"{pre}_hist_games": int(n), f"{pre}_hist_score": (r.score * n + 0.5 * k) / (n + k),
                f"{pre}_hist_colour_score": cs if not pd.isna(cs) else nan, f"{pre}_hist_draw_rate": r.draw_rate,
                f"{pre}_rating_trend": (r.last_elo - r.first_elo) if not pd.isna(r.first_elo) else nan})
    return out, r


def engine_eval(fen, depth=10):
    if not ENGINE_PATH:
        return None
    try:
        import chess.engine
        with chess.engine.SimpleEngine.popen_uci(ENGINE_PATH) as eng:
            info = eng.analyse(chess.Board(fen), chess.engine.Limit(depth=depth))
            return float(max(-1000, min(1000, info["score"].white().score(mate_score=10000))))
    except Exception:
        return None


with tab_pred:
    st.subheader("Predict a game result from the first 10 moves")
    st.caption("Everything the model sees is knowable after move 10. Missing information (no clocks, unknown players, no engine) is allowed: the model was trained with such gaps.")
    eco_names = df.groupby("eco").opening.agg(lambda s: s.mode().iat[0] if len(s) else "")
    eco_list = sorted(df.eco.dropna().unique())
    mode = st.radio("Input mode", ["Quick sliders", "Paste PGN / moves", "Lichess game URL or ID"], horizontal=True)

    c1, c2, c3 = st.columns(3)
    source = c1.selectbox("Where is the game played?", ["lichess", "otb"], format_func=lambda s: {"lichess": "Online (Lichess rating)", "otb": "Over the board (FIDE rating)"}[s])
    w_elo = c1.number_input("White rating", 600, 3200, 1500 if source == "lichess" else 2500, step=10)
    b_elo = c1.number_input("Black rating", 600, 3200, 1500 if source == "lichess" else 2500, step=10)
    tc_options = [t for t in tcs_all if t.startswith("otb")] if source == "otb" else [t for t in tcs_all if not t.startswith("otb")]
    tc_ = c2.selectbox("Time control", tc_options, index=min(1, len(tc_options) - 1))
    base_minutes = c2.number_input("Base time (minutes)", 0.25, 180.0, 90.0 if source == "otb" else 5.0, step=0.25)
    eco = c2.selectbox("Opening (ECO)", eco_list, index=eco_list.index("C50") if "C50" in eco_list else 0, format_func=lambda e: f"{e}: {eco_names.get(e, '')}")
    w_name = c3.text_input("White player name (optional, for history features)", "")
    b_name = c3.text_input("Black player name (optional)", "")
    w_title = c3.selectbox("White title", ["", "GM", "IM", "FM", "CM", "WGM", "WIM", "WFM"], 0)
    b_title = c3.selectbox("Black title", ["", "GM", "IM", "FM", "CM", "WGM", "WIM", "WFM"], 0)
    is_event = c3.checkbox("Tournament / event game", value=(source == "otb"))

    feats = {k: medians.get(k, float("nan")) for k in BASIC_BOARD + EXTRA_BOARD}
    feats.update({k: float("nan") for k in CLOCK + ENGINE})
    game, fen, ok = None, None, True
    if mode == "Quick sliders":
        d1, d2, d3 = st.columns(3)
        feats["mat_diff"] = d1.slider("Material lead (White - Black)", -9, 9, 0)
        feats["w_developed"] = d1.slider("White minor pieces developed", 0, 4, 3); feats["b_developed"] = d1.slider("Black minor pieces developed", 0, 4, 3)
        feats["w_castled"] = int(d2.checkbox("White castled", True)); feats["b_castled"] = int(d2.checkbox("Black castled", False))
        feats["queens_off"] = int(d2.checkbox("Queens traded", False)); feats["center_diff"] = d2.slider("Centre pawns (White - Black)", -2, 2, 0)
        feats["mob_diff"] = d3.slider("Mobility lead (legal moves, White - Black)", -20, 20, 0)
        ev_in = d3.text_input("Engine eval after move 10 in centipawns (optional, e.g. +35)", "")
        if ev_in.strip():
            try:
                feats["eval_cp"] = float(ev_in.replace("+", "")); feats["eval_abs"] = abs(feats["eval_cp"])
            except ValueError:
                st.warning("Could not read the eval, leaving it empty.")
        clk = d3.slider("Clock lead after move 10 (fraction of base time, White - Black)", -0.5, 0.5, 0.0, 0.01)
        if clk != 0.0:
            feats.update({"clock_frac_diff": clk, "w_clock_frac_left": 0.8 + clk / 2, "b_clock_frac_left": 0.8 - clk / 2,
                          "w_max_think_frac": 0.1, "b_max_think_frac": 0.1, "w_time_trouble": 0, "b_time_trouble": 0})
    else:
        if mode == "Paste PGN / moves":
            text = st.text_area("PGN or moves (clock / eval comments like { [%clk 0:04:55] [%eval 0.31] } are used when present)",
                                "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Nb8 10. d4 Nbd7", height=110)
        else:
            gid = st.text_input("Lichess game URL or ID", "https://lichess.org/")
            text = ""
            m = re.search(r"([A-Za-z0-9]{8})", gid.replace("https://lichess.org/", "").split("/")[0]) if gid else None
            if m:
                try:
                    r = requests.get(f"https://lichess.org/game/export/{m.group(1)}", params={"clocks": "true", "evals": "true", "opening": "true"},
                                     headers={"Accept": "application/x-chess-pgn"}, timeout=15)
                    text = r.text if r.status_code == 200 else ""
                    if not text:
                        st.error(f"Lichess returned {r.status_code} for game {m.group(1)}")
                except Exception as e:
                    st.error(f"Could not fetch the game: {e}")
        if text.strip():
            try:
                game = board_from_moves(text)
                h = game.headers
                if "WhiteElo" in h and h["WhiteElo"].isdigit(): w_elo = int(h["WhiteElo"])
                if "BlackElo" in h and h["BlackElo"].isdigit(): b_elo = int(h["BlackElo"])
                if h.get("White", "?") != "?": w_name = w_name or h["White"]
                if h.get("Black", "?") != "?": b_name = b_name or h["Black"]
                if h.get("ECO", "?") != "?" and h["ECO"] in eco_list: eco = h["ECO"]
                tcat = time_category(h.get("TimeControl", ""), h.get("Event", ""))
                if tcat in tcs_all: tc_ = tcat
                gf = game_features(game, plies=bundle["plies"])
                if gf is None:
                    ok = False; st.error("The game needs at least 10 full moves (20 plies) and must still be in progress after move 10.")
                else:
                    feats.update({k: gf[k] for k in BASIC_BOARD + EXTRA_BOARD + CLOCK + ENGINE})
                    fen = gf["fen_after"]
                    if not math.isnan(gf["base_minutes"]): base_minutes = gf["base_minutes"]
                    if ENGINE_PATH and math.isnan(feats["eval_cp"]):
                        e = engine_eval(fen)
                        if e is not None: feats["eval_cp"], feats["eval_abs"] = e, abs(e)
                    st.code(str(chess.Board(fen)))
                    st.caption(f"Position after move 10 · engine eval: {feats['eval_cp'] if not math.isnan(feats['eval_cp']) else 'n/a'} cp · "
                               f"clocks {'found' if gf['has_clock'] else 'not found'} · {'engine available' if ENGINE_PATH else 'no engine on this server (paste a PGN with [%eval] comments or install stockfish)'}")
                    with st.expander("All computed features"):
                        st.json({k: (round(v, 3) if isinstance(v, float) and not math.isnan(v) else v) for k, v in gf.items() if k not in ("fen_after", "moves_san")})
            except Exception as e:
                ok = False; st.error(f"Could not read the moves: {e}")

    if ok and st.button("Predict", type="primary"):
        wh, wrow = history_feats(w_name, source, "w"); bh, brow = history_feats(b_name, source, "b")
        hist = {**wh, **bh}
        hist["hist_games_diff"] = hist["w_hist_games"] - hist["b_hist_games"]
        hist["hist_score_diff"] = hist["w_hist_score"] - hist["b_hist_score"]
        hist["h2h_games"], hist["h2h_score"] = 0, float("nan")
        row = {"elo_diff": w_elo - b_elo, "avg_elo": (w_elo + b_elo) / 2, "source": source, "time_cat": tc_, "eco": eco,
               "w_title_rank": title_rank(w_title), "b_title_rank": title_rank(b_title), "title_rank_diff": title_rank(w_title) - title_rank(b_title),
               "is_event_game": int(is_event), "base_minutes": base_minutes,
               "has_clock": int(not math.isnan(feats.get("w_clock_frac_left", float("nan")))),
               "has_eval": int(not math.isnan(feats.get("eval_cp", float("nan")))), "has_acpl": int(not math.isnan(feats.get("w_acpl", float("nan")))),
               **feats, **hist}
        X = pd.DataFrame([row]).reindex(columns=ALL)
        p = model.predict_proba(X)[0]
        out = pd.DataFrame({"outcome": CLASSES, "probability": p})
        left, right = st.columns(2)
        left.plotly_chart(px.bar(out, x="outcome", y="probability", color="outcome", color_discrete_map=COLORS, range_y=[0, 1], text_auto=".1%"))
        left.success(f"Most likely outcome: **{CLASSES[int(p.argmax())]}** ({p.max():.0%}).")
        notes = []
        if wrow is not None: notes.append(f"White history found: {int(wrow.games)} games, score {wrow.score:.2f}")
        if brow is not None: notes.append(f"Black history found: {int(brow.games)} games, score {brow.score:.2f}")
        if w_name and wrow is None: notes.append("White player not in the data -> history features left empty")
        if b_name and brow is None: notes.append("Black player not in the data -> history features left empty")
        if notes: left.caption(" · ".join(notes))
        sweep = pd.concat([X.assign(elo_diff=d) for d in range(-400, 401, 25)], ignore_index=True)
        sw = pd.DataFrame(model.predict_proba(sweep), columns=CLASSES).assign(x=sweep.elo_diff).melt("x", var_name="outcome", value_name="probability")
        right.plotly_chart(px.line(sw, x="x", y="probability", color="outcome", color_discrete_map=COLORS, title="What-if: rating difference"))
        sweep2 = pd.concat([X.assign(eval_cp=e, eval_abs=abs(e), has_eval=1) for e in range(-400, 401, 25)], ignore_index=True)
        sw2 = pd.DataFrame(model.predict_proba(sweep2), columns=CLASSES).assign(x=sweep2.eval_cp).melt("x", var_name="outcome", value_name="probability")
        right.plotly_chart(px.line(sw2, x="x", y="probability", color="outcome", color_discrete_map=COLORS, title="What-if: engine evaluation after move 10 (cp)"))
        st.caption("Chess is noisy. These are probabilities learned from past games, not guarantees.")

# ============================================================================= MODEL PERFORMANCE
with tab_perf:
    st.subheader("1. Model selection by cross-validation only (training period, folds grouped by month)")
    st.dataframe(load_csv("cv_results.csv", index_col=0))
    st.subheader("2. Ablation: what each feature group adds")
    abl = load_csv("ablation.csv")
    if abl is not None:
        nst = abl[abl.kind == "nested"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nst.step, y=nst.log_loss, error_y=dict(array=nst.log_loss_std), mode="lines+markers", name="CV log loss"))
        fig.add_trace(go.Scatter(x=nst.step, y=nst.accuracy, mode="lines+markers", name="CV accuracy", yaxis="y2"))
        fig.update_layout(yaxis=dict(title="log loss"), yaxis2=dict(title="accuracy", overlaying="y", side="right"), title="Nested ablation (feature groups added one by one)")
        c1, c2 = st.columns(2); c1.plotly_chart(fig)
        dr = abl[abl.kind == "drop-one"].sort_values("delta_vs_full")
        c2.plotly_chart(px.bar(dr, x="delta_vs_full", y="step", orientation="h", title="Increase in CV log loss when a group is removed"))
    st.subheader("3. Final test set (games from " + M["test_from"] + ", used once)")
    st.dataframe(load_csv("test_results.csv", index_col=0))
    ci = M["test_ci95"]
    st.markdown(f"95% bootstrap CIs: accuracy {ci['accuracy'][0]:.3f}-{ci['accuracy'][1]:.3f}, log loss {ci['log_loss'][0]:.3f}-{ci['log_loss'][1]:.3f}, "
                f"macro F1 {ci['f1_macro'][0]:.3f}-{ci['f1_macro'][1]:.3f}. McNemar vs 'higher Elo wins': p = {M['mcnemar_p_vs_rule']:.1e}; vs Elo-only logistic regression: p = {M['mcnemar_p_vs_elo_logreg']:.1e}.")
    c1, c2 = st.columns(2)
    c1.markdown("**By source**"); c1.dataframe(load_csv("test_by_source.csv", index_col=0))
    c2.markdown("**Per class**"); c2.dataframe(pd.DataFrame(M["per_class"]).T.loc[M["classes"], ["precision", "recall", "f1-score", "support"]].round(3))
    st.subheader("4. Draw handling and player-disjoint robustness check")
    c1, c2 = st.columns(2)
    c1.dataframe(load_csv("draw_results.csv", index_col=0)); c2.dataframe(load_csv("player_disjoint.csv", index_col=0))
    st.subheader("5. Figures")
    show_fig("confusion_roc"); show_fig("calibration"); show_fig("error_analysis")
    c1, c2 = st.columns(2)
    with c1: show_fig("permutation_importance")
    with c2: show_fig("shap_summary")
    show_fig("learning_curve")
    st.markdown(f"**Tuned XGBoost parameters:** `{M['best_params']}` · class weights: {'balanced' if M['balanced_weights'] else 'none'} · evaluation source: {M['eval_source']}")

# ============================================================================= ABOUT
with tab_about:
    st.markdown(f"""
### About (v2)
**Data:** {M['dataset']}. {M['n_games']:,} games, {M['n_players']:,} players, {M['n_events']} over-the-board events.
**Target:** white win / draw / black win. **Features ({M['n_features']}):** rating difference and level, source, time control, titles, event flag,
ECO opening, board after 10 moves (material, development, castling, centre, mobility, pawn structure, king safety, bishop pair, open files, space,
hanging pieces, captures/checks), clock usage, engine evaluation + ACPL/blunders, and strictly-past player history (prior score, colour score,
draw rate, rating trend, opening familiarity, days since last game, head-to-head).
**Leakage control:** only information available after move 10; player history uses earlier games only; game length / termination / rating
changes are never features; the test set is the future ({M['test_from']} onwards).
**Models:** Logistic Regression, Random Forest, HistGradientBoosting, XGBoost against majority / Elo-only / 'higher Elo wins' baselines;
chosen by month-grouped cross-validation, tuned on log loss, tested once.

### Limitations (honest list)
- Online games are sampled from the first minutes of each month (time-of-day bias) and online players rarely recur, so player history is
  informative mainly for over-the-board players.
- Lichess and FIDE ratings are different scales; the `source` feature lets the model treat them separately but they are not converted.
- The engine depth is shallow (speed); tactics deeper than that are invisible. Over-the-board files without `%eval` have no ACPL.
- Draws remain hard online (~5 % of games); over the board they are common and the model predicts them reasonably.
- Association, not causation: a feature that predicts wins does not necessarily cause them.

*scikit-learn {M['sklearn']}, XGBoost {M['xgboost']}, Python {M['python']}.*
""")
