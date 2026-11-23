"""
Shared feature code for the Chess Win Pattern Analyzer (v2).

Used by BOTH the training notebook and the Streamlit app, so features are computed
exactly the same way at training time and at prediction time (no train/serve skew).

Everything here is computed from information available AFTER MOVE 10 (20 plies):
  * board features      - material, development, castling, centre, mobility, pawn structure,
                           king safety, bishop pair, open files, space, hanging pieces
  * move-list features  - captures / checks / castling moves in the first 10 moves
  * clock features      - from [%clk] comments (Lichess games since 2017, Lichess broadcasts)
  * engine features     - from [%eval] comments (Lichess analysed games, Lichess broadcasts)
                           and/or a local Stockfish evaluation of the position after ply 20
Player-history features are computed in the notebook (they need the whole dataset in
chronological order) but the helper that turns a history snapshot into features lives here.
"""
import math
import re

import chess
import chess.pgn

PLIES = 20                         # position is read after 20 half-moves = 10 full moves
VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
EVAL_CLIP = 1000                   # centipawns; mate scores are clipped here

# ----------------------------------------------------------------------------- feature lists
BASIC_BOARD = [
    "mat_diff", "total_material", "w_developed", "b_developed", "w_castled", "b_castled",
    "center_diff", "mob_diff", "queens_off",
]
EXTRA_BOARD = [
    "w_doubled", "b_doubled", "w_isolated", "b_isolated", "pawn_islands_diff",
    "w_king_shield", "b_king_shield", "w_king_attackers", "b_king_attackers",
    "w_bishop_pair", "b_bishop_pair", "rooks_open_diff", "space_diff",
    "w_hanging", "b_hanging", "w_pawns_advanced", "b_pawns_advanced",
    "n_captures", "n_checks", "w_king_moved_uncastled", "b_king_moved_uncastled",
]
CLOCK = [
    "w_clock_frac_left", "b_clock_frac_left", "clock_frac_diff",
    "w_max_think_frac", "b_max_think_frac", "w_time_trouble", "b_time_trouble",
]
ENGINE = [
    "eval_cp",            # evaluation (White's point of view, centipawns) after ply 20
    "eval_abs",           # |eval|: how unbalanced the position already is
    "w_acpl", "b_acpl",   # average centipawn loss over the first 10 moves (needs per-move evals)
    "w_blunders", "b_blunders",
]
HISTORY = [
    "w_hist_games", "b_hist_games", "hist_games_diff",
    "w_hist_score", "b_hist_score", "hist_score_diff",
    "w_hist_colour_score", "b_hist_colour_score",
    "w_hist_draw_rate", "b_hist_draw_rate",
    "w_rating_trend", "b_rating_trend",
    "w_eco_familiarity", "b_eco_familiarity",
    "w_days_since_last", "b_days_since_last",
    "h2h_games", "h2h_score",
]
CONTEXT = ["w_title_rank", "b_title_rank", "title_rank_diff", "is_event_game", "base_minutes"]

FEATURE_GROUPS = {
    "rating":  ["elo_diff", "avg_elo"],
    "context": ["source", "time_cat"] + CONTEXT,
    "opening": ["eco"],
    "board_basic": BASIC_BOARD,
    "board_extra": EXTRA_BOARD,
    "clock":   CLOCK,
    "engine":  ENGINE,
    "history": HISTORY,
}
CATEGORICAL = ["source", "time_cat", "eco"]
TITLE_RANK = {"GM": 6, "IM": 5, "WGM": 4, "FM": 4, "WIM": 3, "CM": 3, "WFM": 2, "WCM": 1, "NM": 2, "LM": 1}


# ----------------------------------------------------------------------------- board helpers
def material(board, color):
    return sum(len(board.pieces(pt, color)) * v for pt, v in VALUES.items())


def mobility(board, color):
    b = board.copy(stack=False)
    b.turn = color
    b.ep_square = None
    return b.legal_moves.count()


def developed(board, color):
    r = 0 if color == chess.WHITE else 7
    at_home = 0
    for f in (1, 6):
        p = board.piece_at(chess.square(f, r))
        at_home += int(p is not None and p.piece_type == chess.KNIGHT and p.color == color)
    for f in (2, 5):
        p = board.piece_at(chess.square(f, r))
        at_home += int(p is not None and p.piece_type == chess.BISHOP and p.color == color)
    return 4 - at_home


def castled(board, color):
    r = 0 if color == chess.WHITE else 7
    return int(board.king(color) in (chess.square(6, r), chess.square(2, r)))


def center_pawns(board, color):
    return sum(1 for s in (chess.D4, chess.E4, chess.D5, chess.E5)
               if (p := board.piece_at(s)) is not None and p.piece_type == chess.PAWN and p.color == color)


def pawn_files(board, color):
    return [chess.square_file(s) for s in board.pieces(chess.PAWN, color)]


def doubled_pawns(files):
    return sum(files.count(f) - 1 for f in set(files) if files.count(f) > 1)


def isolated_pawns(files):
    fs = set(files)
    return sum(1 for f in files if (f - 1) not in fs and (f + 1) not in fs)


def pawn_islands(files):
    fs = sorted(set(files))
    if not fs:
        return 0
    return 1 + sum(1 for a, b in zip(fs, fs[1:]) if b - a > 1)


def king_shield(board, color):
    """Own pawns on the 3 squares directly in front of the king (classic pawn-shield count)."""
    k = board.king(color)
    if k is None:
        return 0
    kf, kr = chess.square_file(k), chess.square_rank(k)
    step = 1 if color == chess.WHITE else -1
    n = 0
    for df in (-1, 0, 1):
        f, r = kf + df, kr + step
        if 0 <= f <= 7 and 0 <= r <= 7:
            p = board.piece_at(chess.square(f, r))
            n += int(p is not None and p.piece_type == chess.PAWN and p.color == color)
    return n


def king_attackers(board, color):
    """Number of enemy pieces attacking the king or the squares around it."""
    k = board.king(color)
    if k is None:
        return 0
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[k]) | chess.SquareSet.from_square(k)
    attackers = chess.SquareSet()
    for s in zone:
        attackers |= board.attackers(not color, s)
    return len(attackers)


def rooks_on_open_files(board, color):
    all_pawn_files = set(pawn_files(board, chess.WHITE)) | set(pawn_files(board, chess.BLACK))
    own_pawn_files = set(pawn_files(board, color))
    n = 0
    for s in board.pieces(chess.ROOK, color):
        f = chess.square_file(s)
        if f not in all_pawn_files:
            n += 1          # open file
        elif f not in own_pawn_files:
            n += 0.5        # semi-open file
    return n


def space(board, color):
    """Squares in the opponent's half attacked by this side (a simple space measure)."""
    ranks = range(4, 8) if color == chess.WHITE else range(0, 4)
    squares = [chess.square(f, r) for r in ranks for f in range(8)]
    return sum(1 for s in squares if board.is_attacked_by(color, s))


def hanging_pieces(board, color):
    """Pieces (not pawns) attacked by the enemy and not defended by a friendly piece."""
    n = 0
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for s in board.pieces(pt, color):
            if board.is_attacked_by(not color, s) and not board.is_attacked_by(color, s):
                n += 1
    return n


def pawns_advanced(board, color):
    """Pawn ranks gained beyond the start rank (how far the pawns have pushed)."""
    total = 0
    for s in board.pieces(chess.PAWN, color):
        r = chess.square_rank(s)
        total += (r - 1) if color == chess.WHITE else (6 - r)
    return total


def board_features(board, moves=None):
    """All board features for a position. `moves` = the list of chess.Move played so far (optional,
    used for capture/check counts and 'king moved without castling')."""
    wm, bm = material(board, chess.WHITE), material(board, chess.BLACK)
    wf, bf = pawn_files(board, chess.WHITE), pawn_files(board, chess.BLACK)
    feats = {
        "mat_diff": wm - bm,
        "total_material": wm + bm,
        "w_developed": developed(board, chess.WHITE),
        "b_developed": developed(board, chess.BLACK),
        "w_castled": castled(board, chess.WHITE),
        "b_castled": castled(board, chess.BLACK),
        "center_diff": center_pawns(board, chess.WHITE) - center_pawns(board, chess.BLACK),
        "mob_diff": mobility(board, chess.WHITE) - mobility(board, chess.BLACK),
        "queens_off": int(not board.pieces(chess.QUEEN, chess.WHITE) and not board.pieces(chess.QUEEN, chess.BLACK)),
        # --- pawn structure
        "w_doubled": doubled_pawns(wf), "b_doubled": doubled_pawns(bf),
        "w_isolated": isolated_pawns(wf), "b_isolated": isolated_pawns(bf),
        "pawn_islands_diff": pawn_islands(wf) - pawn_islands(bf),
        # --- king safety
        "w_king_shield": king_shield(board, chess.WHITE), "b_king_shield": king_shield(board, chess.BLACK),
        "w_king_attackers": king_attackers(board, chess.WHITE), "b_king_attackers": king_attackers(board, chess.BLACK),
        # --- pieces
        "w_bishop_pair": int(len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2),
        "b_bishop_pair": int(len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2),
        "rooks_open_diff": rooks_on_open_files(board, chess.WHITE) - rooks_on_open_files(board, chess.BLACK),
        "space_diff": space(board, chess.WHITE) - space(board, chess.BLACK),
        "w_hanging": hanging_pieces(board, chess.WHITE), "b_hanging": hanging_pieces(board, chess.BLACK),
        "w_pawns_advanced": pawns_advanced(board, chess.WHITE), "b_pawns_advanced": pawns_advanced(board, chess.BLACK),
    }
    # --- move-list features
    n_cap = n_chk = 0
    w_king_moved = b_king_moved = False
    w_castle_move = b_castle_move = False
    if moves:
        b = chess.Board()
        for mv in moves:
            if b.is_capture(mv):
                n_cap += 1
            piece = b.piece_at(mv.from_square)
            if piece is not None and piece.piece_type == chess.KING:
                if b.is_castling(mv):
                    if b.turn == chess.WHITE: w_castle_move = True
                    else: b_castle_move = True
                else:
                    if b.turn == chess.WHITE: w_king_moved = True
                    else: b_king_moved = True
            b.push(mv)
            if b.is_check():
                n_chk += 1
    feats.update({
        "n_captures": n_cap, "n_checks": n_chk,
        "w_king_moved_uncastled": int(w_king_moved and not w_castle_move),
        "b_king_moved_uncastled": int(b_king_moved and not b_castle_move),
    })
    return feats


# ----------------------------------------------------------------------------- PGN comments
_CLK = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
_EVAL = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")


def parse_clk(comment):
    m = _CLK.search(comment or "")
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


def parse_eval(comment):
    """Returns centipawns from White's point of view, or None."""
    m = _EVAL.search(comment or "")
    if not m:
        return None
    v = m.group(1)
    if v.startswith("#"):
        n = int(v[1:])
        return EVAL_CLIP if n > 0 else -EVAL_CLIP
    return max(-EVAL_CLIP, min(EVAL_CLIP, float(v) * 100))


def parse_time_control(tc):
    """Returns (base_seconds, increment_seconds) or (None, None).
    Handles '300+3', '600', '40/7200:20/3600:900+30' (first period only), '-' (correspondence)."""
    if not tc or tc in ("-", "?"):
        return None, None
    first = tc.split(":")[0]
    if "/" in first:
        first = first.split("/", 1)[1]
    m = re.match(r"^(\d+)(?:\+(\d+))?$", first.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2) or 0)


def time_category(tc, event=""):
    """Lichess-style categories for online games, otb_* for over-the-board games."""
    base, inc = parse_time_control(tc)
    ev = (event or "").lower()
    if base is None:
        if ev.startswith("rated ") or "correspondence" in ev or "lichess.org" in ev:
            return None                       # Lichess correspondence game ("-"): no clock -> drop
        # OTB files usually have no TimeControl header: use the event name
        if any(k in ev for k in ("blitz", "armageddon", "bullet")):
            return "otb_blitz"
        if "rapid" in ev:
            return "otb_rapid"
        return None if not ev else "otb_classical"
    total = base + 40 * inc
    if total < 180:  return "bullet"
    if total < 480:  return "blitz"
    if total < 1500: return "rapid"
    return "classical"


def clock_features(clocks_w, clocks_b, base, inc):
    """clocks_w / clocks_b: clock readings (seconds) after each of that side's moves, in order.
    base / inc: time control. Returns NaN features when clocks are unavailable."""
    nan = float("nan")
    out = {k: nan for k in CLOCK}
    if not clocks_w or not clocks_b:
        return out
    if not base:
        # no TimeControl header (typical for broadcast PGNs): the first clock reading is the base time
        base = max(clocks_w[0], clocks_b[0])
        if base <= 0:
            return out
    inc = inc or 0

    def side(clocks):
        left = clocks[-1] / base
        thinks = []
        prev = base
        for c in clocks:
            thinks.append(max(0.0, prev - c + inc))
            prev = c
        return left, (max(thinks) / base if thinks else nan)

    wl, wmax = side(clocks_w)
    bl, bmax = side(clocks_b)
    out.update({
        "w_clock_frac_left": wl, "b_clock_frac_left": bl, "clock_frac_diff": wl - bl,
        "w_max_think_frac": wmax, "b_max_think_frac": bmax,
        "w_time_trouble": int(wl < 0.5), "b_time_trouble": int(bl < 0.5),
    })
    return out


def engine_features_from_evals(evals, final_eval=None):
    """evals: list of centipawn evals (White POV) after each ply (None where missing), length <= PLIES.
    final_eval: an engine eval of the final position (overrides evals[-1] if given).
    ACPL for a side = mean loss, from that side's point of view, over its own moves."""
    nan = float("nan")
    out = {k: nan for k in ENGINE}
    e_final = final_eval if final_eval is not None else (evals[-1] if evals and evals[-1] is not None else None)
    if e_final is not None:
        out["eval_cp"] = float(e_final)
        out["eval_abs"] = abs(float(e_final))
    if evals and sum(e is not None for e in evals) >= len(evals) - 1 and len(evals) >= 2:
        prev = 0.0  # start position ~ 0
        w_loss, b_loss, w_bl, b_bl = [], [], 0, 0
        for i, e in enumerate(evals):
            if e is None:
                continue
            if i % 2 == 0:   # White just moved: loss = prev - e (from White's view)
                loss = max(0.0, prev - e); w_loss.append(loss); w_bl += int(loss >= 200)
            else:            # Black just moved: loss = e - prev
                loss = max(0.0, e - prev); b_loss.append(loss); b_bl += int(loss >= 200)
            prev = e
        if w_loss: out["w_acpl"] = sum(w_loss) / len(w_loss); out["w_blunders"] = w_bl
        if b_loss: out["b_acpl"] = sum(b_loss) / len(b_loss); out["b_blunders"] = b_bl
    return out


def title_rank(title):
    if not title or title == "BOT":
        return 0
    return TITLE_RANK.get(title.upper(), 1)


# ----------------------------------------------------------------------------- one game -> row
def game_features(game, plies=PLIES, engine=None, engine_limit=None):
    """Compute every per-game feature from a chess.pgn.Game (headers + first `plies` moves).
    Returns None if the game has fewer than `plies` moves or is not a standard game.
    `engine`: an optional chess.engine.SimpleEngine; if given, the position after `plies`
    is evaluated with it (engine_limit = chess.engine.Limit)."""
    h = game.headers
    if h.get("Variant", "Standard") not in ("Standard", "From Position") or h.get("FEN"):
        return None
    board = game.board()
    moves, clocks_w, clocks_b, evals = [], [], [], []
    node = game
    for _ in range(plies):
        nxt = node.next()
        if nxt is None:
            return None
        mv = nxt.move
        side_white = board.turn == chess.WHITE
        board.push(mv)
        moves.append(mv)
        c = parse_clk(nxt.comment)
        if c is not None:
            (clocks_w if side_white else clocks_b).append(c)
        evals.append(parse_eval(nxt.comment))
        node = nxt
    if board.is_game_over():
        return None

    base, inc = parse_time_control(h.get("TimeControl", ""))
    row = {}
    row.update(board_features(board, moves))
    row.update(clock_features(clocks_w if len(clocks_w) == plies // 2 else [],
                              clocks_b if len(clocks_b) == plies // 2 else [], base, inc))
    final_eval = None
    if engine is not None:
        try:
            info = engine.analyse(board, engine_limit)
            sc = info["score"].white()
            final_eval = float(max(-EVAL_CLIP, min(EVAL_CLIP, sc.score(mate_score=EVAL_CLIP * 10))))
        except Exception:
            final_eval = None
    row.update(engine_features_from_evals(evals, final_eval))
    row["has_clock"] = int(not math.isnan(row["w_clock_frac_left"]))
    row["has_eval"] = int(not math.isnan(row["eval_cp"]))
    row["has_acpl"] = int(not math.isnan(row["w_acpl"]))
    if not base and clocks_w and clocks_b:
        base = max(clocks_w[0], clocks_b[0])          # inferred from the clocks (broadcast PGNs)
    row["base_minutes"] = (base / 60.0) if base else float("nan")
    ev = h.get("Event", "").lower()
    # Lichess casual games are "Rated Blitz game"; arena/swiss games say "tournament"/"swiss";
    # over-the-board games never start with "rated " -> they are always event games
    row["is_event_game"] = int(not ev.startswith("rated ") or "tournament" in ev or "swiss" in ev)
    row["w_title_rank"] = title_rank(h.get("WhiteTitle"))
    row["b_title_rank"] = title_rank(h.get("BlackTitle"))
    row["title_rank_diff"] = row["w_title_rank"] - row["b_title_rank"]
    row["fen_after"] = board.fen()
    row["moves_san"] = " ".join(game.board().variation_san(moves).split()) if moves else ""
    return row


def board_from_moves(text):
    """Build a chess.pgn.Game from pasted moves like '1. e4 e5 2. Nf3 ...' (comments allowed)."""
    import io
    game = chess.pgn.read_game(io.StringIO(text))
    if game is None or game.next() is None:
        raise ValueError("no moves found")
    if game.errors:
        raise ValueError(str(game.errors[0]))
    return game


# ----------------------------------------------------------------------------- player history
def normalise_name(name, source):
    """Key used to recognise the same player across files.
    OTB names come as 'Carlsen, Magnus' / 'Carlsen,M' / 'Magnus Carlsen' in different files, so we
    lower-case, drop punctuation and sort the name tokens. Lichess usernames are already unique."""
    if not name:
        return None
    if source == "lichess":
        return "lichess:" + name.strip().lower()
    toks = re.sub(r"[^a-z ]", " ", name.lower().replace(",", " ")).split()
    return "otb:" + " ".join(sorted(toks)) if toks else None


class PlayerHistory:
    """Running (strictly past) statistics per player. Call `features(...)` BEFORE `update(...)`
    for each game in chronological order, so that no information from the current game or
    the future leaks into its features."""
    SHRINK = 5  # pseudo-games pulled towards a 50% score (Bayesian shrinkage for small samples)

    def __init__(self):
        self.p = {}      # key -> dict
        self.h2h = {}    # (keyA, keyB) sorted -> [games, score_for_A]

    def _get(self, key):
        if key not in self.p:
            self.p[key] = {"n": 0, "score": 0.0, "draws": 0, "as_w_n": 0, "as_w_score": 0.0,
                           "as_b_n": 0, "as_b_score": 0.0, "first_elo": None, "last_elo": None,
                           "eco": {}, "last_date": None}
        return self.p[key]

    def features(self, wkey, bkey, eco, date):
        nan = float("nan")
        out = {}
        for pre, key, colour in (("w", wkey, "w"), ("b", bkey, "b")):
            s = self.p.get(key)
            if s is None or s["n"] == 0:
                out.update({f"{pre}_hist_games": 0, f"{pre}_hist_score": nan, f"{pre}_hist_colour_score": nan,
                            f"{pre}_hist_draw_rate": nan, f"{pre}_rating_trend": nan,
                            f"{pre}_eco_familiarity": 0, f"{pre}_days_since_last": nan})
                continue
            k = self.SHRINK
            n = s["n"]
            cn, cs = (s["as_w_n"], s["as_w_score"]) if colour == "w" else (s["as_b_n"], s["as_b_score"])
            out[f"{pre}_hist_games"] = n
            out[f"{pre}_hist_score"] = (s["score"] + 0.5 * k) / (n + k)
            out[f"{pre}_hist_colour_score"] = (cs + 0.5 * k) / (cn + k) if cn else nan
            out[f"{pre}_hist_draw_rate"] = s["draws"] / n
            out[f"{pre}_rating_trend"] = (s["last_elo"] - s["first_elo"]) if s["first_elo"] is not None else nan
            fam = (eco or "?")[:1]
            out[f"{pre}_eco_familiarity"] = s["eco"].get(fam, 0) / n
            out[f"{pre}_days_since_last"] = (date - s["last_date"]).days if (s["last_date"] is not None and date is not None) else nan
        out["hist_games_diff"] = out["w_hist_games"] - out["b_hist_games"]
        out["hist_score_diff"] = (out["w_hist_score"] - out["b_hist_score"]) if not (
            math.isnan(out["w_hist_score"]) or math.isnan(out["b_hist_score"])) else nan
        pair = tuple(sorted((str(wkey), str(bkey))))
        hh = self.h2h.get(pair)
        if hh:
            out["h2h_games"] = hh[0]
            # score from White's point of view
            sA = hh[1] if pair[0] == str(wkey) else hh[0] - hh[1]
            out["h2h_score"] = (sA + 0.5 * 2) / (hh[0] + 2)
        else:
            out["h2h_games"], out["h2h_score"] = 0, nan
        return out

    def update(self, wkey, bkey, result, w_elo, b_elo, eco, date):
        score_w = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(result)
        if score_w is None:
            return
        fam = (eco or "?")[:1]
        for key, elo, sc, colour in ((wkey, w_elo, score_w, "w"), (bkey, b_elo, 1 - score_w, "b")):
            if key is None:
                continue
            s = self._get(key)
            s["n"] += 1; s["score"] += sc; s["draws"] += int(score_w == 0.5)
            if colour == "w": s["as_w_n"] += 1; s["as_w_score"] += sc
            else:             s["as_b_n"] += 1; s["as_b_score"] += sc
            if s["first_elo"] is None: s["first_elo"] = elo
            s["last_elo"] = elo
            s["eco"][fam] = s["eco"].get(fam, 0) + 1
            s["last_date"] = date
        pair = tuple(sorted((str(wkey), str(bkey))))
        hh = self.h2h.setdefault(pair, [0, 0.0])
        hh[0] += 1
        hh[1] += score_w if pair[0] == str(wkey) else 1 - score_w

    def snapshot(self):
        """Final per-player table (used by the app to look players up)."""
        rows = []
        for key, s in self.p.items():
            if s["n"] == 0:
                continue
            rows.append({"player_key": key, "games": s["n"], "score": s["score"] / s["n"],
                         "draw_rate": s["draws"] / s["n"], "first_elo": s["first_elo"], "last_elo": s["last_elo"],
                         "score_as_white": (s["as_w_score"] / s["as_w_n"]) if s["as_w_n"] else None,
                         "score_as_black": (s["as_b_score"] / s["as_b_n"]) if s["as_b_n"] else None,
                         "last_date": s["last_date"]})
        return rows
