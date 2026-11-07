"""Make one looping GIF per model: the model (white) plays the first moves against Stockfish (black).
python scripts/make_gifs.py [--sizes xs,s,m,l] [--moves 8] [--elo 1320] [--out docs/gifs]
The GIF has no controls. It plays on a loop. Moves are saved in games.json next to the GIFs.
The model plays its best legal move. Stockfish plays at UCI_Elo --elo, 0.05 s per move, so a rerun can differ.
Model code runs on the CPU, one thread. Run it under `nice` while a GPU job trains."""
import argparse
import json
import os
from pathlib import Path

import chess
import chess.engine
import torch
from PIL import Image, ImageDraw, ImageFont

from chesslm.eval.play import choose_move, load_model

STOCKFISH = os.environ.get("STOCKFISH", "/workspace/bin/stockfish")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GLYPH = {"k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟"}
LIGHT, DARK, LAST = (240, 217, 181), (181, 136, 99), (205, 210, 106)
SQ, PAD, CAP = 44, 8, 76
NAMES = {"xs": "XS (5.2M)", "s": "S (11.4M)", "m": "M (26.2M)", "l": "L (50.4M)"}
ELO = {"xs": 1182, "s": 1296, "m": 1410, "l": 1479}


def frame(board, last, title, line, sub):
    w = 8 * SQ + 2 * PAD
    img = Image.new("RGB", (w, w + CAP), (34, 34, 34))
    d = ImageDraw.Draw(img)
    big, small, bold = ImageFont.truetype(FONT, 34), ImageFont.truetype(FONT, 12), ImageFont.truetype(BOLD, 15)
    for sq in chess.SQUARES:
        c, r = chess.square_file(sq), 7 - chess.square_rank(sq)
        x, y = PAD + c * SQ, PAD + r * SQ
        col = LAST if sq in last else (LIGHT if (c + r) % 2 == 0 else DARK)
        d.rectangle([x, y, x + SQ - 1, y + SQ - 1], fill=col)
        p = board.piece_at(sq)
        if p:
            white = p.color == chess.WHITE
            d.text((x + SQ / 2, y + SQ / 2 + 1), GLYPH[p.symbol().lower()], font=big, anchor="mm",
                   fill=(255, 255, 255) if white else (0, 0, 0), stroke_width=1,
                   stroke_fill=(0, 0, 0) if white else (255, 255, 255))
    y0 = w + 6
    d.text((PAD, y0), title, font=bold, fill=(255, 255, 255))
    d.text((PAD, y0 + 22), sub, font=small, fill=(180, 180, 180))
    d.text((PAD, y0 + 40), line, font=small, fill=(230, 230, 230))
    return img


def play_game(model, engine, moves):
    board, hist, sans, plies = chess.Board(), [], [], []
    while not board.is_game_over() and len(hist) < 2 * moves:
        if board.turn == chess.WHITE:
            uci = choose_move(model, hist, board, device="cpu")
        else:
            uci = engine.play(board, chess.engine.Limit(time=0.05)).move.uci()
        move = chess.Move.from_uci(uci)
        sans.append(board.san(move))
        board.push(move)
        hist.append(uci)
    return hist, sans


def movetext(sans, n):
    out = []
    for i in range(n):
        out.append((f"{i // 2 + 1}. " if i % 2 == 0 else "") + sans[i])
    text = " ".join(out)
    return text if len(text) <= 46 else "\u2026 " + text[-43:]


def build(size, hist, sans, elo, path):
    title = f"Chess LM {NAMES[size]} vs Stockfish {elo}"
    sub = f"Model plays white. Its Elo on the Stockfish scale: {ELO[size]}."
    board, frames, last = chess.Board(), [], set()
    frames.append((frame(board, last, title, "Start position", sub), 900))
    for i, u in enumerate(hist):
        m = chess.Move.from_uci(u)
        board.push(m)
        last = {m.from_square, m.to_square}
        frames.append((frame(board, last, title, movetext(sans, i + 1), sub), 750))
    end = board.result() if board.is_game_over() else ""
    imgs = [f for f, _ in frames]
    durs = [t for _, t in frames]
    durs[-1] = 3000
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=durs, loop=0, optimize=True, disposal=2)
    return end


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="xs,s,m,l")
    ap.add_argument("--moves", type=int, default=8, help="full moves per game")
    ap.add_argument("--elo", type=int, default=1320, help="Stockfish UCI_Elo (minimum 1320)")
    ap.add_argument("--out", default="docs/gifs")
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": a.elo})
    games = {}
    try:
        for size in a.sizes.split(","):
            model = load_model(f"size_{size}", device="cpu")
            hist, sans = play_game(model, engine, a.moves)
            end = build(size, hist, sans, a.elo, out / f"play_{size}.gif")
            games[size] = {"stockfish_elo": a.elo, "moves": hist, "san": sans, "result": end}
            print(size, len(hist), "plies", end or "(unfinished)", movetext(sans, len(sans)))
    finally:
        engine.quit()
    old = json.load(open(out / "games.json")) if (out / "games.json").exists() else {}
    old.update(games)
    json.dump(old, open(out / "games.json", "w"), indent=1)
