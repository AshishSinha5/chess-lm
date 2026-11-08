"""Make all figures from the CSV files in results/ (PLAN, Logging). Output: results/figures/*.png.
  python scripts/make_figures.py
Each figure reads only CSV files, so it needs no GPU and no checkpoint. A figure whose CSV is missing is skipped."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"

# Palette: first 3 categorical slots (they pass the colour-blind checks for all pairs), muted ink for controls.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PIECES = ["P", "N", "B", "R", "Q", "K"]
PIECE_NAMES = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
})


def read(name):
    path = RES / name
    return list(csv.DictReader(open(path))) if path.exists() else None


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print(f"-> {OUT / name}")


def end_label(ax, x, y, text, color, dy=0):
    """Direct label at the end of a line, in text ink with a colour mark, so identity is not colour alone.
    dy (points) moves the text up or down when two lines end close together."""
    ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", va="center", color=INK2, fontsize=9)
    ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)


def probe_by_layer(model="size_m"):
    rows, ctrl = read("probe/probe_by_layer.csv"), read("probe/controls.csv")
    if rows is None or ctrl is None:
        return
    def series(sel):
        pts = sorted((int(r["layer"]), float(r["bal_acc"])) for r in sel)
        return [p[0] for p in pts], [p[1] for p in pts]
    lin = series(r for r in rows if r["model"] == model and r["target"] == "absolute" and r["probe"] == "linear")
    mlp = series(r for r in rows if r["model"] == model and r["target"] == "absolute" and r["probe"] == "mlp")
    rnd = series(r for r in ctrl if r["model"] == model and r["control"] == "random_init")
    ref = {r["control"]: float(r["bal_acc"]) for r in ctrl if r["model"] == model and r["layer"] == ""}
    shuf = max(float(r["bal_acc"]) for r in ctrl if r["model"] == model and r["control"] == "shuffled_labels")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (x, y), color, name, dy in [(lin, BLUE, "M, linear probe", -9), (mlp, ORANGE, "M, MLP probe", 7),
                                    (rnd, AQUA, "Random-init M", 0)]:
        ax.plot(x, y, color=color, label=name, marker="o", markersize=4)
        end_label(ax, x[-1], y[-1], name, color, dy)
    for val, name, dy in [(ref["move_history"], "Move history", 0.012), (shuf, "Shuffled labels (best layer)", 0.012),
                          (ref["majority_class"], "Majority class", -0.04)]:
        ax.axhline(val, color=MUTED, linestyle="--", linewidth=1)
        ax.text(0.05, val + dy, f"{name} {val:.2f}", color=INK2, fontsize=8)
    ax.set(xlabel="Layer (0 = embedding, 1-8 = block output)", ylabel="Balanced accuracy, non-empty squares",
           ylim=(0, 1.02), xlim=(-0.2, 10.2), xticks=range(9))
    ax.set_title("The board is readable from M after block 1")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.12), fontsize=8)
    save(fig, "probe_by_layer.png")


def probe_by_ply(model="size_m"):
    rows = read("probe/probe_by_ply.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for src, color, name in [("linear", BLUE, "M, linear probe"), ("mlp", ORANGE, "M, MLP probe"), ("move_history", AQUA, "Move history")]:
        sel = [r for r in rows if r["model"] == model and r["source"] == src]
        if not sel:
            continue
        x = list(range(len(sel)))
        y = [100 * float(r["board_match"]) for r in sel]
        ax.plot(x, y, color=color, label=name, marker="o", markersize=4)
        end_label(ax, x[-1], y[-1], name, color)
        labels = [f"{r['ply_from']}-{r['ply_to']}" if int(r["ply_to"]) < 200 else f"{r['ply_from']}+" for r in sel]
    ax.set_xticks(range(len(labels)), labels)
    ax.set(xlabel="Ply (half-moves played)", ylabel="Whole board right (%)", ylim=(0, 102), xlim=(-0.2, len(labels) + 1.3))
    ax.set_title("Late in the game, M tracks the board better than move history")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "probe_by_ply.png")


def intervention(model="size_m", rank=32):
    rows = read("probe/intervention.csv")
    if rows is None:
        return
    base = next(float(r["ce"]) for r in rows if r["model"] == model and r["kind"] == "none")
    vals = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["kind"] in ("probe", "random", "random_matched") and int(r["rank"]) == rank:
            vals[(r["kind"], int(r["layer"]))].append(float(r["ce"]) - base)
    layers = sorted({k[1] for k in vals})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.26
    for i, (kind, color, name) in enumerate([("probe", BLUE, f"Remove {rank} board directions"),
                                             ("random", ORANGE, f"Remove {rank} random directions"),
                                             ("random_matched", AQUA, f"Remove {rank} random directions, same variance")]):
        mean = [sum(vals[(kind, l)]) / len(vals[(kind, l)]) for l in layers]
        lo = [m - min(vals[(kind, l)]) for m, l in zip(mean, layers)]
        hi = [max(vals[(kind, l)]) - m for m, l in zip(mean, layers)]
        xs = [j + (i - 1) * (w + 0.02) for j in range(len(layers))]
        ax.bar(xs, mean, w, color=color, label=name, yerr=[lo, hi], ecolor=INK2, capsize=3, edgecolor=SURFACE, linewidth=2)
        for x, m in zip(xs, mean):
            ax.text(x, m + 0.008, f"+{m:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(layers)), [f"Block {l}" for l in layers])
    ax.set(ylabel="Loss increase on the human move", xlabel="Block where the directions are removed")
    ax.set_title("Board directions hurt most, even against equal variance")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "intervention.png")


def piece_recall(model="size_m"):
    rows = read("probe/errors_by_piece.csv")
    if rows is None:
        return
    rec = {(r["source"], r["piece"]): float(r["recall"]) for r in rows if r["model"] == model}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.36
    for i, (src, color, name) in enumerate([("linear", BLUE, "M, linear probe"), ("move_history", AQUA, "Move history")]):
        y = [100 * (rec[(src, "w" + p)] + rec[(src, "b" + p)]) / 2 for p in PIECES]  # mean of white and black
        xs = [j + (i - 0.5) * (w + 0.02) for j in range(len(PIECES))]
        ax.bar(xs, y, w, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, y):
            ax.text(x, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(PIECES)), PIECE_NAMES)
    ax.set(ylabel="Recall (%), mean of white and black", ylim=(80, 102.5))
    ax.set_title("M is best on the pieces where move history is worst")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.grid(axis="x", visible=False)
    save(fig, "piece_recall.png")


def board_by_size():
    """Whole board right by ply, linear probe at the best layer, for XS, M and L, with the move-history control."""
    rows = read("probe/probe_by_ply.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    series = [("size_xs", BLUE, "XS (5.2M)"), ("size_m", ORANGE, "M (26.2M)"), ("size_l", AQUA, "L (50.4M)")]
    for model, color, name in series:
        sel = [r for r in rows if r["model"] == model and r["source"] == "linear"]
        if not sel:
            continue
        x, y = list(range(len(sel))), [100 * float(r["board_match"]) for r in sel]
        ax.plot(x, y, color=color, marker="o", markersize=4)
        end_label(ax, x[-1], y[-1], name, color)
        labels = [f"{r['ply_from']}-{r['ply_to']}" if int(r["ply_to"]) < 200 else f"{r['ply_from']}+" for r in sel]
    hist = [r for r in rows if r["model"] == "size_m" and r["source"] == "move_history"]
    ax.plot(range(len(hist)), [100 * float(r["board_match"]) for r in hist], color=MUTED, linestyle="--", marker="o", markersize=3)
    end_label(ax, len(hist) - 1, 100 * float(hist[-1]["board_match"]), "Move history", MUTED, dy=7)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(xlabel="Ply (half-moves played)", ylabel="Whole board right (%)", ylim=(0, 102), xlim=(-0.2, len(labels) + 1.6))
    ax.set_title("A bigger model reads more of the board, mostly late in the game")
    save(fig, "board_by_size.png")


SIZE_ORDER = ["xs", "s", "m", "l"]
SIZE_COLOR = {"xs": BLUE, "s": ORANGE, "m": AQUA, "l": "#8a5cd0"}
SIZE_NAME = {"xs": "XS (5.2M)", "s": "S (11.4M)", "m": "M (26.2M)", "l": "L (50.4M)"}


def loss_by_size():
    """Final loss against parameters at 1B tokens (both seeds), and at 1.6B tokens for M and L."""
    rows = read("runs.csv")
    if rows is None:
        return
    by = {r["run"]: r for r in rows}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, lo, hi = [], [], []
    for m in SIZE_ORDER:
        runs = [by[n] for n in (f"size_{m}", f"seed_{m}_s1") if n in by]
        v = [float(r["final_val_loss"]) for r in runs]
        x = float(runs[0]["params"]) / 1e6
        xs.append(x)
        lo.append(min(v)), hi.append(max(v))
        ax.plot([x, x], [min(v), max(v)], color=INK2, linewidth=3, solid_capstyle="round", zorder=3)
        ax.plot([x], [float(runs[0]["final_val_loss"])], "o", color=BLUE, markersize=7, zorder=4, markeredgecolor=SURFACE)
        ax.annotate(m.upper(), (x, max(v)), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax.plot(xs, [float(by[f"size_{m}"]["final_val_loss"]) for m in SIZE_ORDER], color=BLUE, label="1B tokens (a second seed moves each point by 0.002 to 0.006)")
    big = [(float(by[f"full_{m}"]["params"]) / 1e6, float(by[f"full_{m}"]["final_val_loss"])) for m in ("m", "l") if f"full_{m}" in by]
    if big:
        ax.plot(*zip(*big), "o--", color=ORANGE, markersize=7, markeredgecolor=SURFACE, label="1.6B tokens (M and L only)")
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50], ["5M", "10M", "20M", "50M"])
    ax.set(xlabel="Parameters", ylabel="Validation loss per move")
    ax.set_title("Loss falls with size, and the step from M to L is small")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "loss_by_size.png")


def token_sweep():
    """Loss against tokens seen, one line per size (results/token_sweep.csv)."""
    rows = read("token_sweep.csv")
    if rows is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for m in SIZE_ORDER:
        sel = sorted((float(r["tokens_seen"]) / 1e9, float(r["val_loss"])) for r in rows if r["model"] == m)
        if not sel:
            continue
        x, y = zip(*sel)
        ax.plot(x, y, "o-", color=SIZE_COLOR[m], markersize=5)
        end_label(ax, x[-1], y[-1], SIZE_NAME[m], SIZE_COLOR[m])
    ax.set_xscale("log")
    ax.set_xticks([0.156, 0.313, 0.625, 1.0], ["0.16B", "0.31B", "0.63B", "1B"])
    ax.set(xlabel="Tokens seen (points before 1B: checkpoints finished with a short decay)", ylabel="Validation loss per move", xlim=(0.13, 2.2))
    ax.set_title("Every size still gains from more tokens")
    save(fig, "token_sweep.png")


def elo_by_size():
    """Elo against Stockfish with its 95% interval (results/eval/elo.csv, rows with opponent 'all')."""
    rows = read("eval/elo.csv")
    runs = read("runs.csv")
    if rows is None or runs is None:
        return
    params = {r["model"]: float(r["params"]) / 1e6 for r in runs if r["run"].startswith("size_")}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    pts = []
    for r in rows:
        if r["opponent"] != "all":
            continue
        m = r["model"].replace("size_", "")
        pts.append((params[m], float(r["elo"]), float(r["ci_low"]), float(r["ci_high"]), m))
    pts.sort()
    x, y = [q[0] for q in pts], [q[1] for q in pts]
    ax.errorbar(x, y, yerr=[[q[1] - q[2] for q in pts], [q[3] - q[1] for q in pts]], fmt="o-", color=BLUE, ecolor=INK2,
                capsize=4, markersize=7, markeredgecolor=SURFACE, label="Lenient rule (published): an illegal move is skipped")
    for q in pts:
        ax.annotate(f"{q[4].upper()}  {q[1]:.0f}", (q[0], q[1]), xytext=(8, 6), textcoords="offset points", fontsize=9, color=INK2)
    sv = read("eval/strict_vs_lenient.csv")
    if sv:
        sv = sorted(sv, key=lambda r: params[r["model"].replace("size_", "")])
        sx = [params[r["model"].replace("size_", "")] for r in sv]
        sy = [float(r["elo_strict"]) for r in sv]
        ax.errorbar(sx, sy, yerr=[[float(r["elo_strict"]) - float(r["strict_ci_low"]) for r in sv],
                                  [float(r["strict_ci_high"]) - float(r["elo_strict"]) for r in sv]],
                    fmt="o--", color=ORANGE, ecolor=INK2, capsize=4, markersize=7, markeredgecolor=SURFACE,
                    label="Strict rule (rerun): an illegal move loses")
        for xx, yy, r in zip(sx, sy, sv):
            ax.annotate(f"{yy:.0f}", (xx, yy), xytext=(8, -14), textcoords="offset points", fontsize=9, color=INK2)
        ax.legend(loc="lower right", fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50], ["5M", "10M", "20M", "50M"])
    ax.set(xlabel="Parameters", ylabel="Elo on the Stockfish scale")
    ax.set_title("Strength rises with size, and illegal moves cost Elo")
    save(fig, "elo_by_size.png")


def scaling_law():
    """Left: loss against tokens with the fitted law, and the 2 held-out runs. Right: Elo against loss
    (results/scaling_points.csv, scaling_fit.csv, eval/strict_vs_lenient.csv; made by scripts/fit_scaling.py)."""
    pts, fit, sv, runs = read("scaling_points.csv"), read("scaling_fit.csv"), read("eval/strict_vs_lenient.csv"), read("runs.csv")
    if not (pts and fit and sv and runs):
        return
    f = {r["name"]: float(r["value"]) for r in fit}
    size_of = {int(float(r["params"])): r["model"] for r in runs if r["run"].startswith("size_")}
    law = lambda n, d: f["E"] + f["A"] * (n / 1e6) ** -f["alpha"] + f["B"] * (d / 1e9) ** -f["beta"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = [0.1 * 1.12 ** i for i in range(0, 28)]
    for m in SIZE_ORDER:
        n = next(k for k, v in size_of.items() if v == m)
        ax.plot(grid, [law(n, g * 1e9) for g in grid], "-", color=SIZE_COLOR[m], linewidth=1.2, alpha=0.8)
        sel = [r for r in pts if int(r["params"]) == n and r["split"] == "fit"]
        ax.plot([float(r["tokens"]) / 1e9 for r in sel], [float(r["val_loss"]) for r in sel], "o", color=SIZE_COLOR[m], markersize=5)
        hold = [r for r in pts if int(r["params"]) == n and r["split"] == "held_out"]
        ax.plot([float(r["tokens"]) / 1e9 for r in hold], [float(r["val_loss"]) for r in hold], "D", color=SIZE_COLOR[m],
                markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.8)
        end_label(ax, grid[-1], law(n, grid[-1] * 1e9), SIZE_NAME[m], SIZE_COLOR[m])
    ax.set_xscale("log")
    ax.set_xticks([0.16, 0.31, 0.63, 1.0, 1.6], ["0.16B", "0.31B", "0.63B", "1B", "1.6B"])
    ax.set(xlabel="Tokens seen (dots: fitted runs; open diamonds: held out)", ylabel="Validation loss per move", xlim=(0.13, 2.9))
    ax.set_title(f"Loss law: E + A/N^{f['alpha']:.2f} + B/D^{f['beta']:.2f}")
    loss = {r["run"].replace("size_", ""): float(r["final_val_loss"]) for r in runs if r["run"].startswith("size_")}
    xs = [loss[r["model"].replace("size_", "")] for r in sv]
    xl = [min(xs) - 0.03, max(xs) + 0.03]
    for col, key, color, name in (("elo_lenient", "lenient_rerun", BLUE, "Lenient"), ("elo_strict", "strict", ORANGE, "Strict")):
        bx.plot(xs, [float(r[col]) for r in sv], "o", color=color, markersize=7, markeredgecolor=SURFACE)
        line = [f[f"elo_{key}_slope_per_loss"] * x + f[f"elo_{key}_intercept"] for x in xl]
        bx.plot(xl, line, "-", color=color, linewidth=1.2)
        end_label(bx, xl[0], line[0], f"{name}: {-f[f'elo_{key}_slope_per_loss'] / 10:.0f} Elo per 0.1 loss", color, dy=0)
    for r, x in zip(sv, xs):
        bx.annotate(r["model"].replace("size_", "").upper(), (x, float(r["elo_lenient"])), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK2)
    bx.invert_xaxis()
    bx.set(xlabel="Validation loss per move (1B tokens)", ylabel="Elo on the Stockfish scale")
    bx.set_title("Elo is close to linear in loss")
    save(fig, "scaling_law.png")


GLYPH = {"P": "\u265f", "N": "\u265e", "B": "\u265d", "R": "\u265c", "Q": "\u265b", "K": "\u265a"}


def example_boards(model="size_m"):
    """True board above, probe board below, for 3 positions of one test game (results/probe/example_game.csv)."""
    rows = read("probe/example_game.csv")
    moves = read("probe/example_game_moves.csv")
    if rows is None or moves is None:
        return
    import matplotlib.patheffects as pe

    rows = [r for r in rows if r["model"] == model]
    plies = sorted({int(r["ply"]) for r in rows})
    show = [plies[0], plies[len(plies) // 2], plies[-1]] if len(plies) >= 3 else plies
    by = {(int(r["ply"]), r["square"]): r for r in rows}
    fig, axes = plt.subplots(2, len(show), figsize=(3.4 * len(show), 7.2))
    axes = axes.reshape(2, len(show))
    for j, ply in enumerate(show):
        wrong = sum(by[(ply, f + str(r))]["true"] != by[(ply, f + str(r))]["probe"] for f in "abcdefgh" for r in range(1, 9))
        for i, key in enumerate(["true", "probe"]):
            ax = axes[i, j]
            ax.set_xlim(0, 8), ax.set_ylim(0, 8), ax.set_aspect("equal"), ax.axis("off")
            for fi, f in enumerate("abcdefgh"):
                for r in range(1, 9):
                    cell = by[(ply, f + str(r))]
                    bad = key == "probe" and cell["true"] != cell["probe"]
                    shade = "#e9e4d8" if (fi + r) % 2 == 1 else "#c9c2b0"
                    ax.add_patch(plt.Rectangle((fi, r - 1), 1, 1, color=ORANGE if bad else shade, alpha=0.75 if bad else 1))
                    name = cell[key]
                    if name != "empty":
                        white = name[0] == "w"
                        ax.text(fi + 0.5, r - 0.5, GLYPH[name[1]], ha="center", va="center", fontsize=21,
                                color="white" if white else "black",
                                path_effects=[pe.withStroke(linewidth=1.6, foreground="black" if white else "white")])
            if i == 0:
                side = "white" if ply % 2 == 0 else "black"
                ax.set_title(f"True board after move {ply} ({side} to move)", fontsize=10, loc="left")
            else:
                ax.set_title(f"Probe board: {wrong} wrong squares" if wrong else "Probe board: all 64 squares right",
                             fontsize=10, loc="left", color=INK2)
    fig.suptitle(f"One test game: true board and linear probe board ({model}, layer {rows[0]['layer']})", x=0.02, ha="left",
                 fontsize=12, fontweight="bold")
    save(fig, "example_boards.png")


if __name__ == "__main__":
    probe_by_layer()
    probe_by_ply()
    intervention()
    piece_recall()
    example_boards()
    board_by_size()
    loss_by_size()
    token_sweep()
    elo_by_size()
    scaling_law()
