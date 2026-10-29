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
        if r["model"] == model and r["kind"] in ("probe", "random") and int(r["rank"]) == rank:
            vals[(r["kind"], int(r["layer"]))].append(float(r["ce"]) - base)
    layers = sorted({k[1] for k in vals})
    fig, ax = plt.subplots(figsize=(7, 4.2))
    w = 0.36
    for i, (kind, color, name) in enumerate([("probe", BLUE, f"Remove {rank} board directions"),
                                             ("random", ORANGE, f"Remove {rank} random directions")]):
        mean = [sum(vals[(kind, l)]) / len(vals[(kind, l)]) for l in layers]
        lo = [m - min(vals[(kind, l)]) for m, l in zip(mean, layers)]
        hi = [max(vals[(kind, l)]) - m for m, l in zip(mean, layers)]
        xs = [j + (i - 0.5) * (w + 0.02) for j in range(len(layers))]
        ax.bar(xs, mean, w, color=color, label=name, yerr=[lo, hi], ecolor=INK2, capsize=3, edgecolor=SURFACE, linewidth=2)
        for x, m in zip(xs, mean):
            ax.text(x, m + 0.008, f"+{m:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(layers)), [f"Block {l}" for l in layers])
    ax.set(ylabel="Loss increase on the human move", xlabel="Block where the directions are removed")
    ax.set_title("Removing the board directions hurts M 6 to 8 times more")
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


if __name__ == "__main__":
    probe_by_layer()
    probe_by_ply()
    intervention()
    piece_recall()
