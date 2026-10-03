#!/usr/bin/env python3
"""Gráficas del README: comparación de modelos y estudio de cuantización (PNG en assets/).

    python3 charts/make_charts.py
"""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = pathlib.Path(__file__).resolve().parent.parent / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# Tema oscuro mate; colores validados con validate_palette.js --mode dark (oro, verde, azul)
SURFACE, INK, INK2, GRID = "#1c1c1b", "#ffffff", "#c3c2b7", "#383835"
GOLD, GREEN, BLUE = "#c98500", "#199e70", "#3987e5"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def benchmarks():
    """Barras agrupadas por métrica: este modelo frente al base y a DeepSeek V4 Pro (todo en %)."""
    metrics = ["F1\n(191 test contracts)", "Precision\n(191 test contracts)", "Recall\n(191 test contracts)",
               "F1\n(37 real 2026 exploits)"]
    models = [  # (nombre, color, valores en %)
        ("This model · 27B (Q4_K_Mix, 18.7 GB)", GOLD, [54, 55, 53, 49]),
        ("Qwen3.8-27B base · 27B (Unsloth)", GREEN, [44, 37, 53, 37]),
        ("DeepSeek V4 Pro · 1.6T MoE, 49B active", BLUE, [45, 46, 43, 44]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    n, w, gap = len(models), 0.26, 0.02
    for i, (name, color, vals) in enumerate(models):
        xs = [m + (i - (n - 1) / 2) * (w + gap) for m in range(len(metrics))]
        bars = ax.bar(xs, vals, w, color=color, label=name, zorder=3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v}", ha="center", va="bottom",
                    fontsize=10, color=INK, fontweight="bold" if i == 0 else "normal")
    ax.set_xticks(range(len(metrics)), metrics)
    ax.set_ylim(0, 70)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.09), ncol=3, frameon=False, fontsize=10, handlelength=1.2)
    fig.suptitle("Smart-contract vulnerability detection", x=0.065, y=0.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.065, -0.02, "Location-based automatic metrics, no thinking. Base model measured as UD-Q4_K_M; "
             "DeepSeek V4 Pro via API. Higher is better.", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "benchmarks.png", bbox_inches="tight")
    plt.close(fig)


def quantization():
    """Tamaño (GB) frente a KLD media vs BF16 (escala log): menos es más fiel."""
    pts = [  # (etiqueta, GB, KLD media)
        ("C  generic mix", 14.5, 0.0570),
        ("A  Q4_K_M", 16.8, 0.0263),
        ("B  Q4_K_M-imat", 16.8, 0.0179),
        ("G  custom mix v1", 18.1, 0.0126),
        ("I  Q4_K_Mix (recommended)", 18.7, 0.0102),
        ("D  Q5_K_M-imat", 19.5, 0.0072),
        ("E  Q6_K + imatrix", 22.4, 0.0022),
        ("F  Q8_0", 29.0, 0.0007),
    ]
    offsets = {"A": (10, 4), "B": (-12, 0), "C": (10, 2), "G": (-12, -22), "I": (12, 10), "D": (10, -8), "E": (10, 2), "F": (-10, 10)}
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    for label, gb, kld in pts:
        rec = label.startswith("I")
        ax.scatter(gb, kld, s=150 if rec else 70, color=GOLD, edgecolor=SURFACE, linewidth=2, zorder=4)
        dx, dy = offsets[label[0]]
        ax.annotate(f"{label}\n{gb} GB · KLD {kld:.4f}", (gb, kld), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", fontsize=9,
                    color=INK, fontweight="bold" if rec else "normal")
    ax.axvspan(14, 19.0, color=GOLD, alpha=0.08, zorder=0)
    ax.text(14.15, 0.00052, "Q4-size budget for a 32 GB V100 with long context", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.set_xlim(14, 31)
    ax.set_ylim(0.0004, 0.1)
    ax.set_xlabel("File size (GB)")
    ax.set_ylabel("Mean KLD vs BF16 (log scale, lower = more faithful)")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(color=GRID, linewidth=0.8, zorder=0, which="major")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", length=0)
    fig.suptitle("Quantization: size vs fidelity", x=0.075, y=1.0, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.075, -0.03, "Mean KL divergence against the BF16 model over six domains (Solidity, Foundry, "
             "blockchain, Python, other code, Spanish).", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(OUT / "quantization.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    benchmarks()
    quantization()
    print("OK:", *sorted(p.name for p in OUT.glob("*.png")))
