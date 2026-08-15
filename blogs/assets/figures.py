"""Reproducible figure generator for the Inference Engineering blogs.

Every numeric or schematic figure in the series is produced here so the charts stay
consistent and can be regenerated from scratch. Output is SVG (vector, tiny, renders on
GitHub markdown and in the Astro portfolio with no JavaScript).

The series obeys one color code, and it carries meaning:

    COMPUTE  #C8421A  warm terracotta   arithmetic, FLOPs, compute-bound
    MEMORY   #1F7A8C  cool teal         bytes, bandwidth, memory-bound

Warm is always compute, cool is always memory. Never use them decoratively.

Two style modes:

    house_style()   clean axes, for anything with real numbers
    sketch_style()  hand-drawn wobble, for conceptual schematics

Run:
    uv run python blogs/assets/figures.py
    uv run python blogs/assets/figures.py --blog 01   # only post 01's figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Rectangle

# ----------------------------------------------------------------------------
# Palette + house style (shared across every figure)
# ----------------------------------------------------------------------------
INK = "#0A0A0A"
COMPUTE = "#C8421A"
MEMORY = "#1F7A8C"
MUTED = "#525252"
DIVIDER = "#E5E5E5"
CANVAS = "#FAFAFA"
COMPUTE_SOFT = "#E8A48F"
MEMORY_SOFT = "#9CC5CE"

FONTS = ["Geist", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]

BLOGS_DIR = Path(__file__).resolve().parent.parent


def house_style() -> None:
    """Clean axes with a light grid, for figures carrying real numbers."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": CANVAS,
            "axes.facecolor": CANVAS,
            "savefig.facecolor": CANVAS,
            "font.family": "sans-serif",
            "font.sans-serif": FONTS,
            "font.size": 12,
            "axes.edgecolor": DIVIDER,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": DIVIDER,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "svg.fonttype": "none",  # keep text as text in the SVG
            "axes.spines.top": False,
            "axes.spines.right": False,
            "path.sketch": None,
        }
    )


def sketch_style() -> None:
    """Hand-drawn wobble for conceptual schematics (the 'excalidraw' look).

    No grid, no spines, no ticks: these figures are diagrams, not plots.
    """
    house_style()
    plt.rcParams.update(
        {
            # (scale, length, randomness): how far the line wanders, how often it turns
            "path.sketch": (1.4, 110, 2),
            "path.effects": [],
            "axes.grid": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "lines.linewidth": 2.2,
            "patch.linewidth": 2.0,
        }
    )


def save(fig: plt.Figure, blog: str, name: str) -> None:
    out_dir = BLOGS_DIR / blog / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.svg"
    fig.savefig(path, format="svg", bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  wrote {path.relative_to(BLOGS_DIR)}")


def _blank(ax: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Turn an Axes into a bare drawing canvas for schematic figures."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ----------------------------------------------------------------------------
# Shared reference numbers: NVIDIA H100 SXM + Llama-3-70B at 16-bit
# ----------------------------------------------------------------------------
H100_FLOPS = 989e12  # dense FP16/BF16 tensor-core peak, FLOP/s
H100_BW = 3.35e12  # HBM3 bandwidth, bytes/s
RIDGE = H100_FLOPS / H100_BW  # ~295 ops/byte

WEIGHT_BYTES = 140e9  # 70B params at 2 bytes
KV_BYTES_PER_TOKEN = 2 * 80 * 8 * 128 * 2  # K and V, 80 layers, 8 KV heads, dim 128, fp16

BLOG01 = "inference-01-memory-wall"


# ----------------------------------------------------------------------------
# Blog 01 — The memory wall
# ----------------------------------------------------------------------------
def fig_where_time_goes() -> None:
    """The 30 ms token budget: a sliver of arithmetic, the rest waiting on memory."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    _blank(ax, (-1.5, 33), (-1.4, 2.6))

    total, math_ms = 30.0, 0.1

    ax.add_patch(
        Rectangle((0, 0), total, 1.2, facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.55)
    )
    ax.add_patch(
        Rectangle((0, 0), math_ms, 1.2, facecolor=COMPUTE, edgecolor=COMPUTE)
    )

    ax.text(
        total / 2, 0.6, "waiting for bytes to arrive from memory",
        ha="center", va="center", color=INK, fontsize=13,
    )
    ax.annotate(
        "0.1 ms of arithmetic",
        xy=(math_ms, 1.2), xytext=(3.2, 2.15),
        color=COMPUTE, fontsize=12, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.8),
    )
    ax.text(-0.1, -0.55, "0 ms", ha="left", va="top", color=MUTED, fontsize=11)
    ax.text(total, -0.55, "30 ms", ha="right", va="top", color=MUTED, fontsize=11)
    ax.text(
        total / 2, -1.05,
        "one output token  ·  less than 1% of it is math",
        ha="center", va="top", color=MUTED, fontsize=11.5,
    )
    ax.set_title("Where the 30 milliseconds go", pad=16)
    save(fig, BLOG01, "fig-where-time-goes")


def fig_compute_vs_bandwidth() -> None:
    """A decade of top-tier server GPUs: compute grew 80x, bandwidth 17x."""
    house_style()
    years = np.linspace(2012, 2022, 200)
    t = (years - 2012) / 10
    compute = 80.0**t
    bandwidth = 17.0**t

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.fill_between(years, bandwidth, compute, color=COMPUTE, alpha=0.10, lw=0)
    ax.plot(years, compute, lw=2.6, color=COMPUTE, label="compute throughput (FLOP/s)")
    ax.plot(years, bandwidth, lw=2.6, color=MEMORY, label="memory bandwidth (bytes/s)")

    ax.annotate(
        "80x", xy=(2022, 80), xytext=(2020.3, 104),
        color=COMPUTE, fontweight="bold", fontsize=13,
    )
    ax.annotate(
        "17x", xy=(2022, 17), xytext=(2020.3, 7.6),
        color=MEMORY, fontweight="bold", fontsize=13,
    )
    ax.annotate(
        "4.7x divergence\nin one decade",
        xy=(2019.4, 32), xytext=(2014.4, 26),
        color=INK, fontsize=12,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
    )

    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "50x", "100x"])
    ax.set_ylim(0.9, 150)
    ax.set_xlim(2012, 2022.6)
    ax.set_xlabel("year")
    ax.set_ylabel("growth since 2012  (log scale)")
    ax.set_title("The memory wall: two specs on different physics")
    ax.legend(frameon=False, loc="upper left")
    save(fig, BLOG01, "fig-compute-vs-bandwidth")


def fig_weights_through_bus() -> None:
    """Schematic: 140 GB of weights squeezed through the memory bus every forward pass."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    _blank(ax, (0, 10), (0, 6.4))

    # HBM: a stack of DRAM dies holding the weights
    ax.add_patch(Rectangle((0.3, 0.7), 2.3, 4.6, facecolor="none", edgecolor=MEMORY))
    for i in range(6):
        y = 0.95 + i * 0.73
        ax.add_patch(
            Rectangle((0.5, y), 1.9, 0.56, facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.85)
        )
    ax.text(1.45, 5.6, "HBM", ha="center", color=MEMORY, fontsize=13, fontweight="bold")
    ax.text(1.45, 0.28, "140 GB of weights", ha="center", color=MUTED, fontsize=11)

    # The bus: deliberately drawn as a narrow neck between two large blocks
    ax.add_patch(
        Rectangle((2.6, 2.6), 4.0, 0.85, facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.45)
    )
    ax.add_patch(FancyArrowPatch((2.9, 3.02), (6.3, 3.02), arrowstyle="->",
                                 mutation_scale=22, color=MEMORY, lw=2.4))
    ax.text(4.6, 3.75, "memory bus  ·  3.35 TB/s", ha="center", color=MEMORY, fontsize=11.5)

    # Compute die: a dense grid of arithmetic units
    ax.add_patch(Rectangle((6.6, 0.7), 3.1, 4.6, facecolor="none", edgecolor=COMPUTE))
    for r in range(7):
        for c in range(9):
            ax.add_patch(
                Rectangle((6.82 + c * 0.31, 0.95 + r * 0.62), 0.2, 0.42,
                          facecolor=COMPUTE_SOFT, edgecolor="none")
            )
    ax.text(8.15, 5.6, "arithmetic units", ha="center", color=COMPUTE,
            fontsize=13, fontweight="bold")
    ax.text(8.15, 0.28, "989 TFLOP/s", ha="center", color=MUTED, fontsize=11)

    ax.set_title("Every forward pass drags all 140 GB through the same neck", pad=14)
    save(fig, BLOG01, "fig-weights-through-bus")


def fig_cost_per_token() -> None:
    """The same 140 GB transfer, divided by the tokens that share it."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))

    labels = ["Prefill\n2,000-token prompt", "Decode\none token at a time"]
    values_mb = [140_000 / 2000, 140_000]
    colors = [COMPUTE, MEMORY]

    bars = ax.barh(labels, values_mb, color=colors, height=0.55, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(10, 1.2e6)
    ax.set_xlabel("weight bytes paid per token of output  (log scale)")

    for bar, label in zip(bars, ["70 MB per token", "140 GB per token"]):
        ax.text(
            bar.get_width() * 1.35, bar.get_y() + bar.get_height() / 2, label,
            va="center", ha="left", color=INK, fontsize=12.5, fontweight="bold",
        )

    ax.annotate(
        "", xy=(140_000, 0.5), xytext=(70, 0.5),
        arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.4),
    )
    ax.text(3000, 0.545, "2,000x", ha="center", va="bottom", color=INK,
            fontsize=13, fontweight="bold")

    ax.set_title("Same transfer, different number of tokens to share it")
    ax.grid(axis="y", visible=False)
    save(fig, BLOG01, "fig-cost-per-token")


def fig_roofline() -> None:
    """The roofline for an H100, with prefill and decode plotted on it."""
    house_style()
    fig, ax = plt.subplots(figsize=(9, 5.6))

    ai = np.logspace(-1, 4, 400)
    roof = np.minimum(H100_BW * ai, H100_FLOPS)

    ax.fill_between(ai, 1e11, roof, where=ai <= RIDGE, color=MEMORY, alpha=0.10, lw=0)
    ax.fill_between(ai, 1e11, roof, where=ai >= RIDGE, color=COMPUTE, alpha=0.10, lw=0)

    ax.plot(ai[ai <= RIDGE], H100_BW * ai[ai <= RIDGE], lw=2.8, color=MEMORY)
    ax.plot(ai[ai >= RIDGE], np.full((ai >= RIDGE).sum(), H100_FLOPS), lw=2.8, color=COMPUTE)

    ax.axvline(RIDGE, ls=":", lw=1.4, color=MUTED)
    ax.plot([RIDGE], [H100_FLOPS], "o", ms=9, mfc=CANVAS, mec=INK, mew=2)
    ax.annotate(
        f"ridge  ·  {RIDGE:.0f} ops/byte",
        xy=(RIDGE, H100_FLOPS), xytext=(20, 1.5e15),
        color=INK, fontsize=12,
        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2),
    )

    ax.plot([1], [H100_BW], "o", ms=11, color=MEMORY)
    ax.annotate(
        "DECODE\n1 op/byte\nbus saturated, math idle",
        xy=(1, H100_BW), xytext=(0.14, 1.4e13),
        color=MEMORY, fontsize=11.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=MEMORY, lw=1.4),
    )
    ax.plot([2000], [H100_FLOPS], "o", ms=11, color=COMPUTE)
    ax.annotate(
        "PREFILL\n2,000 ops/byte\nmath saturated, bus idle",
        xy=(2000, H100_FLOPS), xytext=(560, 4.5e13),
        color=COMPUTE, fontsize=11.5, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.4),
    )

    ax.text(2.2, 1.6e11, "MEMORY-BOUND", color=MEMORY, fontsize=11, fontweight="bold")
    ax.text(900, 1.6e11, "COMPUTE-BOUND", color=COMPUTE, fontsize=11, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e11, 3e15)
    ax.set_xlabel("arithmetic intensity  (ops per byte)")
    ax.set_ylabel("achievable performance  (FLOP/s)")
    ax.set_title("The roofline: an H100 with both phases plotted")
    save(fig, BLOG01, "fig-roofline")


def fig_chip_utilization() -> None:
    """What each phase actually keeps busy on the chip."""
    house_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharex=True)
    fig.subplots_adjust(wspace=0.12)

    panels = [
        ("Prefill", [95, 25], "compute-bound", COMPUTE),
        ("Decode", [1, 100], "memory-bound", MEMORY),
    ]
    for idx, (ax, (phase, vals, verdict, verdict_color)) in enumerate(zip(axes, panels)):
        ax.barh(["arithmetic units", "HBM bandwidth"], vals,
                color=[COMPUTE, MEMORY], height=0.5, alpha=0.9)
        for y, v in enumerate(vals):
            ax.text(v + 3, y, f"{v}%", va="center", color=INK,
                    fontsize=12.5, fontweight="bold")
        if idx == 1:
            ax.set_yticklabels([])
        ax.set_xlim(0, 128)
        ax.set_xticks([0, 50, 100])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(f"{phase}  ·  {verdict}", color=verdict_color)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()

    fig.suptitle("Same chip, same model, opposite bottlenecks",
                 fontsize=14, fontweight="bold", color=INK, y=1.06)
    save(fig, BLOG01, "fig-chip-utilization")


def fig_kv_cache_growth() -> None:
    """KV cache size against context length, next to the model's own weight footprint."""
    house_style()
    tokens = np.logspace(2, np.log10(200_000), 300)
    cache_gb = tokens * KV_BYTES_PER_TOKEN / 1e9

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(tokens, cache_gb, lw=2.8, color=MEMORY, label="KV cache")
    ax.axhline(141, ls="--", lw=1.6, color=INK)
    ax.text(120, 158, "model weights  ·  141 GB", color=INK, fontsize=11.5)

    for t, note in [(1_000, "1K tokens\n0.33 GB"), (10_000, "10K\n3.3 GB"),
                    (100_000, "100K\n33 GB")]:
        gb = t * KV_BYTES_PER_TOKEN / 1e9
        ax.plot([t], [gb], "o", ms=8, color=MEMORY)
        ax.annotate(note, xy=(t, gb), xytext=(t * 1.15, gb * 0.34),
                    color=MUTED, fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(100, 200_000)
    ax.set_ylim(0.02, 400)
    ax.set_xlabel("conversation length  (tokens, log scale)")
    ax.set_ylabel("memory held in HBM  (GB, log scale)")
    ax.set_title("The KV cache grows linearly, and it never leaves HBM")
    save(fig, BLOG01, "fig-kv-cache-growth")


def fig_decode_ceiling() -> None:
    """Bandwidth divided by bytes per token, as the cache adds to the bill."""
    house_style()
    tokens = np.linspace(0, 128_000, 400)
    bytes_per_token = WEIGHT_BYTES + tokens * KV_BYTES_PER_TOKEN
    ceiling = H100_BW / bytes_per_token

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.fill_between(tokens / 1000, 0, ceiling, color=MEMORY, alpha=0.12, lw=0)
    ax.plot(tokens / 1000, ceiling, lw=2.8, color=MEMORY)

    for t in (0, 32_000, 128_000):
        rate = H100_BW / (WEIGHT_BYTES + t * KV_BYTES_PER_TOKEN)
        ax.plot([t / 1000], [rate], "o", ms=8, color=MEMORY)
        align = "right" if t == 128_000 else "left"
        dx = -4 if t == 128_000 else 4
        ax.annotate(f"{rate:.1f} tok/s", xy=(t / 1000, rate),
                    xytext=(t / 1000 + dx, rate + 1.1), ha=align,
                    color=INK, fontsize=11.5, fontweight="bold")

    ax.set_xlim(0, 132)
    ax.set_ylim(0, 28)
    ax.set_xlabel("conversation length  (thousands of tokens)")
    ax.set_ylabel("maximum decode rate  (tokens / second)")
    ax.set_title("A physical ceiling, and the KV cache lowers it")
    save(fig, BLOG01, "fig-decode-ceiling")


def fig_goodput() -> None:
    """Throughput counts every token. Goodput counts only the ones delivered in time."""
    sketch_style()
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    ttft_slo = 2.0
    scenarios = [
        ("Small batch  ·  8 requests", 8, (0.7, 1.5), 0.025, 320, 320),
        ("Huge batch  ·  64 requests", 10, (3.4, 4.6), 0.045, 1408, 0),
    ]

    for ax, (title, rows, ttft_range, tpot, throughput, goodput) in zip(axes, scenarios):
        _blank(ax, (-1.1, 8.4), (-1.6, rows + 0.4))
        # bounded rather than axvline so the SLO marker never crosses the caption
        ax.plot([ttft_slo, ttft_slo], [-0.15, rows], ls="--", lw=1.8, color=COMPUTE)
        ax.text(ttft_slo + 0.08, rows + 0.05, "TTFT SLO  ·  2 s",
                color=COMPUTE, fontsize=10.5, va="bottom")

        for i in range(rows):
            y = rows - 1 - i
            ttft = rng.uniform(*ttft_range)
            met = ttft <= ttft_slo
            tone = INK if met else DIVIDER
            ax.add_patch(
                Rectangle((0, y + 0.18), ttft, 0.3,
                          facecolor=MEMORY_SOFT if met else DIVIDER,
                          edgecolor=MEMORY if met else "#BDBDBD", alpha=0.95)
            )
            times = np.arange(ttft, 8.2, tpot * 6)
            ax.plot(times, np.full_like(times, y + 0.33), "o", ms=3.2,
                    color=tone, markeredgewidth=0)
            ax.text(-0.2, y + 0.33, f"user {i + 1:02d}", ha="right", va="center",
                    color=MUTED, fontsize=9)

        ax.text(-1.1, -0.75, f"throughput   {throughput:,} tok/s",
                color=MUTED, fontsize=11.5, va="top")
        ax.text(-1.1, -1.25, f"goodput   {goodput:,} tok/s",
                color=COMPUTE if goodput == 0 else MEMORY,
                fontsize=13, fontweight="bold", va="top")
        ax.set_title(title, pad=12)

    fig.suptitle("Throughput can look great while goodput is zero",
                 fontsize=14, fontweight="bold", color=INK, y=1.03)
    save(fig, BLOG01, "fig-goodput")


# ----------------------------------------------------------------------------
# Blog 02 — Inside the GPU
# ----------------------------------------------------------------------------
BLOG02 = "inference-02-inside-the-gpu"


def fig_cpu_vs_gpu() -> None:
    """Two silicon budgets: a CPU spends it on anticipation, a GPU on arithmetic."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # CPU: a few big cores, most of the die given to control and cache
    ax = axes[0]
    _blank(ax, (0, 10), (0, 8))
    ax.add_patch(Rectangle((0.3, 0.4), 9.4, 7.2, facecolor="none", edgecolor=INK))
    for i in range(4):
        x = 0.8 + (i % 2) * 4.4
        y = 4.6 - (i // 2) * 2.3
        ax.add_patch(Rectangle((x, y), 3.9, 1.9, facecolor=COMPUTE_SOFT,
                               edgecolor=COMPUTE, alpha=0.85))
        ax.text(x + 1.95, y + 0.95, "core", ha="center", va="center",
                color=INK, fontsize=10)
    ax.add_patch(Rectangle((0.8, 0.9), 8.4, 1.3, facecolor=DIVIDER,
                           edgecolor=MUTED, alpha=0.7))
    ax.text(5.0, 1.55, "branch prediction, out-of-order, caches",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    ax.text(5.0, 7.05, "a few tens of fast threads", ha="center",
            color=MUTED, fontsize=11)
    ax.set_title("CPU: rush one thread", pad=12)

    # GPU: the die is arithmetic, control is a thin strip
    ax = axes[1]
    _blank(ax, (0, 10), (0, 8))
    ax.add_patch(Rectangle((0.3, 0.4), 9.4, 7.2, facecolor="none", edgecolor=INK))
    for r in range(7):
        for c in range(16):
            ax.add_patch(Rectangle((0.75 + c * 0.55, 2.6 + r * 0.62), 0.4, 0.44,
                                   facecolor=COMPUTE_SOFT, edgecolor="none"))
    ax.add_patch(Rectangle((0.8, 0.9), 8.4, 0.8, facecolor=DIVIDER,
                           edgecolor=MUTED, alpha=0.7))
    ax.text(5.0, 1.3, "control: a thin strip", ha="center", va="center",
            color=MUTED, fontsize=10.5)
    ax.text(5.0, 7.05, "270,000 threads in flight", ha="center",
            color=MUTED, fontsize=11)
    ax.set_title("GPU: run 270,000 slow ones", pad=12)

    fig.suptitle("The same silicon budget, spent on opposite bets",
                 fontsize=14, fontweight="bold", color=INK, y=1.04)
    save(fig, BLOG02, "fig-cpu-vs-gpu")


def fig_sm_anatomy() -> None:
    """One SM: four identical processing blocks over a shared memory pool.

    The 128 CUDA cores, 4 tensor cores, 4 schedulers and 256 KB register file are
    not four separate regions of the SM. They are four copies of the same block.
    """
    sketch_style()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    _blank(ax, (0, 12.6), (-0.9, 8.6))

    ax.add_patch(Rectangle((0.25, 0.35), 12.1, 7.5, facecolor="none", edgecolor=INK))
    ax.text(6.3, 8.15, "one streaming multiprocessor (SM)  ·  stamped out 132 times",
            ha="center", color=INK, fontsize=12.5, fontweight="bold")

    for p in range(4):
        x0 = 0.6 + p * 2.95
        ax.add_patch(Rectangle((x0, 2.3), 2.6, 5.2, facecolor="none",
                               edgecolor=MUTED, alpha=0.8))
        ax.text(x0 + 1.3, 7.65, f"processing block {p + 1}", ha="center",
                color=MUTED, fontsize=9.5)

        # one warp scheduler per block
        ax.add_patch(Rectangle((x0 + 0.2, 6.75), 2.2, 0.55,
                               facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.7))
        ax.text(x0 + 1.3, 7.02, "warp scheduler", ha="center", va="center",
                color=INK, fontsize=9)

        # its own slice of the register file
        ax.add_patch(Rectangle((x0 + 0.2, 6.05), 2.2, 0.5,
                               facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.7))
        ax.text(x0 + 1.3, 6.3, "64 KB registers", ha="center", va="center",
                color=INK, fontsize=9)

        # 32 CUDA cores
        for r in range(8):
            for c in range(4):
                ax.add_patch(Rectangle((x0 + 0.42 + c * 0.55, 3.55 + r * 0.29),
                                       0.36, 0.2, facecolor=COMPUTE_SOFT,
                                       edgecolor="none"))
        ax.text(x0 + 1.3, 3.22, "32 CUDA cores", ha="center", color=COMPUTE, fontsize=9.5)

        # exactly one tensor core
        ax.add_patch(Rectangle((x0 + 0.42, 2.5), 1.76, 0.6,
                               facecolor=COMPUTE, edgecolor=COMPUTE, alpha=0.9))
        ax.text(x0 + 1.3, 2.8, "1 tensor core", ha="center", va="center",
                color=CANVAS, fontsize=9.5, fontweight="bold")

    # shared by the whole SM, not per block
    ax.add_patch(Rectangle((0.6, 0.75), 8.2, 1.25, facecolor=MEMORY_SOFT,
                           edgecolor=MEMORY, alpha=0.6))
    ax.text(4.7, 1.38, "256 KB SRAM  ·  up to 228 KB as shared memory, rest as L1",
            ha="center", va="center", color=INK, fontsize=10.5)
    ax.add_patch(Rectangle((9.1, 0.75), 2.9, 1.25, facecolor=MEMORY_SOFT,
                           edgecolor=MEMORY, alpha=0.6))
    ax.text(10.55, 1.38, "TMA\n(Hopper)", ha="center", va="center",
            color=INK, fontsize=10.5)

    ax.text(6.3, -0.45,
            "4 blocks x (32 CUDA cores + 1 tensor core) = 128 and 4.  "
            "Only the bottom row is shared.",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    save(fig, BLOG02, "fig-sm-anatomy")


def fig_memory_ladder() -> None:
    """Four levels, each step trading speed for capacity."""
    house_style()
    fig, ax = plt.subplots(figsize=(9, 4.4))

    levels = ["Registers\n(per SM)", "Shared + L1\n(per SM)", "L2 cache\n(whole chip)", "HBM\n(off-die)"]
    capacity_bytes = [256e3, 228e3, 50e6, 80e9]
    latency = ["1 cycle", "tens of cycles", "hundreds of cycles", "~482 cycles"]

    bars = ax.barh(levels, capacity_bytes, color=MEMORY, height=0.55, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(5e4, 1e12)
    ax.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11])
    ax.set_xticklabels(["100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB", "100 GB"])
    ax.set_xlabel("capacity  (log scale)")

    for bar, lat in zip(bars, latency):
        ax.text(bar.get_width() * 1.5, bar.get_y() + bar.get_height() / 2, lat,
                va="center", ha="left", color=INK, fontsize=11.5, fontweight="bold")

    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("The memory ladder: capacity up, speed down")
    save(fig, BLOG02, "fig-memory-ladder")


def fig_latency_hiding() -> None:
    """The GPU does not avoid the stall, it fills it with other warps."""
    sketch_style()
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    n_warps = 6
    _blank(ax, (-1.9, 12.4), (-1.3, n_warps + 0.3))

    for i in range(n_warps):
        y = n_warps - 1 - i
        start = i * 0.55
        # a short burst of compute, then a long wait on HBM
        ax.add_patch(Rectangle((start, y + 0.12), 0.5, 0.5,
                               facecolor=COMPUTE, edgecolor=COMPUTE))
        ax.add_patch(Rectangle((start + 0.5, y + 0.22), 8.4, 0.3,
                               facecolor=MEMORY_SOFT, edgecolor=MEMORY, alpha=0.55))
        ax.text(-0.25, y + 0.37, f"warp {i}", ha="right", va="center",
                color=MUTED, fontsize=10)

    ax.text(0.25, n_warps + 0.02, "compute", ha="left", color=COMPUTE, fontsize=10.5)
    ax.text(4.7, n_warps + 0.02, "waiting on HBM", ha="center", color=MEMORY, fontsize=10.5)

    ax.annotate("", xy=(3.6, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.4))
    ax.text(1.8, -0.72, "the SM is busy this whole time", ha="center", va="top",
            color=INK, fontsize=11.5, fontweight="bold")
    ax.text(1.8, -1.12, "every warp waits; the SM never does", ha="center", va="top",
            color=MUTED, fontsize=10.5)

    ax.set_title("Latency hiding: switching warps costs nothing", pad=12)
    save(fig, BLOG02, "fig-latency-hiding")


def fig_warp_divergence() -> None:
    """One instruction stream, so both branches run back to back."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    _blank(ax, (-0.6, 18.5), (-2.3, 4.6))

    for pass_idx, (label, active_first) in enumerate(
        [("pass 1: if-path", True), ("pass 2: else-path", False)]
    ):
        x0 = pass_idx * 9.4
        ax.text(x0 + 4.0, 4.1, label, ha="center", color=INK,
                fontsize=11.5, fontweight="bold")
        for lane in range(32):
            r, c = divmod(lane, 8)
            x = x0 + c * 1.0
            y = 2.6 - r * 0.72
            active = (lane < 16) == active_first
            ax.add_patch(
                Rectangle((x, y), 0.8, 0.56,
                          facecolor=COMPUTE if active else CANVAS,
                          edgecolor=COMPUTE if active else DIVIDER,
                          alpha=0.9 if active else 1.0)
            )
        ax.text(x0 + 4.0, -0.35, "16 lanes idle" if True else "",
                ha="center", color=MUTED, fontsize=10.5)

    ax.add_patch(Rectangle((0, -1.5), 8.4, 0.5, facecolor=COMPUTE_SOFT,
                           edgecolor=COMPUTE))
    ax.add_patch(Rectangle((9.4, -1.5), 8.4, 0.5, facecolor=COMPUTE_SOFT,
                           edgecolor=COMPUTE))
    ax.text(9.0, -2.0, "wall-clock: both paths, in sequence, for half the work",
            ha="center", va="top", color=INK, fontsize=11.5, fontweight="bold")

    ax.set_title("Warp divergence: a balanced if/else halves throughput", pad=12)
    save(fig, BLOG02, "fig-warp-divergence")


def fig_coalescing() -> None:
    """Same arithmetic, same 32 values, up to 32x the memory traffic."""
    sketch_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.4))

    for ax, (title, coalesced) in zip(axes, [("Coalesced", True), ("Uncoalesced", False)]):
        _blank(ax, (-0.4, 17), (-0.9, 2.6))
        # the memory the hardware actually drags across the bus
        n_blocks = 1 if coalesced else 8
        for b in range(8):
            fetched = coalesced or b < n_blocks
            ax.add_patch(
                Rectangle((b * 2.05, 0.5), 1.9, 0.9,
                          facecolor=MEMORY_SOFT if (coalesced and b == 0) else
                          (MEMORY_SOFT if not coalesced else CANVAS),
                          edgecolor=MEMORY if fetched else DIVIDER,
                          alpha=0.55 if fetched else 1.0)
            )
        # the values the warp actually wanted
        if coalesced:
            for t in range(8):
                ax.add_patch(Rectangle((0.1 + t * 0.22, 0.72), 0.16, 0.46,
                                       facecolor=INK, edgecolor="none"))
            ax.text(8.2, 1.95, "32 values sit in one block, so one transaction serves the warp",
                    ha="center", color=INK, fontsize=11.5, fontweight="bold")
            ax.text(8.2, -0.5, "every byte moved is a byte someone wanted",
                    ha="center", va="top", color=MUTED, fontsize=10.5)
        else:
            for b in range(8):
                ax.add_patch(Rectangle((b * 2.05 + 0.85, 0.72), 0.16, 0.46,
                                       facecolor=INK, edgecolor="none"))
            ax.text(8.2, 1.95, "32 values scattered, so up to 32 separate transactions",
                    ha="center", color=INK, fontsize=11.5, fontweight="bold")
            ax.text(8.2, -0.5, "most of every block is hauled across the bus and thrown away",
                    ha="center", va="top", color=MUTED, fontsize=10.5)
        ax.set_title(title, pad=8)

    fig.suptitle("Same math, same values: only the layout changed",
                 fontsize=14, fontweight="bold", color=INK, y=1.02)
    save(fig, BLOG02, "fig-coalescing")


def fig_bandwidth_vs_compute() -> None:
    """Across generations, compute pulls away from bandwidth."""
    house_style()
    gens = ["A100\n2020", "H100\n2022", "B200\n2024"]
    x = np.arange(3)
    bandwidth = np.array([2.0, 3.35, 8.0])
    # peak dense tensor throughput at each generation's newest format
    compute = np.array([312, 1979, 9000])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(x, compute / compute[0], marker="o", ms=8, lw=2.8, color=COMPUTE,
            label="peak compute, newest format each gen")
    ax.plot(x, bandwidth / bandwidth[0], marker="o", ms=8, lw=2.8, color=MEMORY,
            label="HBM bandwidth")
    ax.fill_between(x, bandwidth / bandwidth[0], compute / compute[0],
                    color=COMPUTE, alpha=0.10, lw=0)

    for i, (fmt, bw) in enumerate(zip(["FP16", "FP8", "FP4"], bandwidth)):
        # the first generation's two labels would otherwise collide at 1x
        ha = "left" if i == 0 else "center"
        dx = 10 if i == 0 else 0
        ax.annotate(fmt, xy=(i, compute[i] / compute[0]),
                    xytext=(dx, 13), textcoords="offset points",
                    ha=ha, color=COMPUTE, fontsize=10.5, fontweight="bold")
        ax.annotate(f"{bw} TB/s", xy=(i, bw / bandwidth[0]),
                    xytext=(dx, -22), textcoords="offset points",
                    ha=ha, color=MEMORY, fontsize=10.5)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(gens)
    ax.set_ylim(0.6, 60)
    ax.set_yticks([1, 2, 5, 10, 20, 50])
    ax.set_yticklabels(["1x", "2x", "5x", "10x", "20x", "50x"])
    ax.set_ylabel("growth since the A100  (log scale)")
    ax.set_title("Each generation widens the gap")
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    save(fig, BLOG02, "fig-bandwidth-vs-compute")


def fig_h100_vs_h200() -> None:
    """The controlled experiment: same compute die, faster memory."""
    house_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))

    panels = [
        ("Peak compute", [989, 989], "TFLOP/s", COMPUTE, "identical"),
        ("HBM bandwidth", [3.35, 4.8], "TB/s", MEMORY, "+43%"),
        ("HBM capacity", [80, 141], "GB", MEMORY, "+76%"),
    ]
    for ax, (title, vals, unit, color, delta) in zip(axes, panels):
        bars = ax.bar(["H100", "H200"], vals, color=color, width=0.55)
        # dim the H100 bar only where the H200 actually changed something
        alphas = [0.9, 0.9] if delta == "identical" else [0.45, 0.95]
        for bar, a in zip(bars, alphas):
            bar.set_alpha(a)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.03, f"{v:g}", ha="center", va="bottom",
                    color=INK, fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.30)
        ax.set_title(f"{title}  ({unit})", fontsize=12)
        ax.text(0.5, 0.90, delta, transform=ax.transAxes, ha="center",
                color=color if delta != "identical" else MUTED,
                fontsize=12, fontweight="bold")
        ax.grid(axis="x", visible=False)

    fig.suptitle("Same compute die. Only the memory changed.",
                 fontsize=14, fontweight="bold", color=INK, y=1.06)
    save(fig, BLOG02, "fig-h100-vs-h200")


def fig_precision_ladder() -> None:
    """Halving the bits pays twice: more math per second, fewer bytes to move."""
    house_style()
    fig, ax = plt.subplots(figsize=(9, 4.4))

    fmts = ["FP32", "FP16 / BF16\nVolta 2017", "FP8\nHopper 2022", "FP4\nBlackwell 2024"]
    throughput = [1, 2, 4, 8]
    bytes_per_weight = [4, 2, 1, 0.5]
    x = np.arange(len(fmts))

    ax.bar(x - 0.19, throughput, width=0.36, color=COMPUTE, alpha=0.9,
           label="relative throughput")
    ax.bar(x + 0.19, bytes_per_weight, width=0.36, color=MEMORY, alpha=0.9,
           label="bytes per weight")

    for i, (t, b) in enumerate(zip(throughput, bytes_per_weight)):
        ax.text(i - 0.19, t + 0.15, f"{t}x", ha="center", color=INK,
                fontsize=11, fontweight="bold")
        ax.text(i + 0.19, b + 0.15, f"{b:g}", ha="center", color=INK,
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(fmts, fontsize=10.5)
    ax.set_ylim(0, 9.4)
    ax.set_ylabel("relative to FP32")
    ax.set_title("Every halving of the bits pays on both ceilings")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, BLOG02, "fig-precision-ladder")


def fig_interconnect() -> None:
    """Inside the NVLink domain traffic is cheap; outside it is the bottleneck."""
    house_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

    ax = axes[0]
    labels = ["InfiniBand\n(between boxes)", "NVLink\nA100", "NVLink\nH100", "NVLink\nBlackwell"]
    vals = [50, 600, 900, 1800]
    colors = [MUTED, MEMORY_SOFT, MEMORY, MEMORY]
    bars = ax.barh(labels, vals, color=colors, height=0.6, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlim(20, 6000)
    ax.set_xlabel("per-GPU link bandwidth, GB/s  (log scale)")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() * 1.25, bar.get_y() + bar.get_height() / 2,
                f"{v:g}", va="center", color=INK, fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("An order of magnitude, per byte", fontsize=12)

    ax = axes[1]
    dom = ["DGX A100", "DGX H100", "GB200 NVL72"]
    sizes = [8, 8, 72]
    ax.bar(dom, sizes, color=MEMORY, alpha=0.9, width=0.55)
    for i, s in enumerate(sizes):
        ax.text(i, s + 2, str(s), ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 88)
    ax.set_ylabel("GPUs in one NVLink domain")
    ax.set_title("So the fast neighborhood keeps growing", fontsize=12)
    ax.grid(axis="x", visible=False)

    save(fig, BLOG02, "fig-interconnect")


BUILDERS: dict[str, list] = {
    "01": [
        fig_where_time_goes,
        fig_compute_vs_bandwidth,
        fig_weights_through_bus,
        fig_cost_per_token,
        fig_roofline,
        fig_chip_utilization,
        fig_kv_cache_growth,
        fig_decode_ceiling,
        fig_goodput,
    ],
    "02": [
        fig_cpu_vs_gpu,
        fig_sm_anatomy,
        fig_memory_ladder,
        fig_latency_hiding,
        fig_warp_divergence,
        fig_coalescing,
        fig_bandwidth_vs_compute,
        fig_h100_vs_h200,
        fig_precision_ladder,
        fig_interconnect,
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blog", help="two-digit post number, e.g. 01")
    args = parser.parse_args()

    targets = [args.blog] if args.blog else sorted(BUILDERS)
    for key in targets:
        if key not in BUILDERS:
            raise SystemExit(f"unknown blog {key!r}; known: {', '.join(sorted(BUILDERS))}")
        print(f"blog {key}:")
        for builder in BUILDERS[key]:
            builder()


if __name__ == "__main__":
    main()
