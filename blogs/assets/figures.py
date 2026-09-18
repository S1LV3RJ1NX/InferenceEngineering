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


# ----------------------------------------------------------------------------
# Blog 03 — Kernels and FlashAttention 1
# ----------------------------------------------------------------------------
BLOG03 = "inference-03-kernels-and-flashattention"


def fig_three_passes() -> None:
    """The naive kernel parks the N x N score matrix in HBM and reads it back twice."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    _blank(ax, (0, 13), (-1.2, 6.4))

    ax.add_patch(Rectangle((0.3, 1.0), 3.2, 4.4, facecolor="none", edgecolor=COMPUTE))
    ax.text(1.9, 5.75, "GPU die", ha="center", color=COMPUTE, fontsize=11.5,
            fontweight="bold")
    ax.text(1.9, 3.2, "tensor\ncores", ha="center", va="center", color=INK, fontsize=11)

    ax.add_patch(Rectangle((9.6, 1.0), 3.1, 4.4, facecolor=MEMORY_SOFT,
                           edgecolor=MEMORY, alpha=0.45))
    ax.text(11.15, 5.75, "HBM", ha="center", color=MEMORY, fontsize=11.5,
            fontweight="bold")
    ax.text(11.15, 3.2, "S  stored\nin full", ha="center", va="center",
            color=INK, fontsize=11)

    passes = [
        (4.6, "pass 1", "compute S, write it out"),
        (3.2, "pass 2", "read S, softmax, write back"),
        (1.8, "pass 3", "read it again, multiply by V"),
    ]
    for y, label, detail in passes:
        ax.add_patch(FancyArrowPatch((3.7, y), (9.4, y), arrowstyle="<->",
                                     mutation_scale=18, color=MEMORY, lw=2))
        ax.text(6.55, y + 0.28, f"{label}: {detail}", ha="center",
                color=INK, fontsize=10.5)

    ax.text(6.55, -0.55, "three full crossings of the bus, each of size N x N",
            ha="center", va="top", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("The naive kernel: same math, three trips through the slow tier", pad=12)
    save(fig, BLOG03, "fig-three-passes")


def fig_nsquared_growth() -> None:
    """The score matrix grows with the square of the sequence length."""
    house_style()
    n = np.logspace(np.log10(512), np.log10(131072), 200)
    entries = n**2

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(n, entries, lw=2.8, color=MEMORY)
    ax.fill_between(n, 1e4, entries, color=MEMORY, alpha=0.10, lw=0)

    for tokens, label in [(1024, "1K tokens\n1.0M entries"),
                          (8192, "8K\n67M"),
                          (32768, "32K\n1.1B")]:
        ax.plot([tokens], [tokens**2], "o", ms=8, color=MEMORY)
        ax.annotate(label, xy=(tokens, tokens**2), xytext=(tokens * 1.12, tokens**2 * 0.16),
                    color=MUTED, fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(512, 131072)
    ax.set_ylim(1e5, 2e10)
    ax.set_xlabel("sequence length  (tokens, log scale)")
    ax.set_ylabel("entries in the score matrix  (log scale)")
    ax.set_title("Double the sequence, quadruple the matrix")
    save(fig, BLOG03, "fig-nsquared-growth")


def fig_tiling() -> None:
    """What is resident on chip, drawn against the full matrices it is cut from.

    The point of the figure is that every solid shape is one slice. Drawing the
    blocks without their parent matrices made them read as whole tensors.
    """
    sketch_style()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    _blank(ax, (0, 31), (-2.4, 10.2))

    n_div = 8
    active = 2

    def tall(x0, label, active_idx, color, soft):
        """A full N x d matrix, faint, with one row block solid."""
        ax.add_patch(Rectangle((x0, 1.2), 3.0, 7.0, facecolor="none",
                               edgecolor=DIVIDER, ls="--"))
        h = 7.0 / n_div
        for i in range(n_div):
            if i == active_idx:
                ax.add_patch(Rectangle((x0, 1.2 + i * h), 3.0, h,
                                       facecolor=soft, edgecolor=color))
        ax.text(x0 + 1.5, 8.7, label, ha="center", color=color,
                fontsize=11.5, fontweight="bold")

    tall(0.6, "Q", n_div - 1 - active, MEMORY, MEMORY_SOFT)
    ax.text(2.1, 0.55, "one block,\nstays put", ha="center", va="top",
            color=MUTED, fontsize=9.5)

    tall(5.0, "K", 3, MEMORY, MEMORY_SOFT)
    tall(9.4, "V", 3, MEMORY, MEMORY_SOFT)
    ax.text(9.4, 0.55, "one block each,\nstreaming past", ha="center", va="top",
            color=MUTED, fontsize=9.5)

    # the score matrix, with exactly one tile live
    ax.add_patch(Rectangle((14.6, 1.2), 7.0, 7.0, facecolor="none",
                           edgecolor=DIVIDER, ls="--"))
    c = 7.0 / n_div
    for r in range(n_div):
        for col in range(n_div):
            live = r == active and col == 3
            ax.add_patch(Rectangle((14.6 + col * c, 1.2 + (n_div - 1 - r) * c), c, c,
                                   facecolor=COMPUTE if live else "none",
                                   edgecolor=COMPUTE if live else DIVIDER,
                                   ls="-" if live else "--"))
    ax.text(18.1, 8.7, "S", ha="center", color=COMPUTE, fontsize=11.5,
            fontweight="bold")
    ax.text(18.1, 0.55, "one tile, built and\noverwritten each step",
            ha="center", va="top", color=MUTED, fontsize=9.5)

    tall(24.6, "O", n_div - 1 - active, MEMORY, MEMORY_SOFT)
    ax.text(26.1, 0.55, "one block,\naccumulating", ha="center", va="top",
            color=MUTED, fontsize=9.5)

    ax.text(15.5, 9.6,
            "solid = resident on chip          dashed = the full matrix, which never is",
            ha="center", color=INK, fontsize=11.5, fontweight="bold")
    ax.text(15.5, -1.9,
            "every solid shape is a single slice. the dashed outlines are drawn only to show what it is a slice of",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    save(fig, BLOG03, "fig-tiling")


def fig_attention_shapes() -> None:
    """Every tensor in attention is thin except the one in the middle."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    _blank(ax, (0, 26), (-2.6, 7.6))

    def strip(x, label, sub):
        ax.add_patch(Rectangle((x, 0.8), 1.5, 5.4, facecolor=MEMORY_SOFT,
                               edgecolor=MEMORY, alpha=0.8))
        ax.text(x + 0.75, 6.6, label, ha="center", color=INK,
                fontsize=12, fontweight="bold")
        ax.text(x + 0.75, 0.2, sub, ha="center", va="top", color=MUTED, fontsize=10)

    strip(0.6, "Q", "8192 x 128")
    strip(2.9, "K", "8192 x 128")
    strip(5.2, "V", "8192 x 128")

    ax.text(7.6, 3.5, "give", ha="center", va="center", color=MUTED, fontsize=11)

    # the middle object, drawn wide because it is
    ax.add_patch(Rectangle((9.0, 0.8), 8.2, 5.4, facecolor=COMPUTE,
                           edgecolor=COMPUTE, alpha=0.9))
    ax.text(13.1, 6.6, "S = Q K$^\\top$", ha="center", color=INK,
            fontsize=12, fontweight="bold")
    ax.text(13.1, 3.5, "8192 x 8192", ha="center", va="center",
            color=CANVAS, fontsize=13, fontweight="bold")
    ax.text(13.1, 0.2, "128 MiB", ha="center", va="top", color=COMPUTE,
            fontsize=11, fontweight="bold")

    ax.text(18.6, 3.5, "then", ha="center", va="center", color=MUTED, fontsize=11)

    strip(20.4, "O", "8192 x 128")
    ax.text(23.4, 3.5, "2 MiB", ha="left", va="center", color=MUTED, fontsize=11)

    ax.text(13.0, -1.5,
            "the inputs and the output are all thin. only the intermediate is square,\n"
            "and it is 64x wider than anything that produced it",
            ha="center", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.set_title("The shape of the problem", pad=14)
    save(fig, BLOG03, "fig-attention-shapes")


def fig_softmax_wall() -> None:
    """A tile sees a slice; softmax needs the row."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(11, 4.0))
    _blank(ax, (-2.4, 34), (-3.0, 4.2))

    # the full row of scores
    for i in range(32):
        inside = i < 4
        ax.add_patch(Rectangle((i, 1.6), 0.9, 1.2,
                               facecolor=COMPUTE if inside else CANVAS,
                               edgecolor=COMPUTE if inside else DIVIDER,
                               alpha=0.9 if inside else 1.0))
    ax.text(-0.4, 2.2, "one row\nof scores", ha="right", va="center",
            color=INK, fontsize=10.5)

    # what the tile can see
    ax.add_patch(Rectangle((-0.15, 1.35), 4.2, 1.7, facecolor="none",
                           edgecolor=COMPUTE, lw=2.4))
    ax.text(1.95, 3.5, "what this tile holds", ha="center", color=COMPUTE,
            fontsize=11, fontweight="bold")

    # what softmax demands
    ax.add_patch(FancyArrowPatch((0.2, 0.9), (31.7, 0.9), arrowstyle="<->",
                                 mutation_scale=18, color=MEMORY, lw=2))
    ax.text(16, 0.25, "softmax needs the maximum and the sum over ALL of this",
            ha="center", va="top", color=MEMORY, fontsize=11.5, fontweight="bold")

    ax.text(16, -1.5,
            "you hold 64 of 8,192 scores. you cannot subtract a maximum you have not seen,\n"
            "and you cannot divide by a total that does not exist yet",
            ha="center", va="center", color=INK, fontsize=11.5)
    ax.set_title("Why the matmul tiles but the softmax does not", pad=12)
    save(fig, BLOG03, "fig-softmax-wall")


def fig_block_sweep() -> None:
    """The sweep, and what it does to the accumulator.

    The budget table lives in the prose, so this figure answers the other
    question instead: after one tile, how much of O exists?
    """
    sketch_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    _blank(ax, (-3.4, 18.4), (-4.6, 6.4))

    n = 8
    done, now = 3, 3

    # --- the row of S tiles this query block is sweeping ---
    ax.text(-0.6, 4.75, "one\nQ block", ha="right", va="center",
            color=MEMORY, fontsize=10.5, fontweight="bold")
    for i in range(n):
        if i < done:
            fc, ec, ls = COMPUTE_SOFT, COMPUTE, "-"
        elif i == now:
            fc, ec, ls = COMPUTE, COMPUTE, "-"
        else:
            fc, ec, ls = "none", DIVIDER, "--"
        ax.add_patch(Rectangle((i * 2.1, 4.1), 1.85, 1.3,
                               facecolor=fc, edgecolor=ec, ls=ls))
    ax.text(now * 2.1 + 0.92, 4.75, "now", ha="center", va="center",
            color=CANVAS, fontsize=9.5, fontweight="bold")
    ax.text(8.4, 6.0, "S tiles, one per K/V block, swept left to right",
            ha="center", color=COMPUTE, fontsize=11)
    ax.text(8.4, 3.5, "128 of them at 8K tokens", ha="center", va="top",
            color=MUTED, fontsize=10)

    # --- the accumulator underneath, same shape throughout ---
    labels = [
        (0.0, "after tile 1", "every row holds a\npartial value", 0.25),
        (6.2, "after tile 4", "every row still\npartial", 0.55),
        (12.4, "after tile 128", "complete,\nnot yet divided", 1.0),
    ]
    for x, when, state, maturity in labels:
        # intensity, not a fill level: a partially filled box would wrongly
        # suggest part of the block is finished while the rest is empty
        ax.add_patch(Rectangle((x, 0.4), 4.4, 1.5, facecolor=MEMORY_SOFT,
                               edgecolor=MEMORY, alpha=maturity))
        ax.text(x + 2.2, 2.25, when, ha="center", color=INK, fontsize=10.5,
                fontweight="bold")
        ax.text(x + 2.2, -0.05, state, ha="center", va="top", color=MUTED,
                fontsize=9.5)
        if x < 12:
            ax.add_patch(FancyArrowPatch((x + 4.7, 1.15), (x + 5.9, 1.15),
                                         arrowstyle="->", mutation_scale=16,
                                         color=MUTED, lw=1.6))

    ax.text(17.4, 1.15, "÷ $\\ell$", ha="left", va="center", color=COMPUTE,
            fontsize=13, fontweight="bold")

    ax.text(8.4, -2.4,
            "the accumulator is full size from the first step.\n"
            "what changes across the sweep is its value, not its shape",
            ha="center", va="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(8.4, -3.9,
            "each tile adds a partial contribution to every row of the block, so no row is finished until the last tile lands",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    save(fig, BLOG03, "fig-block-sweep")


def fig_online_softmax() -> None:
    """When the running maximum moves, one multiply repairs everything banked."""
    house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    tile1 = np.array([2.0, 1.0, 3.0, 0.5])
    tile2 = np.array([5.0, 2.5, 1.0, 3.5])

    # before: tile 1 banked on the scale of its own maximum, 3.0
    ax = axes[0]
    p1 = np.exp(tile1 - tile1.max())
    ax.bar(range(4), p1, color=MEMORY, alpha=0.9, width=0.6)
    for i, (s, p) in enumerate(zip(tile1, p1)):
        ax.text(i, p + 0.03, f"{p:.2f}", ha="center", color=INK, fontsize=10.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s:g}" for s in tile1])
    ax.set_xlabel("tile 1 scores")
    ax.set_ylabel("banked term  $e^{s-m}$")
    ax.set_title("Running max $m=3$,  denominator $=1.585$", fontsize=12)
    ax.grid(axis="x", visible=False)

    # after: the max jumps to 5, so every banked term takes the same factor
    ax = axes[1]
    corrected = p1 * np.exp(3.0 - 5.0)
    p2 = np.exp(tile2 - 5.0)
    ax.bar(range(4), corrected, color=MEMORY, alpha=0.45, width=0.6)
    ax.bar(range(4, 8), p2, color=COMPUTE, alpha=0.9, width=0.6)
    for i, p in enumerate(np.concatenate([corrected, p2])):
        ax.text(i, p + 0.03, f"{p:.2f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(8))
    ax.set_xticklabels([f"{s:g}" for s in np.concatenate([tile1, tile2])], fontsize=9)
    ax.set_xlabel("all eight scores")
    ax.set_title("Running max $m=5$,  denominator $=1.538$", fontsize=12)
    ax.grid(axis="x", visible=False)
    ax.annotate("these four scaled by\n$e^{3-5}=0.135$, one multiply",
                xy=(1.5, 0.17), xytext=(0.55, 0.72),
                color=MEMORY, fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=MEMORY, lw=1.6))

    fig.suptitle("Online softmax: the banked terms were all wrong by the same factor",
                 fontsize=14, fontweight="bold", color=INK, y=1.04)
    save(fig, BLOG03, "fig-online-softmax")


def fig_memory_traffic() -> None:
    """Score-matrix traffic across HBM. The unambiguous claim: it goes to zero.

    Deliberately counts only S, not total traffic. FlashAttention still re-reads
    K and V once per query block, so a total-traffic comparison depends on the
    block size and is a much smaller ratio than this figure would imply.
    """
    house_style()
    fig, ax = plt.subplots(figsize=(8.6, 4.2))

    lengths = [1024, 8192, 32768]
    labels = ["1K tokens", "8K tokens", "32K tokens"]
    naive = [3 * n**2 for n in lengths]

    x = np.arange(len(lengths))
    ax.bar(x, naive, color=MEMORY, alpha=0.9, width=0.45)
    for i, v in enumerate(naive):
        ax.text(i, v * 1.35, f"{v / 1e9:.2f}B" if v > 1e9 else f"{v / 1e6:.0f}M",
                ha="center", color=INK, fontsize=12, fontweight="bold")
        ax.text(i + 0.45, v * 0.05, "none", ha="center", va="bottom",
                color=COMPUTE, fontsize=11.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yscale("log")
    ax.set_ylim(1e5, 1e11)
    ax.set_ylabel("score-matrix elements crossing HBM  (log scale)")
    ax.set_title("What the naive kernel moves, and what FlashAttention moves")
    ax.text(0.985, 0.93,
            "teal: naive, $3N^2$ elements\nterracotta: FlashAttention, zero",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5, color=MUTED)
    ax.grid(axis="x", visible=False)
    save(fig, BLOG03, "fig-memory-traffic")


def fig_peak_fraction() -> None:
    """The number this series tracks: fraction of peak arithmetic sustained."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.8))

    # only figures the papers actually report; no interpolated baseline
    gens = ["FlashAttention 1\nA100", "FlashAttention 2\nA100",
            "FlashAttention 3\nH100", "FlashAttention 4\nB200"]
    low = np.array([25, 50, 75, 71])
    high = np.array([40, 73, 75, 71])

    y = np.arange(len(gens))
    ax.barh(y, high - low + 1.2, left=low, color=COMPUTE, alpha=0.9, height=0.55)
    for i, (lo, hi) in enumerate(zip(low, high)):
        label = f"{lo}-{hi}%" if lo != hi else f"{hi}%"
        ax.text(hi + 3, i, label, va="center", color=INK, fontsize=11.5,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(gens, fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("fraction of the chip's peak arithmetic actually sustained")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("The number this series tracks")
    save(fig, BLOG03, "fig-peak-fraction")


BLOG04 = "inference-04-flashattention-2"

# NVIDIA A100 SXM, the chip FlashAttention 2 was written for
A100_TENSOR = 312e12   # dense FP16 tensor-core peak, ops/s
A100_SCALAR = 19.5e12  # FP32 non-tensor path, ops/s
A100_SMS = 108


def fig_scalar_exchange_rate() -> None:
    """One scalar operation costs about sixteen matmul operations of machine time."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(11, 3.6))
    _blank(ax, (0, 34), (-2.2, 6.4))

    ratio = round(A100_TENSOR / A100_SCALAR)  # 16
    span = 28.0

    # the same wall-clock span, spent two different ways
    ax.add_patch(Rectangle((3, 3.4), span, 1.5, facecolor=MUTED, alpha=0.16,
                           edgecolor=MUTED))
    ax.text(3 + span / 2, 4.15, "one scalar operation", ha="center", va="center",
            color=INK, fontsize=12, fontweight="bold")
    ax.text(2.4, 4.15, "CUDA\ncores", ha="right", va="center", color=MUTED, fontsize=10)

    w = span / ratio
    for i in range(ratio):
        ax.add_patch(Rectangle((3 + i * w + 0.12, 0.9), w - 0.24, 1.5,
                               facecolor=COMPUTE_SOFT, edgecolor=COMPUTE))
    ax.text(2.4, 1.65, "tensor\ncores", ha="right", va="center", color=MUTED, fontsize=10)
    ax.text(3 + span / 2, 0.1, f"{ratio} matrix operations fit in the same time",
            ha="center", va="top", color=COMPUTE, fontsize=12, fontweight="bold")

    ax.text(17, 5.8,
            f"A100: {A100_TENSOR/1e12:.0f} trillion matmul ops/s against "
            f"{A100_SCALAR/1e12:.1f} trillion scalar ops/s",
            ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(17, -1.7,
            "every rescale and division the kernel cannot avoid is paid for at this rate",
            ha="center", va="center", color=MUTED, fontsize=10.5)
    save(fig, BLOG04, "fig-scalar-exchange-rate")


def fig_block_occupancy() -> None:
    """Blocks are what fills multiprocessors, and version 1 did not make enough."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    cols, rows = 12, 9  # 108 cells, one per A100 multiprocessor
    panels = [
        (axes[0], 32, "FlashAttention 1", "batch 2 x 16 heads = 32 blocks"),
        (axes[1], A100_SMS, "FlashAttention 2", "query blocks join the count, so the grid fills"),
    ]
    for ax, live, title, note in panels:
        _blank(ax, (-0.6, cols + 0.6), (-2.4, rows + 1.6))
        for r in range(rows):
            for c in range(cols):
                on = r * cols + c < live
                ax.add_patch(Rectangle((c + 0.08, rows - 1 - r + 0.08), 0.84, 0.84,
                                       facecolor=COMPUTE_SOFT if on else "none",
                                       edgecolor=COMPUTE if on else DIVIDER))
        ax.text(cols / 2, rows + 0.9, title, ha="center", color=INK,
                fontsize=12.5, fontweight="bold")
        idle = A100_SMS - live
        busy = f"{live} of {A100_SMS} multiprocessors busy"
        ax.text(cols / 2, -0.7, busy, ha="center", color=COMPUTE,
                fontsize=11.5, fontweight="bold")
        tail = note if idle == 0 else f"{note}, so {idle} sit dark"
        ax.text(cols / 2, -1.7, tail, ha="center", color=MUTED, fontsize=10)

    fig.suptitle("A block runs on one multiprocessor, so blocks are the unit of occupancy",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    save(fig, BLOG04, "fig-block-occupancy")


def fig_warp_split() -> None:
    """Splitting the contracted index forces a merge; splitting the surviving one does not."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    for ax, mode in zip(axes, ("K, V", "Q")):
        _blank(ax, (0, 24), (-2.6, 10.4))
        splits_k = mode.startswith("K")
        for w in range(4):
            y = 7.4 - w * 2.0
            ax.text(0.4, y + 0.45, f"warp {w}", ha="left", va="center",
                    color=MUTED, fontsize=10)
            ax.add_patch(Rectangle((4.2, y), 4.2, 0.95,
                                   facecolor=MEMORY_SOFT, edgecolor=MEMORY))
            ax.text(6.3, y + 0.47, f"{mode} slice", ha="center", va="center",
                    color=INK, fontsize=10.5)
            if splits_k:
                ax.annotate("", xy=(12.4, y + 0.47), xytext=(8.6, y + 0.47),
                            arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.6))
                ax.annotate("", xy=(17.2, y + 0.47), xytext=(15.4, y + 0.47),
                            arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.6))
            else:
                ax.annotate("", xy=(17.2, y + 0.47), xytext=(8.6, y + 0.47),
                            arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.6))
            ax.add_patch(Rectangle((17.5, y), 5.0, 0.95,
                                   facecolor=COMPUTE_SOFT if not splits_k else "none",
                                   edgecolor=COMPUTE))
            label = "partial rows" if splits_k else "finished rows"
            ax.text(20.0, y + 0.47, label, ha="center", va="center",
                    color=INK if not splits_k else MUTED, fontsize=10.5)

        if splits_k:
            ax.add_patch(Rectangle((12.6, 0.6), 2.6, 8.4,
                                   facecolor=COMPUTE_SOFT, alpha=0.5, edgecolor=COMPUTE))
            ax.text(13.9, 4.8, "shared\nmemory", ha="center", va="center",
                    color=INK, fontsize=10.5, fontweight="bold")

        title = ("FlashAttention 1: split K and V" if splits_k
                 else "FlashAttention 2: split Q")
        ax.text(12, 9.7, title, ha="center", color=INK, fontsize=12.5,
                fontweight="bold")
        tail = ("every warp holds a piece of the same rows,\nso they must trade before anything finishes"
                if splits_k else
                "every warp owns whole rows end to end,\nso there is nothing to trade")
        ax.text(12, -1.9, tail, ha="center", va="center", color=MUTED, fontsize=10.5)

    save(fig, BLOG04, "fig-warp-split")


def fig_fa2_across_chips() -> None:
    """The same kernel, two chips: Hopper moved and the kernel did not."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.6, 3.4))

    labels = ["FlashAttention 1\non A100", "FlashAttention 2\non A100",
              "FlashAttention 2\non H100"]
    low = np.array([25, 50, 35])
    high = np.array([40, 73, 35])
    colors = [MUTED, COMPUTE, COMPUTE]
    alphas = [0.45, 0.9, 0.45]

    y = np.arange(len(labels))
    bars = ax.barh(y, high - low + 1.2, left=low, color=colors, height=0.55)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    for i, (lo, hi) in enumerate(zip(low, high)):
        text = f"{lo}-{hi}%" if lo != hi else f"about {hi}%"
        ax.text(hi + 3, i, text, va="center", color=INK, fontsize=11.5,
                fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("fraction of the chip's peak arithmetic actually sustained")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("The same kernel, measured on two chips")
    save(fig, BLOG04, "fig-fa2-across-chips")


def fig_work_units() -> None:
    """Where the extra blocks come from: one head's query rows, cut up."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))

    for ax, split in zip(axes, (False, True)):
        _blank(ax, (0, 16), (-3.0, 11.6))
        ax.text(8, 10.7, "FlashAttention 2" if split else "FlashAttention 1",
                ha="center", color=INK, fontsize=12.5, fontweight="bold")
        ax.text(8, 9.7, "one head of one sequence, 8,192 query rows",
                ha="center", color=MUTED, fontsize=10.5)

        if split:
            stripes = 16  # drawn count; the real one is 128
            h = 8.0 / stripes
            for i in range(stripes):
                ax.add_patch(Rectangle((4.5, 0.9 + i * h + 0.04), 7.0, h - 0.08,
                                       facecolor=COMPUTE_SOFT, edgecolor=COMPUTE))
            ax.text(8, 0.1, "128 blocks, 64 query rows each",
                    ha="center", va="top", color=COMPUTE, fontsize=11.5,
                    fontweight="bold")
            ax.text(8, -1.5, "128 multiprocessors can work on this one head at once",
                    ha="center", va="top", color=MUTED, fontsize=10)
        else:
            ax.add_patch(Rectangle((4.5, 0.9), 7.0, 8.0,
                                   facecolor=COMPUTE_SOFT, edgecolor=COMPUTE))
            ax.text(8, 4.9, "1 block", ha="center", va="center", color=INK,
                    fontsize=13, fontweight="bold")
            ax.text(8, 0.1, "1 block, all 8,192 query rows",
                    ha="center", va="top", color=COMPUTE, fontsize=11.5,
                    fontweight="bold")
            ax.text(8, -1.5, "one multiprocessor works through all 8,192 rows alone",
                    ha="center", va="top", color=MUTED, fontsize=10)

    fig.suptitle("Same work, cut into different numbers of pieces",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.03)
    save(fig, BLOG04, "fig-work-units")


BLOG05 = "inference-05-flashattention-3-and-4"

# Hopper H100 and Blackwell B200, the two chips this post compares
H100_BF16 = 989e12    # dense BF16 tensor-core peak, ops/s
B200_BF16 = 2250e12   # dense BF16 tensor-core peak, ops/s


def fig_warp_specialization() -> None:
    """Loads and math in turn, against loads and math at the same time."""
    sketch_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 4.8))

    load, math = MEMORY_SOFT, COMPUTE_SOFT
    load_e, math_e = MEMORY, COMPUTE

    # version 2: one lane, alternating
    ax = axes[0]
    _blank(ax, (0, 32), (-1.6, 3.4))
    x = 1.0
    for i in range(4):
        ax.add_patch(Rectangle((x, 1.2), 2.6, 1.3, facecolor=load, edgecolor=load_e))
        ax.text(x + 1.3, 1.85, "load", ha="center", va="center", color=INK, fontsize=9.5)
        x += 2.8
        ax.add_patch(Rectangle((x, 1.2), 4.2, 1.3, facecolor=math, edgecolor=math_e))
        ax.text(x + 2.1, 1.85, "compute", ha="center", va="center", color=INK, fontsize=9.5)
        x += 4.4
    ax.text(0.6, 3.0, "FlashAttention 2: one lane, strictly in turn",
            ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.text(x + 0.4, 1.85, "time", ha="left", va="center", color=MUTED, fontsize=10)

    # version 3: two lanes, overlapped
    ax = axes[1]
    _blank(ax, (0, 32), (-1.6, 4.6))
    for i in range(5):
        lx = 1.0 + i * 4.2
        ax.add_patch(Rectangle((lx, 2.6), 3.9, 1.1, facecolor=load, edgecolor=load_e))
        ax.text(lx + 1.95, 3.15, "load", ha="center", va="center", color=INK, fontsize=9.5)
    for i in range(4):
        cx = 5.2 + i * 4.2
        ax.add_patch(Rectangle((cx, 1.0), 3.9, 1.1, facecolor=math, edgecolor=math_e))
        ax.text(cx + 1.95, 1.55, "compute", ha="center", va="center", color=INK, fontsize=9.5)
    ax.text(0.6, 4.2, "FlashAttention 3: producer warps load, consumer warps compute",
            ha="left", color=INK, fontsize=12, fontweight="bold")
    ax.text(0.2, 3.15, "producers", ha="right", va="center", color=MUTED, fontsize=9.5)
    ax.text(0.2, 1.55, "consumers", ha="right", va="center", color=MUTED, fontsize=9.5)
    ax.text(30.0, -1.0,
            "the same tiles, the same arithmetic, finished in less wall clock",
            ha="right", va="center", color=MUTED, fontsize=10.5)

    save(fig, BLOG05, "fig-warp-specialization")


def fig_fp8_outliers() -> None:
    """Where FP8's levels land depends on the widest value, not the common one."""
    sketch_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 5.0))

    def bell(ax, centre, width, height, base):
        xs = np.linspace(centre - width, centre + width, 120)
        ys = base + height * np.exp(-0.5 * ((xs - centre) / (width / 2.6)) ** 2)
        ax.fill_between(xs, base, ys, color=MEMORY_SOFT, edgecolor=MEMORY, lw=1.6)

    n_levels = 25

    # before: the scale is stretched by two far-out values
    ax = axes[0]
    _blank(ax, (-7.2, 7.2), (-1.5, 3.2))
    ax.plot([-6.4, 6.4], [0, 0], color=MUTED, lw=1.4)
    for x in np.linspace(-6.4, 6.4, n_levels):
        ax.plot([x, x], [-0.22, 0.22], color=MUTED, lw=1.2)
    bell(ax, 0.0, 1.05, 2.2, 0.0)
    for x in (-5.4, 5.6):
        ax.plot([x], [0.38], marker="o", color=COMPUTE, markersize=8)
    ax.text(5.6, 1.0, "outliers", ha="center", color=COMPUTE, fontsize=10.5,
            fontweight="bold")
    ax.text(0, 2.9, "Before: the scale must reach the outliers",
            ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(0, -1.1, "3 of the 25 levels land where almost every value actually is",
            ha="center", va="center", color=MUTED, fontsize=10.5)

    # after: a rotation spreads the loud coordinates thinly, so nothing is far out
    ax = axes[1]
    _blank(ax, (-7.2, 7.2), (-1.5, 3.2))
    ax.plot([-2.6, 2.6], [0, 0], color=MUTED, lw=1.4)
    for x in np.linspace(-2.6, 2.6, n_levels):
        ax.plot([x, x], [-0.22, 0.22], color=MUTED, lw=1.2)
    bell(ax, 0.0, 2.3, 2.2, 0.0)
    ax.text(0, 2.9, "After a random rotation: nothing is far out",
            ha="center", color=INK, fontsize=12, fontweight="bold")
    ax.text(0, -1.1, "the same 25 levels now spread across the values that occur",
            ha="center", va="center", color=MUTED, fontsize=10.5)

    fig.suptitle("FP8 has a couple hundred levels; the widest value decides where they land",
                 fontsize=13, fontweight="bold", color=INK, y=1.02)
    save(fig, BLOG05, "fig-fp8-outliers")


def fig_asymmetric_scaling() -> None:
    """One engine got 2.25 times faster. Its neighbours did not move."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.4))

    parts = ["Tensor cores\nBF16 matmul", "Special function unit\nexponentials",
             "Shared memory\nbandwidth"]
    growth = [B200_BF16 / H100_BF16, 1.0, 1.0]
    colors = [COMPUTE, MUTED, MUTED]

    y = np.arange(len(parts))
    bars = ax.barh(y, growth, color=colors, height=0.55)
    bars[1].set_alpha(0.45)
    bars[2].set_alpha(0.45)
    for i, g in enumerate(growth):
        ax.text(g + 0.04, i, f"{g:.2f}x", va="center", color=INK, fontsize=11.5,
                fontweight="bold")

    ax.axvline(1.0, color=DIVIDER, lw=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(parts, fontsize=10.5)
    ax.set_xlim(0, 2.7)
    ax.set_xlabel("growth from H100 to B200")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("Asymmetric hardware scaling")
    save(fig, BLOG05, "fig-asymmetric-scaling")


def fig_exponential_gap() -> None:
    """Per multiprocessor per cycle, the tensor cores outrun the exponentials 512 to 1."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.2))

    labels = ["Tensor cores\nBF16 ops", "Special function unit\nexponentials"]
    vals = [8192, 16]
    bars = ax.barh([0, 1], vals, color=[COMPUTE, MEMORY], height=0.5)
    bars[1].set_alpha(0.9)
    for i, v in enumerate(vals):
        ax.text(v * 1.25, i, f"{v:,} per cycle", va="center", color=INK,
                fontsize=11.5, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(8, 40000)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("operations per multiprocessor per cycle, B200, log scale")
    ax.set_title(f"The consumer is {8192 // 16} times faster than the producer")
    save(fig, BLOG05, "fig-exponential-gap")


def fig_fraction_vs_absolute() -> None:
    """The fraction plateaued near three quarters; the work done did not."""
    house_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.6))

    gens = ["FA1\nA100", "FA2\nA100", "FA3\nH100", "FA4\nB200"]
    frac = [32.5, 61.5, 75, 71]  # midpoints of the reported ranges
    absolute = [None, None, 740, 1605]  # measured ops/s, in trillions

    ax = axes[0]
    ax.plot(gens, frac, marker="o", color=COMPUTE, lw=2.4, markersize=8)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Fraction of peak sustained", fontsize=12)
    ax.text(2.5, 84, "plateau", ha="center", color=MUTED, fontsize=10.5)

    ax = axes[1]
    idx = [i for i, v in enumerate(absolute) if v is not None]
    ax.bar([gens[i] for i in idx], [absolute[i] for i in idx],
           color=COMPUTE, alpha=0.9, width=0.5)
    for i in idx:
        ax.text(gens[i], absolute[i] + 60, f"{absolute[i]:,}", ha="center",
                color=INK, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 2000)
    ax.set_ylabel("trillion ops/s, 16-bit")
    ax.set_title("Work actually done", fontsize=12)
    ax.set_xlim(-0.7, 1.7)

    fig.suptitle("A flat fraction of a rising ceiling is still more arithmetic",
                 fontsize=13, fontweight="bold", color=INK, y=1.05)
    save(fig, BLOG05, "fig-fraction-vs-absolute")


BLOG06 = "inference-06-paged-attention"


def fig_reservation_waste() -> None:
    """Reserve for the worst case and most of the reservation is never used."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))

    # internal: one reservation, mostly empty
    ax = axes[0]
    _blank(ax, (0, 12), (-2.6, 11.2))
    ax.add_patch(Rectangle((4.0, 0.8), 4.0, 8.4, facecolor="none",
                           edgecolor=MEMORY, ls="--"))
    used = 8.4 * 200 / 4096
    ax.add_patch(Rectangle((4.0, 0.8), 4.0, used, facecolor=MEMORY_SOFT,
                           edgecolor=MEMORY))
    ax.text(6.0, 9.9, "one reservation", ha="center", color=INK, fontsize=12,
            fontweight="bold")
    ax.text(8.5, 0.8 + used / 2, "200 tokens used", ha="left", va="center",
            color=MEMORY, fontsize=10.5, fontweight="bold")
    ax.text(8.5, 5.4, "3,896 slots\nreserved and empty", ha="left", va="center",
            color=MUTED, fontsize=10.5)
    ax.text(3.6, 5.0, "4,096\ntoken limit", ha="right", va="center", color=MUTED,
            fontsize=10.5)
    ax.text(6.0, -1.6, "internal fragmentation", ha="center", va="center",
            color=COMPUTE, fontsize=11.5, fontweight="bold")

    # external: the gaps between reservations go ragged
    ax = axes[1]
    _blank(ax, (0, 12), (-2.6, 11.2))
    spans = [(0.8, 2.2), (3.6, 1.6), (6.4, 1.1), (8.6, 0.6)]
    for y, h in spans:
        ax.add_patch(Rectangle((1.2, y), 4.6, h, facecolor=MEMORY_SOFT,
                               edgecolor=MEMORY))
    ax.text(3.5, 9.9, "memory after a while", ha="center", color=INK, fontsize=12,
            fontweight="bold")
    # the holes left between reservations, drawn as holes rather than labelled
    for y, h in [(3.0, 0.6), (5.2, 1.2), (7.5, 1.1)]:
        ax.add_patch(Rectangle((1.2, y), 4.6, h, facecolor="none",
                               edgecolor=DIVIDER, ls="--"))
        ax.text(3.5, y + h / 2, "free", ha="center", va="center", color=MUTED,
                fontsize=9.5)
    ax.text(6.4, 6.4, "a new request fits in the\ntotal free space, but not\nin any single gap",
            ha="left", va="center", color=MUTED, fontsize=10)
    ax.text(3.5, -1.6, "external fragmentation", ha="center", va="center",
            color=COMPUTE, fontsize=11.5, fontweight="bold")

    save(fig, BLOG06, "fig-reservation-waste")


def fig_block_table() -> None:
    """One table turns scattered physical blocks into one logical sequence."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(12, 5.0))
    _blank(ax, (0, 34), (-2.4, 11.6))

    # what the kernel believes it is reading
    ax.text(3.2, 10.4, "what the kernel sees", ha="center", color=INK,
            fontsize=11.5, fontweight="bold")
    for i in range(5):
        y = 8.2 - i * 1.6
        ax.add_patch(Rectangle((1.6, y), 3.2, 1.3, facecolor=MEMORY_SOFT,
                               edgecolor=MEMORY))
        ax.text(3.2, y + 0.65, f"block {i}", ha="center", va="center", color=INK,
                fontsize=10)
    ax.text(3.2, -0.9, "one contiguous sequence", ha="center", va="top",
            color=MUTED, fontsize=10)

    # the indirection
    ax.text(11.4, 10.4, "block table", ha="center", color=INK, fontsize=11.5,
            fontweight="bold")
    ax.add_patch(Rectangle((8.2, 0.4), 6.4, 9.2, facecolor="none", edgecolor=DIVIDER))
    phys = [10, 43, 61, 24, 52]
    for i, blk in enumerate(phys):
        y = 8.2 - i * 1.6
        ax.text(9.0, y + 0.65, f"{i}", ha="left", va="center", color=MUTED,
                fontsize=10)
        ax.text(13.8, y + 0.65, f"blk-{blk}", ha="right", va="center", color=INK,
                fontsize=10)
        ax.annotate("", xy=(8.0, y + 0.65), xytext=(5.0, y + 0.65),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    # where the bytes actually are
    ax.text(25.4, 10.4, "HBM", ha="center", color=INK, fontsize=11.5,
            fontweight="bold")
    cols, rows = 7, 6
    live = {(0, 2), (2, 5), (3, 1), (4, 6), (5, 3)}
    for r in range(rows):
        for c in range(cols):
            on = (r, c) in live
            ax.add_patch(Rectangle((17.6 + c * 1.6, 0.6 + (rows - 1 - r) * 1.5),
                                   1.35, 1.25,
                                   facecolor=MEMORY_SOFT if on else "none",
                                   edgecolor=MEMORY if on else DIVIDER))
    ax.text(25.4, -0.9, "the same five blocks, wherever they happened to fit",
            ha="center", va="top", color=MUTED, fontsize=10)

    ax.text(17, -2.1,
            "every free block is the same size, so any free block will do",
            ha="center", va="center", color=COMPUTE, fontsize=11,
            fontweight="bold")
    save(fig, BLOG06, "fig-block-table")


def fig_cache_utilization() -> None:
    """How much of the most expensive memory in the machine held anything."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.2))

    labels = ["Before paging\nreserve for the limit", "With paging\n16-token blocks"]
    low = np.array([20.4, 96.2])
    high = np.array([38.2, 96.2])

    y = np.arange(len(labels))
    bars = ax.barh(y, np.maximum(high - low, 1.2), left=low,
                   color=[MUTED, COMPUTE], height=0.5)
    bars[0].set_alpha(0.55)
    ax.text(high[0] + 2.5, 0, "20.4-38.2%", va="center", color=INK, fontsize=11.5,
            fontweight="bold")
    ax.text(high[1] + 2.5, 1, "about 96%", va="center", color=INK, fontsize=11.5,
            fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("share of KV cache memory actually holding token states")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("More than half of it was storing nothing")
    save(fig, BLOG06, "fig-cache-utilization")


def fig_cache_levers() -> None:
    """The architectural lever: how much each token has to store at all."""
    house_style()
    fig, ax = plt.subplots(figsize=(9.4, 3.6))

    # elements cached per token per layer, so two models can sit on one axis
    labels = [
        "Llama-3-70B\nfull multi-head",
        "Llama-3-70B\ngrouped, 8 groups",
        "DeepSeek-V2\nfull multi-head",
        "DeepSeek-V2\nlatent",
    ]
    vals = [2 * 64 * 128, 2 * 8 * 128, 2 * 128 * 128, 576]
    colors = [MUTED, MEMORY, MUTED, MEMORY]

    y = np.arange(len(labels))
    bars = ax.barh(y, vals, color=colors, height=0.55)
    for b in (bars[0], bars[2]):
        b.set_alpha(0.5)
    for i, v in enumerate(vals):
        ax.text(v * 1.15, i, f"{v:,}", va="center", color=INK, fontsize=11,
                fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(300, 150000)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("elements cached per token per layer, log scale")
    ax.set_title("Grouping divides the head count; compression replaces it")
    save(fig, BLOG06, "fig-cache-levers")


BLOG07 = "inference-07-flash-decoding"

H100_SMS = 132


def fig_decode_block_collapse() -> None:
    """Prefill has thousands of blocks to hand out. Decode has almost none."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    cols, rows = 12, 11  # 132 cells, one per H100 multiprocessor
    for ax, live, title, note in (
        (axes[0], H100_SMS, "Prefill", "thousands of blocks, more than enough to go around"),
        (axes[1], 32, "Decode, one request", "batch 1 x 32 heads = 32 blocks"),
    ):
        _blank(ax, (-0.6, cols + 0.6), (-2.6, rows + 1.6))
        for r in range(rows):
            for c in range(cols):
                on = r * cols + c < live
                ax.add_patch(Rectangle((c + 0.08, rows - 1 - r + 0.08), 0.84, 0.84,
                                       facecolor=COMPUTE_SOFT if on else "none",
                                       edgecolor=COMPUTE if on else DIVIDER))
        ax.text(cols / 2, rows + 0.9, title, ha="center", color=INK, fontsize=12.5,
                fontweight="bold")
        idle = H100_SMS - live
        ax.text(cols / 2, -0.8, f"{live} of {H100_SMS} multiprocessors busy",
                ha="center", color=COMPUTE, fontsize=11.5, fontweight="bold")
        tail = note if idle == 0 else f"{note}, so {idle} stand idle"
        ax.text(cols / 2, -1.8, tail, ha="center", color=MUTED, fontsize=10)

    fig.suptitle("The query at decode is a single row, so it contributes one block",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    save(fig, BLOG07, "fig-decode-block-collapse")


def fig_flash_decoding_split() -> None:
    """Split the one dimension nobody had split: the cache itself."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))

    # before: one worker walks the whole cache
    ax = axes[0]
    _blank(ax, (0, 20), (-2.6, 10.4))
    ax.add_patch(Rectangle((2.0, 1.4), 15.0, 2.4, facecolor=MEMORY_SOFT,
                           edgecolor=MEMORY))
    ax.text(9.5, 2.6, "the whole cached sequence", ha="center", va="center",
            color=INK, fontsize=11)
    ax.add_patch(Rectangle((2.0, 6.4), 2.6, 1.8, facecolor=COMPUTE_SOFT,
                           edgecolor=COMPUTE))
    ax.text(3.3, 7.3, "1 SM", ha="center", va="center", color=INK, fontsize=10.5)
    ax.annotate("", xy=(16.4, 4.4), xytext=(3.3, 6.2),
                arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.6))
    ax.text(9.5, 9.5, "One worker, tile after tile", ha="center", color=INK,
            fontsize=12, fontweight="bold")
    ax.text(9.5, -1.4, "the longer the context, the longer the serial walk",
            ha="center", va="center", color=MUTED, fontsize=10.5)

    # after: chunks in parallel, then one merge
    ax = axes[1]
    _blank(ax, (0, 20), (-2.6, 10.4))
    for i in range(5):
        x = 2.0 + i * 3.1
        ax.add_patch(Rectangle((x, 1.4), 2.8, 2.4, facecolor=MEMORY_SOFT,
                               edgecolor=MEMORY))
        ax.add_patch(Rectangle((x + 0.3, 5.0), 2.2, 1.5, facecolor=COMPUTE_SOFT,
                               edgecolor=COMPUTE))
        ax.text(x + 1.4, 5.75, "SM", ha="center", va="center", color=INK, fontsize=9.5)
        # each worker reads its own chunk, then hands a partial result upward
        ax.annotate("", xy=(x + 1.4, 4.9), xytext=(x + 1.4, 3.9),
                    arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=1.4))
        ax.annotate("", xy=(9.5, 7.3), xytext=(x + 1.4, 6.6),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax.add_patch(Rectangle((6.6, 7.4), 5.8, 1.3, facecolor="none", edgecolor=MUTED))
    ax.text(9.5, 8.05, "merge kernel", ha="center", va="center", color=INK,
            fontsize=10.5, fontweight="bold")
    ax.text(9.5, 9.5, "One worker per chunk, all at once", ha="center", color=INK,
            fontsize=12, fontweight="bold")
    ax.text(9.5, -1.4,
            "each chunk keeps its own running max and sum; the merge rescales them onto one scale",
            ha="center", va="center", color=MUTED, fontsize=9.5)

    save(fig, BLOG07, "fig-flash-decoding-split")


def fig_launch_overhead() -> None:
    """At batch one the gaps between kernels are the same size as the kernels."""
    sketch_style()
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 4.2))

    ax = axes[0]
    _blank(ax, (0, 34), (-1.4, 3.0))
    x = 1.0
    for i in range(7):
        ax.add_patch(Rectangle((x, 1.0), 2.6, 1.3, facecolor=COMPUTE_SOFT,
                               edgecolor=COMPUTE))
        x += 2.6
        ax.add_patch(Rectangle((x, 1.0), 1.5, 1.3, facecolor="none",
                               edgecolor=MUTED, ls="--"))
        x += 1.6
    ax.text(0.6, 2.7, "One kernel at a time: the GPU waits for the CPU between each",
            ha="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.text(x + 0.4, 1.65, "time", ha="left", va="center", color=MUTED, fontsize=10)
    ax.text(17, -0.9, "dashed gaps are launch overhead, 20 to 30% of the step at batch 1",
            ha="center", va="center", color=MUTED, fontsize=10)

    ax = axes[1]
    _blank(ax, (0, 34), (-1.6, 3.0))
    x = 1.0
    ax.add_patch(Rectangle((x, 1.0), 1.5, 1.3, facecolor="none", edgecolor=MUTED,
                           ls="--"))
    x += 1.6
    for i in range(7):
        ax.add_patch(Rectangle((x, 1.0), 2.6, 1.3, facecolor=COMPUTE_SOFT,
                               edgecolor=COMPUTE))
        x += 2.7
    ax.text(0.6, 2.7, "Graph replay: one call dispatches the whole recorded sequence",
            ha="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.text(x + 0.4, 1.65, "time", ha="left", va="center", color=COMPUTE,
            fontsize=10, fontweight="bold")

    save(fig, BLOG07, "fig-launch-overhead")


def fig_decode_speedup() -> None:
    """Reported gains, which land at long context and small batch."""
    house_style()
    fig, ax = plt.subplots(figsize=(8.8, 3.0))

    labels = ["The attention operation", "End-to-end decoding"]
    vals = [50, 8]
    bars = ax.barh([0, 1], vals, color=[MEMORY, COMPUTE], height=0.5)
    for i, v in enumerate(vals):
        ax.text(v + 1.2, i, f"up to {v}x", va="center", color=INK, fontsize=11.5,
                fontweight="bold")

    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, 60)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("speedup over unsplit FlashAttention, CodeLlama 34B at long context")
    ax.set_title("Where the gain lands")
    save(fig, BLOG07, "fig-decode-speedup")


BLOG08 = "inference-08-quantization"

# desaturated tones for the bit-field diagram, so the semantic teal/terracotta
# pair is not spent on a chart where neither means compute or memory
EXP_C = "#4E7784"
MANT_C = "#B5734E"


def fig_bit_fields() -> None:
    """What each format spends its bits on, drawn to scale by total width."""
    house_style()
    fig, ax = plt.subplots(figsize=(10.4, 4.8))

    # name, exponent bits, mantissa bits, integer (INT formats)
    fmts = [
        ("FP32", 8, 23, False),
        ("TF32", 8, 10, False),
        ("FP16", 5, 10, False),
        ("BF16", 8, 7, False),
        ("FP8 E4M3", 4, 3, False),
        ("FP8 E5M2", 5, 2, False),
        ("INT8", 0, 7, True),
    ]
    unit = 0.30
    h = 0.66
    for row, (name, exp, mant, is_int) in enumerate(fmts):
        y = len(fmts) - 1 - row
        x = 0.0
        ax.add_patch(Rectangle((x, y), unit, h, facecolor=INK, edgecolor="white"))
        x += unit
        if exp:
            ax.add_patch(Rectangle((x, y), exp * unit, h, facecolor=EXP_C,
                                   edgecolor="white"))
            ax.text(x + exp * unit / 2, y + h / 2, str(exp), ha="center",
                    va="center", color="white", fontsize=10, fontweight="bold")
            x += exp * unit
        label = "integer" if is_int else "mantissa"
        ax.add_patch(Rectangle((x, y), mant * unit, h, facecolor=MANT_C,
                               edgecolor="white"))
        ax.text(x + mant * unit / 2, y + h / 2, str(mant), ha="center",
                va="center", color="white", fontsize=10, fontweight="bold")
        x += mant * unit
        total = 1 + exp + mant
        ax.text(-0.25, y + h / 2, name, ha="right", va="center", color=INK,
                fontsize=11.5, fontweight="bold")
        ax.text(x + 0.2, y + h / 2, f"{total} bits", ha="left", va="center",
                color=MUTED, fontsize=10)

    ax.text(0.15, len(fmts) + 0.15, "sign", color=INK, fontsize=9.5, rotation=90,
            va="bottom", ha="center")
    ax.text(2.2, len(fmts) + 0.05, "exponent = range", color=EXP_C, fontsize=11,
            fontweight="bold")
    ax.text(6.0, len(fmts) + 0.05, "mantissa = precision", color=MANT_C,
            fontsize=11, fontweight="bold")

    _blank(ax, (-3.0, 12.6), (-0.5, len(fmts) + 0.9))
    ax.set_title("Every format below FP32 throws away bits from one field or the other",
                 fontsize=13, fontweight="bold", color=INK, loc="left", x=-0.02)
    save(fig, BLOG08, "fig-bit-fields")


def fig_float_spacing() -> None:
    """Floats crowd their resolution near zero, exactly where weights live."""
    sketch_style()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    _blank(ax, (-9.8, 9.5), (-2.6, 6.0))

    ax.text(-9.6, 5.6, "Floats are a logarithmic ruler; integers are a linear one",
            ha="left", va="center", color=INK, fontsize=13, fontweight="bold")

    # a bell of weights, hugging zero
    xs = np.linspace(-8, 8, 240)
    bell = 1.5 * np.exp(-0.5 * (xs / 1.1) ** 2)
    ax.fill_between(xs, 2.6, 2.6 + bell, color=MEMORY_SOFT, edgecolor=MEMORY,
                    lw=1.4)
    ax.text(0, 4.5, "where a trained network's weights actually sit",
            ha="center", color=MEMORY, fontsize=10.5, fontweight="bold")

    # float ticks: powers of two with mantissa steps, both signs
    floats = []
    for e in range(-4, 4):
        base = 2.0 ** e
        for m in range(4):
            floats.append(base * (1 + m / 4))
    floats = sorted(set(floats + [-f for f in floats] + [0.0]))
    ax.plot([-8.2, 8.2], [1.4, 1.4], color=MUTED, lw=1.2)
    for f in floats:
        if -8.2 <= f <= 8.2:
            ax.plot([f, f], [1.2, 1.6], color=MANT_C, lw=1.3)
    ax.text(-8.7, 1.4, "float", ha="right", va="center", color=INK, fontsize=11,
            fontweight="bold")

    # integer ticks: evenly spaced
    ints = np.linspace(-8, 8, 17)
    ax.plot([-8.2, 8.2], [-0.4, -0.4], color=MUTED, lw=1.2)
    for i in ints:
        ax.plot([i, i], [-0.6, -0.2], color=EXP_C, lw=1.3)
    ax.text(-8.7, -0.4, "integer", ha="right", va="center", color=INK,
            fontsize=11, fontweight="bold")

    ax.text(0, -1.6,
            "same tick budget: floats spend it near zero, integers spread it evenly",
            ha="center", color=MUTED, fontsize=10.5)
    save(fig, BLOG08, "fig-float-spacing")


def fig_outlier_channels() -> None:
    """A few activation channels run far hotter than the rest, and one scale must reach them."""
    house_style()
    fig, ax = plt.subplots(figsize=(10.2, 3.8))

    rng = np.random.default_rng(7)
    n = 48
    mags = np.abs(rng.normal(0, 0.22, n)) + 0.05
    outliers = [7, 15, 23, 30, 38, 44]
    for c in outliers:
        mags[c] = rng.uniform(70, 130)

    colors = [COMPUTE if i in outliers else MEMORY for i in range(n)]
    ax.bar(range(n), mags, color=colors, width=0.82)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 400)
    ax.set_xlim(-1, n)
    ax.set_xlabel("activation channel (the same positions in every token's vector)")
    ax.set_ylabel("magnitude, log scale")
    ax.set_xticks([])

    ax.axhline(mags.max(), ls=":", lw=1.3, color=COMPUTE)
    ax.text(n - 1, mags.max() * 1.15, "one shared scale must reach up here",
            ha="right", color=COMPUTE, fontsize=10.5, fontweight="bold")
    ax.text(1, 0.5, "so everything ordinary collapses into the bottom few steps",
            ha="left", color=MUTED, fontsize=10.5)
    ax.set_title("Outlier features: about 0.1% of channels, running ~100x hotter")
    ax.grid(axis="x", visible=False)
    save(fig, BLOG08, "fig-outlier-channels")


def fig_roofline_slide() -> None:
    """Quantization slides a workload right; only the memory-bound side gains."""
    house_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))

    ai = np.logspace(-1, 4, 400)
    roof = np.minimum(H100_BW * ai, H100_FLOPS)
    ax.fill_between(ai, 1e11, roof, where=ai <= RIDGE, color=MEMORY, alpha=0.10, lw=0)
    ax.fill_between(ai, 1e11, roof, where=ai >= RIDGE, color=COMPUTE, alpha=0.10, lw=0)
    ax.plot(ai[ai <= RIDGE], H100_BW * ai[ai <= RIDGE], lw=2.6, color=MEMORY)
    ax.plot(ai[ai >= RIDGE], np.full((ai >= RIDGE).sum(), H100_FLOPS), lw=2.6,
            color=COMPUTE)
    ax.axvline(RIDGE, ls=":", lw=1.4, color=MUTED)
    ax.annotate(f"ridge · {RIDGE:.0f} ops/byte", xy=(RIDGE, 2e11),
                xytext=(RIDGE * 1.4, 1.5e11), color=INK, fontsize=10.5)

    # decode slides right along the rising bandwidth roof: pure speedup
    ax.annotate("", xy=(4, H100_BW * 4), xytext=(1, H100_BW * 1),
                arrowprops=dict(arrowstyle="->", color=MEMORY, lw=2.4))
    ax.plot([1], [H100_BW], "o", ms=10, color=MEMORY)
    ax.text(0.42, H100_BW * 5.5, "DECODE\nfewer bytes, slides right\nunder the roof: pure speedup",
            color=MEMORY, fontsize=10.5, fontweight="bold")

    # prefill already on the flat roof: sliding right buys nothing
    ax.annotate("", xy=(4000, H100_FLOPS), xytext=(1000, H100_FLOPS),
                arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=2.4))
    ax.plot([1000], [H100_FLOPS], "o", ms=10, color=COMPUTE)
    ax.text(360, 1.4e14, "PREFILL\nalready on the ceiling\nfewer bytes buy almost nothing",
            color=COMPUTE, fontsize=10.5, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 1e4)
    ax.set_ylim(1e11, 3e15)
    ax.set_xlabel("arithmetic intensity (ops per byte)")
    ax.set_ylabel("achievable performance (FLOP/s)")
    ax.set_title("Quantization pushes right on the roofline")
    save(fig, BLOG08, "fig-roofline-slide")


def fig_strategy_split() -> None:
    """Two bottlenecks, two schemes: shrink the weights, or shrink both sides."""
    sketch_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    def operand(ax, x, y, w, label, bits, color):
        ax.add_patch(Rectangle((x, y), w, 1.6, facecolor=color, edgecolor=INK))
        ax.text(x + w / 2, y + 0.95, label, ha="center", va="center", color=INK,
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + 0.4, bits, ha="center", va="center", color=INK,
                fontsize=9.5)

    # W4A16: only the weight shrinks; aimed at memory-bound decode
    ax = axes[0]
    _blank(ax, (0, 16), (-2.4, 6.2))
    operand(ax, 1.0, 2.0, 2.0, "W", "4-bit", COMPUTE_SOFT)
    ax.text(3.6, 2.8, "x", ha="center", va="center", color=MUTED, fontsize=14)
    operand(ax, 4.4, 2.0, 4.0, "A", "16-bit", MEMORY_SOFT)
    ax.annotate("", xy=(13.4, 2.8), xytext=(8.8, 2.8),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.text(11.0, 3.4, "fewer bytes\nfrom HBM", ha="center", va="center",
            color=INK, fontsize=10.5)
    ax.text(8, 5.4, "W4A16", ha="center", color=INK, fontsize=13, fontweight="bold")
    ax.text(8, -1.4, "shrinks the weights only · attacks memory-bound decode · win is bandwidth",
            ha="center", va="center", color=MUTED, fontsize=10)

    # W8A8 / W4A4: both sides low; unlocks low-precision tensor cores
    ax = axes[1]
    _blank(ax, (0, 16), (-2.4, 6.2))
    operand(ax, 1.6, 2.0, 2.6, "W", "8 or 4-bit", COMPUTE_SOFT)
    ax.text(4.7, 2.8, "x", ha="center", va="center", color=MUTED, fontsize=14)
    operand(ax, 5.2, 2.0, 2.6, "A", "8 or 4-bit", COMPUTE_SOFT)
    ax.annotate("", xy=(13.8, 2.8), xytext=(8.2, 2.8),
                arrowprops=dict(arrowstyle="->", color=COMPUTE, lw=2.0))
    ax.text(11.0, 3.5, "~2x tensor-core\nthroughput", ha="center", va="center",
            color=COMPUTE, fontsize=10.5, fontweight="bold")
    ax.text(8, 5.4, "W8A8 or W4A4", ha="center", color=INK, fontsize=13,
            fontweight="bold")
    ax.text(8, -1.4, "shrinks both sides · attacks compute-bound prefill · win is arithmetic",
            ha="center", va="center", color=MUTED, fontsize=10)

    fig.suptitle("The bottleneck decides which side of the multiply to shrink",
                 fontsize=13.5, fontweight="bold", color=INK, y=1.02)
    save(fig, BLOG08, "fig-strategy-split")


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
    "03": [
        fig_attention_shapes,
        fig_three_passes,
        fig_nsquared_growth,
        fig_tiling,
        fig_softmax_wall,
        fig_block_sweep,
        fig_online_softmax,
        fig_memory_traffic,
        fig_peak_fraction,
    ],
    "04": [
        fig_scalar_exchange_rate,
        fig_work_units,
        fig_block_occupancy,
        fig_warp_split,
        fig_fa2_across_chips,
    ],
    "05": [
        fig_warp_specialization,
        fig_fp8_outliers,
        fig_asymmetric_scaling,
        fig_exponential_gap,
        fig_fraction_vs_absolute,
    ],
    "06": [
        fig_reservation_waste,
        fig_block_table,
        fig_cache_utilization,
        fig_cache_levers,
    ],
    "07": [
        fig_decode_block_collapse,
        fig_flash_decoding_split,
        fig_launch_overhead,
        fig_decode_speedup,
    ],
    "08": [
        fig_bit_fields,
        fig_float_spacing,
        fig_outlier_channels,
        fig_roofline_slide,
        fig_strategy_split,
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
