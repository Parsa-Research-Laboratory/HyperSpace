import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_hyperspace_latency_bars(
    data: dict,
    *,
    title: str = "Operation Latency: HRR vs FHRR",
    ylabel: str = "Mean latency per sample (ms)",
    hrr_label: str = "HRR",
    fhrr_label: str = "FHRR",
    hrr_color: str = "purple",
    fhrr_color: str = "green",
    figsize=(7.2, 3.2),          # good for 1-column-ish figures; adjust as needed
    dpi: int = 300,
    group_spacing: float = 1.25, # spacing between operation groups
    bar_w: float = 0.28,         # bar width
    bar_offset: float = 0.22,    # distance from group center; controls gap between HRR/FHRR
    rotate_xticks: int = 35,
    ylim: tuple | None = None,
    savepath: str | None = None, # e.g., "latency_ops.pdf"
):
    """
    data format expected:
      data[op_name] = [[hrr_value, fhrr_value], ["HRR", "FHRR"]]
    (Your current structure works as-is.)
    """

    # --- Deterministic ordering ---
    ops = list(data.keys())

    # --- Robust extraction (uses labels list to locate HRR/FHRR indices) ---
    hrr_vals = []
    fhrr_vals = []
    for op in ops:
        vals, labels = data[op]
        try:
            hrr_idx = labels.index(hrr_label)
            fhrr_idx = labels.index(fhrr_label)
        except ValueError as e:
            raise ValueError(
                f"Expected labels to include '{hrr_label}' and '{fhrr_label}' for op='{op}', got {labels}"
            ) from e
        hrr_vals.append(vals[hrr_idx])
        fhrr_vals.append(vals[fhrr_idx])

    hrr_vals = np.asarray(hrr_vals, dtype=float)
    fhrr_vals = np.asarray(fhrr_vals, dtype=float)

    # --- X positions with group spacing ---
    x = np.arange(len(ops)) * group_spacing

    # --- Styling defaults that look good in papers ---
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.bar(x - bar_offset, hrr_vals, width=bar_w, color=hrr_color, edgecolor="none")
    ax.bar(x + bar_offset, fhrr_vals, width=bar_w, color=fhrr_color, edgecolor="none")

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(ops, rotation=rotate_xticks, ha="right")

    # Clean paper-ish look
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ylim is not None:
        ax.set_ylim(*ylim)

    # Single legend
    handles = [
        Patch(facecolor=hrr_color, label=hrr_label),
        Patch(facecolor=fhrr_color, label=fhrr_label),
    ]
    ax.legend(handles=handles, loc="best", frameon=False)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    return fig, ax