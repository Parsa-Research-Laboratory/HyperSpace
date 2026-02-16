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

def plot_best_triptych(
    image_dict: dict,
    *,
    gt_key: str = "Ground Truth",
    best_hrr_key: str,
    best_fhrr_key: str,
    cmap: str = "viridis",
    preset: str = "single",          # "single" (1-col) or "double" (2-col)
    width_single: float = 3.35,
    width_double: float = 6.9,
    aspect: float = 0.42,            # height = width*aspect for 1×3
    share_scale: bool = True,        # same vmin/vmax across all 3 for fair visual compare
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
    savepath: str | None = None,
    dpi: int = 300,
):
    keys = [gt_key, best_hrr_key, best_fhrr_key]
    for k in keys:
        if k not in image_dict:
            raise KeyError(f"Key '{k}' not found in image_dict.")

    imgs = [np.asarray(image_dict[k]) for k in keys]

    if share_scale and (vmin is None or vmax is None):
        stacked = np.stack(imgs, axis=0)
        if vmin is None:
            vmin = float(np.nanmin(stacked))
        if vmax is None:
            vmax = float(np.nanmax(stacked))

    width = width_single if preset == "single" else width_double
    height = width * aspect

    plt.rcParams.update({
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
    })

    fig, axes = plt.subplots(1, 3, figsize=(width, height), dpi=dpi, constrained_layout=True)
    mappable = None
    for ax, k, img in zip(axes, keys, imgs):
        mappable = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

        if "FHRR" in k:
            k = "FHRR"

        elif "HRR" in k:
            k = "HRR"

        ax.set_title(k, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # # one shared colorbar
    # cbar = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.02)
    # cbar.set_label("Regressed value")

    if title is not None:
        fig.suptitle(title, y=1.02)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")

    return fig, axes