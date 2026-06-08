from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    table_path = base_dir / "results" / "tables" / "exp2_concentration_grid.csv"
    fig_dir = base_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    grid = np.loadtxt(table_path, delimiter=",")
    alphas = np.linspace(0.0, 5.0, grid.shape[1])
    betas = np.linspace(0.0, 5.0, grid.shape[0])
    alpha_mesh, beta_mesh = np.meshgrid(alphas, betas)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "network_concentration",
        [
            "#25324D",
            "#315F7B",
            "#3A9C8B",
            "#8FCF73",
            "#F1D66B",
            "#F6A64F",
        ],
    )

    setup_style()
    fig, ax = plt.subplots(figsize=(8.6, 6.5), constrained_layout=True)
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#FBFAF6")

    levels = np.linspace(0, 1, 101)
    heatmap = ax.contourf(alpha_mesh, beta_mesh, grid, levels=levels, cmap=cmap, vmin=0, vmax=1)

    contour_levels = [0.2, 0.5, 0.8, 0.95]
    contours = ax.contour(
        alpha_mesh,
        beta_mesh,
        grid,
        levels=contour_levels,
        colors=["#DDE5DD", "#FFFFFF", "#FFFFFF", "#5D4A36"],
        linewidths=[1.0, 1.45, 1.25, 1.1],
        alpha=0.92,
    )
    ax.clabel(contours, inline=True, fmt={level: f"C={level:.2g}" for level in contour_levels}, fontsize=9.2)

    ax.scatter([1.0], [1.0], s=70, marker="o", color="#FFFFFF", edgecolor="#26364E", linewidth=1.5, zorder=4)
    ax.text(
        1.08,
        1.08,
        "基准参数\nα=β=1",
        color="#26364E",
        fontsize=10.2,
        va="bottom",
        ha="left",
    )

    ax.text(
        0.38,
        0.45,
        "低集中度\n平台共存",
        color="#F1F5F0",
        fontsize=11.5,
        weight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        3.85,
        4.25,
        "高集中度\n市场锁定",
        color="#5B3B18",
        fontsize=12.0,
        weight="bold",
        ha="center",
        va="center",
    )

    ax.set_title("实验 2：双边网络效应强度与市场集中度", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("用户侧受商户吸引强度  α", fontsize=12.3, labelpad=10)
    ax.set_ylabel("商户侧受用户吸引强度  β", fontsize=12.3, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 5.1, 1))
    ax.set_yticks(np.arange(0, 5.1, 1))
    ax.tick_params(axis="both", labelsize=10.5, colors="#32383A")
    ax.spines["left"].set_color("#A9A197")
    ax.spines["bottom"].set_color("#A9A197")
    ax.grid(color="#FFFFFF", alpha=0.17, linewidth=0.8)

    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035, ticks=np.linspace(0, 1, 6))
    colorbar.set_label("市场集中度  C", fontsize=11.8, labelpad=10)
    colorbar.ax.tick_params(labelsize=10.2, colors="#32383A")
    colorbar.outline.set_edgecolor("#B8B0A5")

    png_path = fig_dir / "exp2_network_effect_heatmap_beautiful.png"
    svg_path = fig_dir / "exp2_network_effect_heatmap_beautiful.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
