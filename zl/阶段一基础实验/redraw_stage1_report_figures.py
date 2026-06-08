from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from model import PlatformParams, run_simulation


FIG_BG = "#F7F5EF"
AX_BG = "#FBFAF6"
GRID = "#D8D1C5"
TEXT = "#32383A"
GREEN = "#2F9C5B"
ORANGE = "#E58F3A"
BLUE = "#557C83"
RED = "#D95F59"
INK = "#25324D"


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def finish_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_BG)
    ax.tick_params(axis="both", labelsize=10.5, colors=TEXT)
    ax.spines["left"].set_color("#A9A197")
    ax.spines["bottom"].set_color("#A9A197")


def save_report_figure(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    png = fig_dir / f"{stem}.png"
    original = fig_dir / f"{stem}_original.png"
    beautiful_png = fig_dir / f"{stem}_beautiful.png"
    beautiful_svg = fig_dir / f"{stem}_beautiful.svg"

    if png.exists() and not original.exists():
        shutil.copy2(png, original)

    fig.savefig(beautiful_png, bbox_inches="tight")
    fig.savefig(beautiful_svg, bbox_inches="tight")
    plt.close(fig)
    shutil.copy2(beautiful_png, png)
    print(f"Updated: {png.name}")


def plot_exp1_share(fig_dir: Path, kind: str) -> None:
    params = PlatformParams()
    cases = [
        ("对称初始", r"$u_0=m_0=0.50$", 0.50, 0.50, BLUE, 2.2),
        ("轻微领先", r"$u_0=m_0=0.55$", 0.55, 0.55, ORANGE, 2.7),
        ("明显领先", r"$u_0=m_0=0.70$", 0.70, 0.70, GREEN, 2.7),
    ]
    title_name = "用户份额" if kind == "user" else "商户份额"
    ylabel = "平台 A 用户份额  u(t)" if kind == "user" else "平台 A 商户份额  m(t)"
    stem = "exp1_user_share_dynamics" if kind == "user" else "exp1_merchant_share_dynamics"

    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.80, 0.86, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.5, color="#6F7D7D", lw=1.1, ls=(0, (4, 4)), alpha=0.45)

    offsets = {"对称初始": -0.026, "轻微领先": -0.034, "明显领先": 0.034}
    finals: list[float] = []
    for label, detail, u0, m0, color, lw in cases:
        t, u, m = run_simulation(params, u0=u0, m0=m0)
        y = u if kind == "user" else m
        finals.append(float(y[-1]))
        ax.plot(t, y, color=color, lw=lw, solid_capstyle="round", zorder=3)
        ax.scatter([t[-1]], [y[-1]], s=34, color=color, edgecolor=AX_BG, linewidth=1.1, zorder=4)
        ax.text(
            t[-1] + 1.15,
            y[-1] + offsets[label],
            f"{label}  {y[-1]:.3f}\n{detail}",
            color=color,
            fontsize=10.2,
            va="center",
            linespacing=1.25,
        )

    equilibrium = max(finals)
    ax.axhline(equilibrium, color=GREEN, lw=1.0, ls=(0, (2, 4)), alpha=0.42, zorder=1)
    ax.text(1.0, equilibrium + 0.018, f"领先情形稳态 ≈ {equilibrium:.3f}", color="#2F6F49", fontsize=10.5)
    ax.set_title(f"实验 1：平台 A {title_name}动态演化", fontsize=18, pad=18, weight="bold")
    ax.set_xlabel("时间 t", fontsize=12.5, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=12.5, labelpad=12)
    ax.set_xlim(-0.5, 64.0)
    ax.set_ylim(0.44, 0.88)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_yticks(np.arange(0.45, 0.89, 0.05))
    ax.grid(axis="y", color="#CFC8BA", alpha=0.45, linewidth=0.85)
    ax.grid(axis="x", color=GRID, alpha=0.22, linewidth=0.75)
    save_report_figure(fig, fig_dir, stem)


def plot_exp2_heatmap(fig_dir: Path, table_dir: Path) -> None:
    grid = np.loadtxt(table_dir / "exp2_concentration_grid.csv", delimiter=",")
    alphas = np.linspace(0.0, 5.0, grid.shape[1])
    betas = np.linspace(0.0, 5.0, grid.shape[0])
    aa, bb = np.meshgrid(alphas, betas)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "network_concentration",
        ["#25324D", "#315F7B", "#3A9C8B", "#8FCF73", "#F1D66B", "#F6A64F"],
    )

    fig, ax = plt.subplots(figsize=(8.6, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    heatmap = ax.contourf(aa, bb, grid, levels=np.linspace(0, 1, 101), cmap=cmap, vmin=0, vmax=1)
    contour_levels = [0.2, 0.5, 0.8, 0.95]
    contours = ax.contour(
        aa,
        bb,
        grid,
        levels=contour_levels,
        colors=["#DDE5DD", "#FFFFFF", "#FFFFFF", "#5D4A36"],
        linewidths=[1.0, 1.45, 1.25, 1.1],
        alpha=0.92,
    )
    ax.clabel(contours, inline=True, fmt={v: f"C={v:.2g}" for v in contour_levels}, fontsize=9.2)
    ax.scatter([1.0], [1.0], s=70, marker="o", color="#FFFFFF", edgecolor=INK, linewidth=1.5, zorder=4)
    ax.text(1.08, 1.08, "基准参数\nα=β=1", color=INK, fontsize=10.2, va="bottom", ha="left")
    ax.text(0.38, 0.45, "低集中度\n平台共存", color="#F1F5F0", fontsize=11.5, weight="bold", ha="center", va="center")
    ax.text(3.85, 4.25, "高集中度\n市场锁定", color="#5B3B18", fontsize=12.0, weight="bold", ha="center", va="center")
    ax.set_title("实验 2：双边网络效应强度与市场集中度", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("用户侧受商户吸引强度  α", fontsize=12.3, labelpad=10)
    ax.set_ylabel("商户侧受用户吸引强度  β", fontsize=12.3, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 5.1, 1))
    ax.set_yticks(np.arange(0, 5.1, 1))
    ax.grid(color="#FFFFFF", alpha=0.17, linewidth=0.8)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035, ticks=np.linspace(0, 1, 6))
    colorbar.set_label("市场集中度  C", fontsize=11.8, labelpad=10)
    colorbar.ax.tick_params(labelsize=10.2, colors=TEXT)
    colorbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp2_network_effect_heatmap")


def plot_exp3_initial_advantage(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp3_initial_advantage.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["setting"], []).append(row)

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.5, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhspan(0.0, 0.5, color=RED, alpha=0.045, linewidth=0)
    ax.axvline(0.5, color="#6F7D7D", lw=1.1, ls=(0, (4, 4)), alpha=0.6)
    ax.axhline(0.5, color="#6F7D7D", lw=1.1, ls=(0, (4, 4)), alpha=0.6)

    palette = {
        "弱网络效应 alpha=beta=0.5": BLUE,
        "强网络效应 alpha=beta=3.0": GREEN,
    }
    labels = {
        "弱网络效应 alpha=beta=0.5": "弱网络效应  α=β=0.5",
        "强网络效应 alpha=beta=3.0": "强网络效应  α=β=3.0",
    }
    for setting, items in groups.items():
        x = np.array([float(r["x"]) for r in items])
        y = np.array([float(r["L_A"]) for r in items])
        ax.plot(x, y, color=palette.get(setting, ORANGE), lw=2.8, solid_capstyle="round", label=labels.get(setting, setting))

    ax.text(0.19, 0.54, "弱网络效应：回归共存", color=BLUE, fontsize=10.5)
    ax.text(0.64, 0.88, "强网络效应：初始优势被放大", color="#247B46", fontsize=10.5)
    ax.text(0.52, 0.08, "B 初始优势区", color="#A94844", fontsize=10.2)
    ax.text(0.52, 0.96, "A 初始优势区", color="#247B46", fontsize=10.2, va="top")
    ax.set_title("实验 3：初始规模差异与最终市场份额", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel(r"初始份额  $x = u_A(0) = m_A(0)$", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0.1, 0.91, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=10.2)
    save_report_figure(fig, fig_dir, "exp3_initial_advantage")


def plot_exp3_asymmetric(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp3_asymmetric_initial_cases.csv")
    name_map = {
        "双边同步领先": "双边同步领先",
        "用户领先商户落后：A总份额略占优": "用户先行 / A 略占优",
        "用户领先商户落后：B总份额略占优": "用户先行 / B 略占优",
        "商户领先用户落后：A总份额略占优": "商户先行 / A 略占优",
        "商户领先用户落后：B总份额略占优": "商户先行 / B 略占优",
        "双边同步落后": "双边同步落后",
    }
    labels = [f"{name_map.get(r['case'], r['case'])}\nu0={float(r['u0']):.4g}, m0={float(r['m0']):.4g}" for r in rows]
    shares = np.array([float(r["L_A"]) for r in rows])
    centered = shares - 0.5
    y = np.arange(len(rows))[::-1]
    colors = np.where(shares >= 0.5, GREEN, RED)

    fig, ax = plt.subplots(figsize=(10.2, 6.1), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.spines["left"].set_visible(False)
    ax.axvspan(-0.5, 0, color=RED, alpha=0.07, linewidth=0)
    ax.axvspan(0, 0.5, color=GREEN, alpha=0.07, linewidth=0)
    ax.axvline(0, color="#687070", lw=1.25, ls=(0, (4, 4)), alpha=0.7)
    ax.barh(y, centered, height=0.56, color=colors, alpha=0.92, edgecolor="none")
    ax.scatter(centered, y, s=62, color=colors, edgecolor=AX_BG, linewidth=1.3, zorder=3)
    for share, yy in zip(shares, y):
        ax.text(
            0.245 if share >= 0.5 else -0.245,
            yy,
            f"{'A 锁定' if share >= 0.5 else 'B 锁定'}\n$L_A$={share:.3f}",
            ha="center",
            va="center",
            fontsize=10.8,
            color="white",
            weight="bold",
            linespacing=1.2,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.7, linespacing=1.25)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.set_xlim(-0.56, 0.56)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels(["B 完全锁定", "B 优势", "均势线\nL_A=0.5", "A 优势", "A 完全锁定"], fontsize=10)
    ax.grid(axis="x", color=GRID, alpha=0.45, linewidth=0.85)
    ax.set_title("实验 3 补充：非同步初始优势的锁定方向", fontsize=17.5, pad=18, weight="bold")
    ax.set_xlabel(r"最终综合份额相对均势线的偏离  $L_A - 0.5$", fontsize=12.2, labelpad=12)
    ax.text(-0.5, len(rows) - 0.36, "平台 B 获胜区域", color="#A94844", fontsize=10.5, va="top")
    ax.text(0.5, len(rows) - 0.36, "平台 A 获胜区域", color="#247B46", fontsize=10.5, va="top", ha="right")
    save_report_figure(fig, fig_dir, "exp3_asymmetric_initial_cases")


def threshold_from(rows: list[dict[str, str]], x_key: str) -> float | None:
    for row in rows:
        success = int(float(row["success"])) == 1 if "success" in row else float(row["L_A"]) > 0.6
        if success:
            return float(row[x_key])
    return None


def plot_exp4_quality_threshold(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp4_quality_threshold.csv")
    x = np.array([float(r["delta_q"]) for r in rows])
    y = np.array([float(r["L_A"]) for r in rows])
    threshold = threshold_from(rows, "delta_q")

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.06, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    ax.plot(x, y, color=GREEN, lw=2.8, solid_capstyle="round")
    ax.fill_between(x, y, 0, color=GREEN, alpha=0.08)
    if threshold is not None:
        ax.axvline(threshold, color=ORANGE, lw=1.4, ls=(0, (4, 4)))
        ax.scatter([threshold], [0.6], s=72, color=ORANGE, edgecolor=AX_BG, linewidth=1.2, zorder=4)
        ax.text(threshold + 0.08, 0.64, f"逆袭阈值\nΔq*≈{threshold:.2f}", color="#B4671F", fontsize=10.7, va="bottom")
    ax.text(0.15, 0.08, "弱势平台锁定失败", color="#A94844", fontsize=10.5)
    ax.text(3.5, 0.86, "达到反转标准\n$L_A>0.6$", color="#247B46", fontsize=10.8, ha="center")
    ax.set_title("实验 4：服务质量优势与弱势平台逆袭阈值", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("服务质量优势  Δq", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    save_report_figure(fig, fig_dir, "exp4_quality_threshold")


def plot_exp4_quality_compare(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp4_quality_network_compare.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["setting"], []).append(row)
    colors = {"1.0": BLUE, "3.0": GREEN}

    fig, ax = plt.subplots(figsize=(8.9, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    for setting, items in groups.items():
        strength = items[0]["network_strength"]
        x = np.array([float(r["delta_q"]) for r in items])
        y = np.array([float(r["L_A"]) for r in items])
        threshold = threshold_from(items, "delta_q")
        label = f"α=β={float(strength):.1f}"
        if threshold is not None:
            label += f"  阈值≈{threshold:.2f}"
        ax.plot(x, y, color=colors.get(strength, ORANGE), lw=2.7, solid_capstyle="round", label=label)
        if threshold is not None:
            ax.scatter([threshold], [0.6], s=55, color=colors.get(strength, ORANGE), edgecolor=AX_BG, linewidth=1.1, zorder=4)
    ax.text(0.14, 0.73, "网络效应越弱\n质量优势越容易突破", color=BLUE, fontsize=10.5)
    ax.text(2.4, 0.38, "强网络效应下\n锁定更难逆转", color="#247B46", fontsize=10.5)
    ax.set_title("实验 4 补充：不同网络效应下的服务质量逆袭对比", fontsize=16.5, weight="bold", pad=16)
    ax.set_xlabel("服务质量优势  Δq", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=10.0)
    save_report_figure(fig, fig_dir, "exp4_quality_network_compare")


def plot_exp5_subsidy_dynamics(fig_dir: Path) -> None:
    params = PlatformParams()
    selected = [0.0, 0.5, 1.0, 1.5, 2.0]
    colors = ["#6E7885", BLUE, "#55A58F", ORANGE, GREEN]

    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.1, ls=(0, (4, 4)), alpha=0.65)
    for b, color in zip(selected, colors):
        p = params.with_updates(alpha=3.0, beta=3.0, bAU=b, bAM=b, bBU=0.0, bBM=0.0)
        t, u, m = run_simulation(p, u0=0.3, m0=0.3)
        share = 0.5 * (u + m)
        ax.plot(t, share, color=color, lw=2.55, solid_capstyle="round", label=f"b={b:.1f}")
    ax.text(1.5, 0.08, "无补贴：弱势锁定", color="#A94844", fontsize=10.5)
    ax.text(25, 0.83, "双边统一补贴推动份额跃迁", color="#247B46", fontsize=10.8)
    ax.set_title("实验 5：不同补贴强度下的动态演化", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("时间 t", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 综合份额  $L_A(t)$", fontsize=12.2, labelpad=10)
    ax.set_xlim(-0.5, 59)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(
        title="补贴强度",
        loc="center right",
        frameon=True,
        facecolor=AX_BG,
        edgecolor="#D7D0C2",
        framealpha=0.94,
        fontsize=9.7,
        title_fontsize=9.8,
    )
    save_report_figure(fig, fig_dir, "exp5_subsidy_dynamics")


def plot_exp5_subsidy_final(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp5_subsidy_basic.csv")
    x = np.array([float(r["b"]) for r in rows])
    y = np.array([float(r["L_A"]) for r in rows])
    threshold = threshold_from(rows, "b")

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    ax.plot(x, y, color=GREEN, lw=2.8, solid_capstyle="round")
    ax.fill_between(x, y, 0, color=GREEN, alpha=0.08)
    if threshold is not None:
        ax.axvline(threshold, color=ORANGE, lw=1.4, ls=(0, (4, 4)))
        ax.scatter([threshold], [0.6], s=72, color=ORANGE, edgecolor=AX_BG, linewidth=1.2, zorder=4)
        ax.text(threshold + 0.08, 0.64, f"补贴阈值\nb*≈{threshold:.2f}", color="#B4671F", fontsize=10.7, va="bottom")
    ax.set_title("实验 5：补贴强度与最终市场份额", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("统一补贴强度  b", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    save_report_figure(fig, fig_dir, "exp5_subsidy_final_share")


def plot_exp5_subsidy_strategy(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp5_subsidy_strategy_compare.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["strategy"], []).append(row)
    colors = {"只补贴用户": BLUE, "只补贴商户": ORANGE, "双边统一补贴": GREEN}

    fig, ax = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    for strategy, items in groups.items():
        x = np.array([float(r["b"]) for r in items])
        y = np.array([float(r["L_A"]) for r in items])
        threshold = threshold_from(items, "b")
        label = strategy if threshold is None else f"{strategy}  阈值≈{threshold:.2f}"
        ax.plot(x, y, color=colors.get(strategy, ORANGE), lw=2.7, solid_capstyle="round", label=label)
        if threshold is not None:
            ax.scatter([threshold], [0.6], s=55, color=colors.get(strategy, ORANGE), edgecolor=AX_BG, linewidth=1.1, zorder=4)
    ax.text(0.22, 0.36, "单边补贴：突破较慢", color=BLUE, fontsize=10.5)
    ax.text(2.0, 0.86, "双边统一补贴：协同效应更强", color="#247B46", fontsize=10.8)
    ax.set_title("实验 5 补充：不同基础补贴对象的效果对比", fontsize=16.7, weight="bold", pad=16)
    ax.set_xlabel("补贴强度  b", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=9.7)
    save_report_figure(fig, fig_dir, "exp5_subsidy_strategy_compare")


def main() -> None:
    setup_style()
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_exp1_share(fig_dir, "user")
    plot_exp1_share(fig_dir, "merchant")
    plot_exp2_heatmap(fig_dir, table_dir)
    plot_exp3_initial_advantage(fig_dir, table_dir)
    plot_exp3_asymmetric(fig_dir, table_dir)
    plot_exp4_quality_threshold(fig_dir, table_dir)
    plot_exp4_quality_compare(fig_dir, table_dir)
    plot_exp5_subsidy_dynamics(fig_dir)
    plot_exp5_subsidy_final(fig_dir, table_dir)
    plot_exp5_subsidy_strategy(fig_dir, table_dir)


if __name__ == "__main__":
    main()
