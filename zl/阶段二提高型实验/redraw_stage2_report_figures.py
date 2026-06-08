from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_DIR / "阶段一基础实验"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from model import PlatformParams, run_simulation  # noqa: E402


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


def matrix_csv(path: Path, prefix: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        xs = np.array([float(item.split("=")[1]) for item in header[1:]])
        ys: list[float] = []
        values: list[list[float]] = []
        for row in reader:
            ys.append(float(row[0].split("=")[1]))
            values.append([float(v) for v in row[1:]])
    return xs, np.array(ys), np.array(values)


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


def concentration_cmap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "concentration",
        ["#25324D", "#315F7B", "#3A9C8B", "#8FCF73", "#F1D66B", "#F6A64F"],
    )


def plot_exp6_concentration(fig_dir: Path, table_dir: Path) -> None:
    alpha, beta, grid = matrix_csv(table_dir / "exp6_concentration_grid.csv", "alpha")
    aa, bb = np.meshgrid(alpha, beta)
    fig, ax = plt.subplots(figsize=(8.6, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)

    heatmap = ax.contourf(aa, bb, grid, levels=np.linspace(0, 1, 101), cmap=concentration_cmap(), vmin=0, vmax=1)
    contours = ax.contour(aa, bb, grid, levels=[0.2, 0.5, 0.8], colors=["#E6EAE6", "#FFFFFF", "#5D4A36"], linewidths=[1.1, 1.4, 1.25])
    ax.clabel(contours, inline=True, fmt={0.2: "C=0.2", 0.5: "C=0.5", 0.8: "C=0.8"}, fontsize=9.2)
    ax.scatter([1.2], [1.2], s=62, color="#FFFFFF", edgecolor=INK, linewidth=1.4, zorder=4)
    ax.text(1.3, 1.28, "锁定阈值附近\nk≈1.2", color=INK, fontsize=10.2, va="bottom")
    ax.text(0.45, 0.45, "共存区", color="#F1F5F0", fontsize=12.5, weight="bold", ha="center")
    ax.text(3.8, 4.2, "高集中度\n市场锁定", color="#5B3B18", fontsize=12.0, weight="bold", ha="center")
    ax.set_title("实验 6：市场集中度连续热力图", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("用户侧网络效应强度  α", fontsize=12.2, labelpad=10)
    ax.set_ylabel("商户侧网络效应强度  β", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(0, 5.1, 1))
    ax.set_yticks(np.arange(0, 5.1, 1))
    ax.grid(color="#FFFFFF", alpha=0.16, linewidth=0.8)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035, ticks=np.linspace(0, 1, 6))
    cbar.set_label("市场集中度  C", fontsize=11.8, labelpad=10)
    cbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp6_concentration_heatmap")


def plot_exp6_classification(fig_dir: Path, table_dir: Path) -> None:
    alpha, beta, grid = matrix_csv(table_dir / "exp6_classification_grid.csv", "alpha")
    colors = ["#6EA37B", "#E9B65F", "#D95F59"]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    im = ax.imshow(grid, origin="lower", extent=[alpha[0], alpha[-1], beta[0], beta[-1]], aspect="equal", cmap=cmap, norm=norm)
    aa, bb = np.meshgrid(alpha, beta)
    ax.contour(aa, bb, grid, levels=[0.5, 1.5], colors="#FFFFFF", linewidths=1.6, alpha=0.9)
    ax.text(0.65, 0.55, "双平台共存", color="white", fontsize=12, weight="bold", ha="center")
    ax.text(1.35, 1.05, "市场倾斜", color="#5D3B14", fontsize=12, weight="bold", ha="center")
    ax.text(3.65, 3.9, "市场锁定", color="white", fontsize=12, weight="bold", ha="center")
    ax.set_title("实验 6：共存区、倾斜区与锁定区", fontsize=17.2, weight="bold", pad=16)
    ax.set_xlabel("用户侧网络效应强度  α", fontsize=12.2, labelpad=10)
    ax.set_ylabel("商户侧网络效应强度  β", fontsize=12.2, labelpad=10)
    ax.set_xticks(np.arange(0, 5.1, 1))
    ax.set_yticks(np.arange(0, 5.1, 1))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["共存", "倾斜", "锁定"])
    cbar.set_label("区域类型", fontsize=11.8, labelpad=10)
    cbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp6_lockin_region_classification")


def plot_exp6_diagonal(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp6_diagonal_k_scan.csv")
    k = np.array([float(r["k"]) for r in rows])
    c = np.array([float(r["C"]) for r in rows])
    tilt = next((float(r["k"]) for r in rows if float(r["C"]) >= 0.2), None)
    lock = next((float(r["k"]) for r in rows if float(r["C"]) >= 0.8), None)

    fig, ax = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.0, 0.2, color=BLUE, alpha=0.07, linewidth=0)
    ax.axhspan(0.2, 0.8, color=ORANGE, alpha=0.08, linewidth=0)
    ax.axhspan(0.8, 1.05, color=RED, alpha=0.07, linewidth=0)
    ax.plot(k, c, color=GREEN, lw=2.8, solid_capstyle="round")
    ax.axhline(0.2, color=ORANGE, lw=1.2, ls=(0, (4, 4)))
    ax.axhline(0.8, color=RED, lw=1.2, ls=(0, (4, 4)))
    if tilt is not None:
        ax.axvline(tilt, color=ORANGE, lw=1.2, ls=(0, (2, 4)))
        ax.text(tilt + 0.07, 0.25, f"倾斜阈值\nk≈{tilt:.2f}", color="#B4671F", fontsize=10.3)
    if lock is not None:
        ax.axvline(lock, color=RED, lw=1.2, ls=(0, (2, 4)))
        ax.text(lock + 0.07, 0.83, f"锁定阈值\nk≈{lock:.2f}", color="#A94844", fontsize=10.3, va="bottom")
    ax.set_title("实验 6：对称网络效应路径下的临界跃迁", fontsize=17.0, weight="bold", pad=16)
    ax.set_xlabel(r"对称网络效应强度  $k = \alpha = \beta$", fontsize=12.2, labelpad=10)
    ax.set_ylabel("市场集中度  C", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    save_report_figure(fig, fig_dir, "exp6_diagonal_k_critical_curve")


def plot_exp7_threshold(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp7_reversal_thresholds.csv")
    ks = sorted({float(r["network_strength"]) for r in rows})
    ss = sorted({float(r["switching_cost"]) for r in rows})
    grid = np.full((len(ss), len(ks)), np.nan)
    for row in rows:
        i = ss.index(float(row["switching_cost"]))
        j = ks.index(float(row["network_strength"]))
        grid[i, j] = float(row["delta_q_star"]) if row["delta_q_star"] else np.nan

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    cmap = mcolors.LinearSegmentedColormap.from_list("threshold", ["#F4E6A1", "#F0A45A", "#A94844", "#4A2B47"])
    im = ax.imshow(grid, origin="lower", extent=[min(ks) - 0.5, max(ks) + 0.5, min(ss) - 0.25, max(ss) + 0.25], aspect="auto", cmap=cmap, vmin=0, vmax=np.nanmax(grid))
    for i, s in enumerate(ss):
        for j, k_value in enumerate(ks):
            value = grid[i, j]
            text_color = "white" if value >= 1.2 else "#3D3325"
            ax.text(k_value, s, f"{value:.2f}", ha="center", va="center", fontsize=10.2, color=text_color, weight="bold")
    ax.set_title("实验 7：网络效应与切换成本下的逆袭阈值", fontsize=16.8, weight="bold", pad=16)
    ax.set_xlabel(r"网络效应强度  $k = \alpha = \beta$", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"切换成本  $s = s_U = s_M$", fontsize=12.2, labelpad=10)
    ax.set_xticks(ks)
    ax.set_yticks(ss)
    ax.grid(color="#FFFFFF", alpha=0.25, linewidth=1.0)
    ax.text(0.62, -0.13, "阈值低，更易逆袭", color="#4F3A1D", fontsize=10.2, weight="bold", ha="center")
    ax.text(3.35, -0.13, "网络效应越强，所需质量优势越高", color="#5D2B2E", fontsize=10.2, weight="bold", ha="center")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"最小质量优势  $\Delta q^*$", fontsize=11.8, labelpad=10)
    cbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp7_reversal_threshold_heatmap")


def plot_exp8_allocation_critical(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp8_subsidy_allocation.csv")
    selected = [1.4, 1.5, 1.6, 1.7, 1.8]
    colors = ["#6E7885", BLUE, GREEN, ORANGE, RED]
    fig, ax = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhspan(0.6, 1.0, color=GREEN, alpha=0.055, linewidth=0)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    for budget, color in zip(selected, colors):
        items = [r for r in rows if abs(float(r["budget"]) - budget) < 1e-9]
        rho = np.array([float(r["rho"]) for r in items])
        share = np.array([float(r["L_A"]) for r in items])
        best_idx = int(np.argmax(share))
        ax.plot(rho, share, color=color, lw=2.5, solid_capstyle="round", label=f"B={budget:.1f}, 最优ρ={rho[best_idx]:.2f}")
    ax.text(0.1, 0.08, "预算不足：分配比例影响有限", color="#A94844", fontsize=10.5)
    ax.text(0.58, 0.84, "临界预算附近\n比例选择变得关键", color="#247B46", fontsize=10.8, ha="center")
    ax.set_title("实验 8：临界预算附近的双边补贴分配", fontsize=16.8, weight="bold", pad=16)
    ax.set_xlabel(r"用户补贴比例  $\rho$", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=9.4)
    save_report_figure(fig, fig_dir, "exp8_subsidy_allocation_critical")


def plot_exp8_budget_rho(fig_dir: Path, table_dir: Path) -> None:
    rho, budgets, grid = matrix_csv(table_dir / "exp8_budget_rho_grid.csv", "rho")
    rr, bb = np.meshgrid(rho, budgets)
    fig, ax = plt.subplots(figsize=(8.6, 6.1), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    heatmap = ax.contourf(rr, bb, grid, levels=np.linspace(0, 1, 101), cmap=concentration_cmap(), vmin=0, vmax=1)
    contours = ax.contour(rr, bb, grid, levels=[0.6, 0.9], colors=["#FFFFFF", "#5D4A36"], linewidths=[1.45, 1.15])
    ax.clabel(contours, inline=True, fmt={0.6: r"$L_A=0.6$", 0.9: r"$L_A=0.9$"}, fontsize=9.2)
    ax.text(0.5, 2.45, "高预算区\n锁定成功", color="#5B3B18", fontsize=11.5, weight="bold", ha="center")
    ax.text(0.18, 0.82, "低预算区", color="#F1F5F0", fontsize=11.0, weight="bold", ha="center")
    ax.set_title("实验 8：预算与补贴比例的二维效果图", fontsize=16.8, weight="bold", pad=16)
    ax.set_xlabel(r"用户补贴比例  $\rho$", fontsize=12.2, labelpad=10)
    ax.set_ylabel("总补贴预算  B", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(float(budgets[0]), float(budgets[-1]))
    ax.grid(color="#FFFFFF", alpha=0.16, linewidth=0.8)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035, ticks=np.linspace(0, 1, 6))
    cbar.set_label(r"最终综合份额  $L_A$", fontsize=11.8, labelpad=10)
    cbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp8_budget_rho_heatmap")


def plot_exp8_scenarios(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp8_network_scenario_allocation.csv")
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["scenario"], []).append(row)
    label_map = {
        "对称网络效应": "对称网络效应",
        "用户更依赖商户": "用户更依赖商户",
        "商户更依赖用户": "商户更依赖用户",
    }
    colors = [BLUE, GREEN, ORANGE]
    fig, ax = plt.subplots(figsize=(9.0, 5.7), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    ax.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    for (scenario, items), color in zip(groups.items(), colors):
        rho = np.array([float(r["rho"]) for r in items])
        share = np.array([float(r["L_A"]) for r in items])
        best_idx = int(np.argmax(share))
        ax.plot(rho, share, color=color, lw=2.7, solid_capstyle="round", label=f"{label_map.get(scenario, scenario)}  最优ρ={rho[best_idx]:.2f}")
        ax.scatter([rho[best_idx]], [share[best_idx]], s=52, color=color, edgecolor=AX_BG, linewidth=1.1, zorder=4)
    ax.text(0.16, 0.13, "对称情形：预算不足时难以突破", color=BLUE, fontsize=10.4)
    ax.text(0.18, 0.86, "结构性不对称会改变最优补贴侧重", color="#247B46", fontsize=10.7)
    ax.set_title("实验 8 补充：不同网络效应结构下的补贴比例", fontsize=16.6, weight="bold", pad=16)
    ax.set_xlabel(r"用户补贴比例  $\rho$", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"平台 A 最终综合份额  $L_A$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, alpha=0.34, linewidth=0.85)
    ax.legend(loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=9.4)
    save_report_figure(fig, fig_dir, "exp8_network_scenario_allocation")


def plot_exp9_exit(fig_dir: Path, table_dir: Path) -> None:
    b0, ts, grid = matrix_csv(table_dir / "exp9_subsidy_exit_grid.csv", "b0")
    bb, tt = np.meshgrid(b0, ts)
    fig, ax = plt.subplots(figsize=(8.6, 6.1), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax)
    heatmap = ax.contourf(bb, tt, grid, levels=np.linspace(0, 1, 101), cmap=concentration_cmap(), vmin=0, vmax=1)
    contours = ax.contour(bb, tt, grid, levels=[0.6, 0.9], colors=["#FFFFFF", "#5D4A36"], linewidths=[1.45, 1.15])
    ax.clabel(contours, inline=True, fmt={0.6: r"$L_A=0.6$", 0.9: r"$L_A=0.9$"}, fontsize=9.2)
    ax.text(1.1, 5.0, "退出过早\n锁定失败", color="#F1F5F0", fontsize=11.0, weight="bold", ha="center")
    ax.text(3.7, 23.0, "补贴足够强且持续足够久\n退出后仍能保持优势", color="#5B3B18", fontsize=11.0, weight="bold", ha="center")
    ax.set_title("实验 9：阶段性补贴退出后的最终份额", fontsize=16.8, weight="bold", pad=16)
    ax.set_xlabel(r"补贴强度  $b_0$", fontsize=12.2, labelpad=10)
    ax.set_ylabel(r"补贴持续时间  $T_s$", fontsize=12.2, labelpad=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 30)
    ax.grid(color="#FFFFFF", alpha=0.16, linewidth=0.8)
    cbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.035, ticks=np.linspace(0, 1, 6))
    cbar.set_label(r"最终综合份额  $L_A$", fontsize=11.8, labelpad=10)
    cbar.outline.set_edgecolor("#B8B0A5")
    save_report_figure(fig, fig_dir, "exp9_subsidy_exit_heatmap")


def plot_exp10_profit(fig_dir: Path, table_dir: Path) -> None:
    rows = read_csv(table_dir / "exp10_profit_constraint.csv")
    b = np.array([float(r["b"]) for r in rows])
    share = np.array([float(r["L_A"]) for r in rows])
    profit = np.array([float(r["profit_total"]) for r in rows])
    best_idx = int(np.argmax(profit))
    breakthrough_idx = next((i for i, value in enumerate(share) if value > 0.6), None)

    fig, ax1 = plt.subplots(figsize=(9.0, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(FIG_BG)
    finish_axes(ax1)
    ax1.axhspan(0.6, 1.05, color=GREEN, alpha=0.055, linewidth=0)
    ax1.axhline(0.6, color="#687070", lw=1.15, ls=(0, (4, 4)), alpha=0.7)
    line_share = ax1.plot(b, share, color=GREEN, lw=2.8, solid_capstyle="round", label=r"最终份额 $L_A$")
    ax1.set_xlabel("统一补贴强度  b", fontsize=12.2, labelpad=10)
    ax1.set_ylabel(r"最终综合份额  $L_A$", color=GREEN, fontsize=12.2, labelpad=10)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(0, 5)
    ax1.grid(color=GRID, alpha=0.34, linewidth=0.85)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#A9A197")
    line_profit = ax2.plot(b, profit, color=RED, lw=2.4, solid_capstyle="round", label="累计利润")
    ax2.set_ylabel("累计利润", color=RED, fontsize=12.2, labelpad=10)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax2.axvline(b[best_idx], color=RED, lw=1.25, ls=(0, (4, 4)))
    ax2.scatter([b[best_idx]], [profit[best_idx]], s=68, color=RED, edgecolor=AX_BG, linewidth=1.2, zorder=5)
    ax2.text(b[best_idx] + 0.12, profit[best_idx], f"利润最优\nb≈{b[best_idx]:.2f}", color="#A94844", fontsize=10.4, va="center")
    if breakthrough_idx is not None:
        ax1.axvline(b[breakthrough_idx], color=GREEN, lw=1.25, ls=(0, (2, 4)))
        ax1.scatter([b[breakthrough_idx]], [0.6], s=60, color=GREEN, edgecolor=AX_BG, linewidth=1.1, zorder=5)
    ax1.set_title("实验 10：补贴强度、市场份额与累计利润", fontsize=16.8, weight="bold", pad=16)
    ax1.legend(line_share + line_profit, [l.get_label() for l in line_share + line_profit], loc="lower right", frameon=True, facecolor=AX_BG, edgecolor="#D7D0C2", framealpha=0.94, fontsize=10)
    save_report_figure(fig, fig_dir, "exp10_profit_constraint")


def main() -> None:
    setup_style()
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_exp6_concentration(fig_dir, table_dir)
    plot_exp6_classification(fig_dir, table_dir)
    plot_exp6_diagonal(fig_dir, table_dir)
    plot_exp7_threshold(fig_dir, table_dir)
    plot_exp8_allocation_critical(fig_dir, table_dir)
    plot_exp8_budget_rho(fig_dir, table_dir)
    plot_exp8_scenarios(fig_dir, table_dir)
    plot_exp9_exit(fig_dir, table_dir)
    plot_exp10_profit(fig_dir, table_dir)


if __name__ == "__main__":
    main()
