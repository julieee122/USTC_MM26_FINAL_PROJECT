from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.integrate import solve_ivp

PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_DIR / "阶段一基础实验"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from metrics import combined_share, concentration, directional_market_state, market_state  # noqa: E402
from model import PlatformParams, platform_dynamics, run_final_state, run_simulation  # noqa: E402
from plotting import save_figure  # noqa: E402


def classify_concentration(c: float) -> int:
    if c < 0.2:
        return 0
    if c < 0.8:
        return 1
    return 2


def exp6_lockin_region(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    alphas = np.linspace(0, 5, 51)
    betas = np.linspace(0, 5, 51)
    c_grid = np.zeros((len(betas), len(alphas)))
    class_grid = np.zeros_like(c_grid, dtype=int)

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            p = params.with_updates(alpha=float(alpha), beta=float(beta))
            u_inf, m_inf = run_final_state(p, u0=0.55, m0=0.55)
            c = concentration(u_inf, m_inf)
            c_grid[i, j] = c
            class_grid[i, j] = classify_concentration(c)

    plt.figure(figsize=(7.2, 5.6))
    im = plt.imshow(c_grid, origin="lower", extent=[0, 5, 0, 5], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="市场集中度 C")
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("实验6：市场集中度连续热力图")
    save_figure(fig_dir / "exp6_concentration_heatmap.png")

    cmap = ListedColormap(["#7FC97F", "#FDB462", "#E15759"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    plt.figure(figsize=(7.2, 5.6))
    im = plt.imshow(class_grid, origin="lower", extent=[0, 5, 0, 5], aspect="auto", cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["双平台共存", "市场倾斜", "市场锁定"])
    plt.xlabel("alpha")
    plt.ylabel("beta")
    plt.title("实验6：共存区、倾斜区与锁定区")
    save_figure(fig_dir / "exp6_lockin_region_classification.png")

    diag_rows = []
    k_star = None
    k_tilt = None
    diag_ks = []
    diag_cs = []
    for k in alphas:
        p = params.with_updates(alpha=float(k), beta=float(k))
        u_inf, m_inf = run_final_state(p, u0=0.55, m0=0.55)
        c = concentration(u_inf, m_inf)
        state = market_state(c)
        diag_ks.append(float(k))
        diag_cs.append(c)
        if c >= 0.2 and k_tilt is None:
            k_tilt = float(k)
        if c >= 0.8 and k_star is None:
            k_star = float(k)
        diag_rows.append([k, u_inf, m_inf, c, state])

    plt.figure(figsize=(7.4, 5))
    plt.plot(diag_ks, diag_cs, color="#4C78A8", linewidth=2)
    plt.axhline(0.2, color="#F58518", linestyle="--", linewidth=1.3, label="倾斜阈值 C=0.2")
    plt.axhline(0.8, color="#E45756", linestyle="--", linewidth=1.3, label="锁定阈值 C=0.8")
    if k_tilt is not None:
        plt.axvline(k_tilt, color="#F58518", linestyle=":", linewidth=1.2, label=f"共存→倾斜 k≈{k_tilt:.2f}")
    if k_star is not None:
        plt.axvline(k_star, color="#E45756", linestyle=":", linewidth=1.2, label=f"倾斜→锁定 k≈{k_star:.2f}")
    plt.xlabel("对称网络效应强度 k = alpha = beta")
    plt.ylabel("市场集中度 C")
    plt.title("实验6：对称网络效应路径下的临界跃迁")
    plt.ylim(-0.02, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp6_diagonal_k_critical_curve.png")

    _write_csv(table_dir / "exp6_concentration_grid.csv", [""] + [f"alpha={a:.1f}" for a in alphas], _matrix_rows(betas, c_grid, "beta"))
    _write_csv(table_dir / "exp6_classification_grid.csv", [""] + [f"alpha={a:.1f}" for a in alphas], _matrix_rows(betas, class_grid, "beta"))
    _write_csv(table_dir / "exp6_diagonal_k_scan.csv", ["k", "u_inf", "m_inf", "C", "state"], diag_rows)
    _write_csv(table_dir / "exp6_summary.csv", ["metric", "value"], [
        ["coexist_count", int((class_grid == 0).sum())],
        ["tilt_count", int((class_grid == 1).sum())],
        ["lockin_count", int((class_grid == 2).sum())],
        ["diagonal_tilt_threshold_k", "" if k_tilt is None else k_tilt],
        ["diagonal_lockin_threshold_k", "" if k_star is None else k_star],
    ])


def exp7_reversal_threshold(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    network_strengths = [0.5, 1.0, 2.0, 3.0, 4.0]
    switching_costs = [0.0, 0.5, 1.0, 1.5]
    dq_values = np.linspace(0, 5, 101)
    threshold_grid = np.full((len(switching_costs), len(network_strengths)), np.nan)
    rows = []

    for i, s in enumerate(switching_costs):
        for j, k in enumerate(network_strengths):
            q_star = np.nan
            u_star = np.nan
            m_star = np.nan
            for dq in dq_values:
                p = params.with_updates(alpha=k, beta=k, sU=s, sM=s, qAU=float(dq), qAM=float(dq), qBU=0.0, qBM=0.0)
                u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
                if u_inf > 0.5 and m_inf > 0.5:
                    q_star = float(dq)
                    u_star = u_inf
                    m_star = m_inf
                    break
            threshold_grid[i, j] = q_star
            rows.append([k, s, q_star, u_star, m_star])

    plt.figure(figsize=(7.2, 5.2))
    masked = np.ma.masked_invalid(threshold_grid)
    im = plt.imshow(masked, origin="lower", aspect="auto", cmap="magma_r", extent=[0.25, 4.25, -0.25, 1.75], vmin=0, vmax=5)
    plt.colorbar(im, label="最小质量优势 Δq*")
    plt.xticks(network_strengths)
    plt.yticks(switching_costs)
    plt.xlabel("网络效应强度 k = alpha = beta")
    plt.ylabel("切换成本 s = sU = sM")
    plt.title("实验7：逆袭阈值受网络效应和切换成本影响")
    for i, s in enumerate(switching_costs):
        for j, k in enumerate(network_strengths):
            text = "无" if np.isnan(threshold_grid[i, j]) else f"{threshold_grid[i, j]:.2f}"
            plt.text(k, s, text, ha="center", va="center", color="white" if not np.isnan(threshold_grid[i, j]) and threshold_grid[i, j] > 2.5 else "black", fontsize=8)
    save_figure(fig_dir / "exp7_reversal_threshold_heatmap.png")

    _write_csv(table_dir / "exp7_reversal_thresholds.csv", ["network_strength", "switching_cost", "delta_q_star", "u_at_star", "m_at_star"], rows)


def exp8_subsidy_allocation(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    rhos = np.linspace(0, 1, 101)
    budgets = np.round(np.arange(0.5, 3.01, 0.1), 2)
    rows = []
    l_grid = np.zeros((len(budgets), len(rhos)))

    for i, budget in enumerate(budgets):
        values = []
        best = (-1.0, None)
        for j, rho in enumerate(rhos):
            b_u = float(rho * budget)
            b_m = float((1.0 - rho) * budget)
            p = params.with_updates(alpha=3.0, beta=3.0, bAU=b_u, bAM=b_m, bBU=0.0, bBM=0.0)
            u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
            l_a = combined_share(u_inf, m_inf)
            c = concentration(u_inf, m_inf)
            values.append(l_a)
            l_grid[i, j] = l_a
            if l_a > best[0]:
                best = (l_a, float(rho))
            rows.append([budget, rho, b_u, b_m, u_inf, m_inf, l_a, c, directional_market_state(u_inf, m_inf)])

    selected_budgets = [1.4, 1.5, 1.6, 1.7, 1.8]
    plt.figure(figsize=(8, 5))
    for budget in selected_budgets:
        idx = int(np.argmin(np.abs(budgets - budget)))
        best_idx = int(np.argmax(l_grid[idx]))
        value_range = float(np.max(l_grid[idx]) - np.min(l_grid[idx]))
        if value_range < 0.05:
            label = f"B={budgets[idx]:.1f}, 对 rho 不敏感"
        else:
            label = f"B={budgets[idx]:.1f}, 最优 rho={rhos[best_idx]:.2f}"
        plt.plot(rhos, l_grid[idx], label=label)
    plt.xlabel("用户补贴比例 rho")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验8：临界预算附近的双边补贴分配")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp8_subsidy_allocation_critical.png")

    plt.figure(figsize=(7.6, 5.4))
    im = plt.imshow(l_grid, origin="lower", extent=[0, 1, float(budgets[0]), float(budgets[-1])], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="最终综合份额 L_A")
    plt.xlabel("用户补贴比例 rho")
    plt.ylabel("总补贴预算 B")
    plt.title("实验8：预算与补贴比例的二维效果图")
    save_figure(fig_dir / "exp8_budget_rho_heatmap.png")

    scenario_rows = []
    scenario_budget = 1.5
    scenarios = [
        ("对称网络效应", 3.0, 3.0),
        ("用户更依赖商户", 4.0, 2.0),
        ("商户更依赖用户", 2.0, 4.0),
    ]
    plt.figure(figsize=(8, 5))
    for label, alpha, beta in scenarios:
        values = []
        best = (-1.0, None, None, None, None)
        for rho in rhos:
            b_u = float(rho * scenario_budget)
            b_m = float((1.0 - rho) * scenario_budget)
            p = params.with_updates(alpha=alpha, beta=beta, bAU=b_u, bAM=b_m, bBU=0.0, bBM=0.0)
            u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
            l_a = combined_share(u_inf, m_inf)
            c = concentration(u_inf, m_inf)
            values.append(l_a)
            if l_a > best[0]:
                best = (l_a, float(rho), u_inf, m_inf, c)
            scenario_rows.append([label, alpha, beta, scenario_budget, rho, b_u, b_m, u_inf, m_inf, l_a, c, directional_market_state(u_inf, m_inf)])
        plt.plot(rhos, values, label=f"{label}，最优 rho={best[1]:.2f}")
    plt.xlabel("用户补贴比例 rho")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验8补充：不同网络效应结构下的补贴比例")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp8_network_scenario_allocation.png")

    _write_csv(table_dir / "exp8_subsidy_allocation.csv", ["budget", "rho", "bAU", "bAM", "u_inf", "m_inf", "L_A", "C", "state"], rows)
    _write_csv(table_dir / "exp8_budget_rho_grid.csv", [""] + [f"rho={rho:.2f}" for rho in rhos], _matrix_rows(budgets, l_grid, "B"))
    _write_csv(
        table_dir / "exp8_network_scenario_allocation.csv",
        ["scenario", "alpha", "beta", "budget", "rho", "bAU", "bAM", "u_inf", "m_inf", "L_A", "C", "state"],
        scenario_rows,
    )

    summary_rows = []
    for i, budget in enumerate(budgets):
        best_idx = int(np.argmax(l_grid[i]))
        success = l_grid[i, best_idx] > 0.6
        summary_rows.append([budget, rhos[best_idx], l_grid[i, best_idx], int(success)])
    for label, alpha, beta in scenarios:
        scenario_data = [row for row in scenario_rows if row[0] == label]
        best_row = max(scenario_data, key=lambda row: row[9])
        summary_rows.append([f"{label}(B=1.5)", best_row[4], best_row[9], int(best_row[9] > 0.6)])
    _write_csv(table_dir / "exp8_summary.csv", ["case_or_budget", "best_rho", "best_L_A", "success"], summary_rows)


def exp9_subsidy_exit(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    b0_values = np.linspace(0, 5, 41)
    ts_values = np.linspace(0, 30, 31)
    l_grid = np.zeros((len(ts_values), len(b0_values)))
    rows = []

    for i, ts in enumerate(ts_values):
        for j, b0 in enumerate(b0_values):
            t, u, m = run_simulation_with_exit_subsidy(params, b0=float(b0), ts=float(ts), u0=0.3, m0=0.3, T=60.0, n_points=500)
            l_a = combined_share(float(u[-1]), float(m[-1]))
            l_grid[i, j] = l_a
            u_inf = float(u[-1])
            m_inf = float(m[-1])
            c = concentration(u_inf, m_inf)
            rows.append([b0, ts, u_inf, m_inf, l_a, c, directional_market_state(u_inf, m_inf)])

    plt.figure(figsize=(7.4, 5.4))
    im = plt.imshow(l_grid, origin="lower", extent=[0, 5, 0, 30], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="最终综合份额 L_A")
    plt.xlabel("补贴强度 b0")
    plt.ylabel("补贴持续时间 Ts")
    plt.title("实验9：阶段性补贴退出后的最终份额")
    save_figure(fig_dir / "exp9_subsidy_exit_heatmap.png")

    _write_csv(table_dir / "exp9_subsidy_exit.csv", ["b0", "Ts", "u_inf", "m_inf", "L_A", "C", "state"], rows)
    _write_csv(table_dir / "exp9_subsidy_exit_grid.csv", [""] + [f"b0={b:.2f}" for b in b0_values], _matrix_rows(ts_values, l_grid, "Ts"))


def exp10_profit_constraint(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    subsidies = np.linspace(0, 5, 101)
    revenue_rate = 3.0
    fixed_cost = 0.05
    rows = []
    shares = []
    profits = []

    for b in subsidies:
        p = params.with_updates(alpha=3.0, beta=3.0, bAU=float(b), bAM=float(b), bBU=0.0, bBM=0.0)
        t, u, m = run_simulation(p, u0=0.3, m0=0.3, T=50.0, n_points=600)
        profit_t = revenue_rate * u * m - b * u - b * m - fixed_cost
        total_profit = float(np.trapezoid(profit_t, t))
        l_a = combined_share(float(u[-1]), float(m[-1]))
        shares.append(l_a)
        profits.append(total_profit)
        rows.append([b, u[-1], m[-1], l_a, total_profit])

    best_idx = int(np.argmax(profits))
    share_threshold_idx = next((idx for idx, value in enumerate(shares) if value > 0.6), None)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(subsidies, shares, color="#2CA02C", label="最终综合份额 L_A")
    ax1.set_xlabel("统一补贴强度 b")
    ax1.set_ylabel("最终综合份额 L_A", color="#2CA02C")
    ax1.tick_params(axis="y", labelcolor="#2CA02C")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(subsidies, profits, color="#D62728", label="累计利润")
    ax2.set_ylabel("累计利润", color="#D62728")
    ax2.tick_params(axis="y", labelcolor="#D62728")
    ax2.axvline(subsidies[best_idx], color="#D62728", linestyle="--", linewidth=1, label=f"利润最优 b={subsidies[best_idx]:.2f}")
    if share_threshold_idx is not None:
        ax1.axvline(subsidies[share_threshold_idx], color="#2CA02C", linestyle=":", linewidth=1, label=f"份额突破 b={subsidies[share_threshold_idx]:.2f}")
    plt.title("实验10：补贴强度、市场份额与累计利润")
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    save_figure(fig_dir / "exp10_profit_constraint.png")

    _write_csv(table_dir / "exp10_profit_constraint.csv", ["b", "u_inf", "m_inf", "L_A", "profit_total"], rows)
    _write_csv(table_dir / "exp10_summary.csv", ["metric", "value"], [
        ["revenue_rate_R", revenue_rate],
        ["fixed_cost_c", fixed_cost],
        ["profit_optimal_b", float(subsidies[best_idx])],
        ["profit_optimal_value", float(profits[best_idx])],
        ["share_breakthrough_b", "" if share_threshold_idx is None else float(subsidies[share_threshold_idx])],
    ])


def run_simulation_with_exit_subsidy(
    params: PlatformParams,
    b0: float,
    ts: float,
    u0: float,
    m0: float,
    T: float = 60.0,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_params = params.with_updates(alpha=3.0, beta=3.0, bBU=0.0, bBM=0.0)
    t_eval = np.linspace(0.0, T, n_points)

    def dynamics(t: float, y: np.ndarray) -> list[float]:
        subsidy = b0 if t <= ts else 0.0
        p = base_params.with_updates(bAU=subsidy, bAM=subsidy)
        return platform_dynamics(t, y, p)

    sol = solve_ivp(
        fun=dynamics,
        t_span=(0.0, T),
        y0=[u0, m0],
        t_eval=t_eval,
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    return sol.t, np.clip(sol.y[0], 0.0, 1.0), np.clip(sol.y[1], 0.0, 1.0)


def _matrix_rows(index_values: np.ndarray | list[float], matrix: np.ndarray, index_name: str) -> list[list[object]]:
    rows = []
    for value, row in zip(index_values, matrix):
        rows.append([f"{index_name}={float(value):.2f}", *[float(x) for x in row]])
    return rows


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
