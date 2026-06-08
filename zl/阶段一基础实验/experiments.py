from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from metrics import combined_share, concentration, directional_market_state, market_state
from model import PlatformParams, run_final_state, run_simulation
from plotting import save_figure


def exp1_basic_dynamics(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    cases = [
        ("对称初始 u0=m0=0.50", 0.50, 0.50),
        ("轻微领先 u0=m0=0.55", 0.55, 0.55),
        ("明显领先 u0=m0=0.70", 0.70, 0.70),
    ]
    rows = []

    plt.figure(figsize=(8, 5))
    for label, u0, m0 in cases:
        t, u, m = run_simulation(params, u0=u0, m0=m0)
        plt.plot(t, u, label=label)
        rows.append([label, u0, m0, u[-1], m[-1], concentration(u[-1], m[-1]), market_state(concentration(u[-1], m[-1]))])
    plt.xlabel("时间 t")
    plt.ylabel("平台 A 用户份额 u(t)")
    plt.title("实验1：平台 A 用户份额动态演化")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp1_user_share_dynamics.png")

    plt.figure(figsize=(8, 5))
    for label, u0, m0 in cases:
        t, u, m = run_simulation(params, u0=u0, m0=m0)
        plt.plot(t, m, label=label)
    plt.xlabel("时间 t")
    plt.ylabel("平台 A 商户份额 m(t)")
    plt.title("实验1：平台 A 商户份额动态演化")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp1_merchant_share_dynamics.png")

    _write_csv(table_dir / "exp1_final_states.csv", ["case", "u0", "m0", "u_inf", "m_inf", "C", "state"], rows)


def exp2_network_effect(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    alphas = np.linspace(0, 5, 51)
    betas = np.linspace(0, 5, 51)
    c_grid = np.zeros((len(betas), len(alphas)))
    lockin_grid = np.zeros_like(c_grid)

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            p = params.with_updates(alpha=float(alpha), beta=float(beta))
            u_inf, m_inf = run_final_state(p, u0=0.55, m0=0.55)
            c_grid[i, j] = concentration(u_inf, m_inf)
            lockin_grid[i, j] = 1.0 if (u_inf > 0.8 and m_inf > 0.8) else 0.0

    plt.figure(figsize=(7, 5.6))
    im = plt.imshow(c_grid, origin="lower", extent=[0, 5, 0, 5], aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="市场集中度 C")
    plt.xlabel("用户侧受商户吸引强度 alpha")
    plt.ylabel("商户侧受用户吸引强度 beta")
    plt.title("实验2：双边网络效应强度与市场集中度")
    save_figure(fig_dir / "exp2_network_effect_heatmap.png")

    np.savetxt(table_dir / "exp2_concentration_grid.csv", c_grid, delimiter=",", fmt="%.6f")
    np.savetxt(table_dir / "exp2_A_lockin_grid.csv", lockin_grid, delimiter=",", fmt="%.0f")


def exp3_initial_advantage(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    xs = np.linspace(0.1, 0.9, 81)
    settings = [("弱网络效应 alpha=beta=0.5", 0.5), ("强网络效应 alpha=beta=3.0", 3.0)]
    rows = []

    plt.figure(figsize=(8, 5))
    for label, k in settings:
        l_values = []
        for x in xs:
            p = params.with_updates(alpha=k, beta=k)
            u_inf, m_inf = run_final_state(p, u0=float(x), m0=float(x))
            l_a = combined_share(u_inf, m_inf)
            l_values.append(l_a)
            rows.append([label, x, u_inf, m_inf, l_a])
        plt.plot(xs, l_values, label=label)
    plt.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("初始份额 x = u_A(0) = m_A(0)")
    plt.ylabel("最终综合份额 L_A")
    plt.title("实验3：初始规模差异与最终市场份额")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp3_initial_advantage.png")

    _write_csv(table_dir / "exp3_initial_advantage.csv", ["setting", "x", "u_inf", "m_inf", "L_A"], rows)

    asymmetric_cases = [
        ("双边同步领先", 0.60, 0.60),
        ("用户领先商户落后：A总份额略占优", 0.7001, 0.3001),
        ("用户领先商户落后：B总份额略占优", 0.6999, 0.2999),
        ("商户领先用户落后：A总份额略占优", 0.3001, 0.7001),
        ("商户领先用户落后：B总份额略占优", 0.2999, 0.6999),
        ("双边同步落后", 0.40, 0.40),
    ]
    asymmetric_rows = []
    labels = []
    l_values = []

    for label, u0, m0 in asymmetric_cases:
        p = params.with_updates(alpha=3.0, beta=3.0)
        u_inf, m_inf = run_final_state(p, u0=u0, m0=m0)
        l_a = combined_share(u_inf, m_inf)
        c = concentration(u_inf, m_inf)
        labels.append(label)
        l_values.append(l_a)
        asymmetric_rows.append([label, u0, m0, u_inf, m_inf, l_a, c, directional_market_state(u_inf, m_inf)])

    plt.figure(figsize=(9, 5))
    bars = plt.bar(np.arange(len(labels)), l_values, color=["#4C78A8", "#F58518", "#F58518", "#54A24B", "#54A24B", "#B279A2"])
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.ylabel("最终综合份额 L_A")
    plt.title("实验3补充：非同步初始优势的代表性对比")
    plt.ylim(0, 1.05)
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, l_values):
        plt.text(bar.get_x() + bar.get_width() / 2, min(value + 0.03, 1.02), f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig_dir / "exp3_asymmetric_initial_cases.png")

    _write_csv(
        table_dir / "exp3_asymmetric_initial_cases.csv",
        ["case", "u0", "m0", "u_inf", "m_inf", "L_A", "C", "state"],
        asymmetric_rows,
    )

    # Scan the full initial-condition plane to identify whether the asymmetric
    # points are isolated thresholds or part of a basin boundary.
    p_strong = params.with_updates(alpha=3.0, beta=3.0)
    grid_values = np.linspace(0.1, 0.9, 81)
    basin_grid = np.zeros((len(grid_values), len(grid_values)))
    for i, m0 in enumerate(grid_values):
        for j, u0 in enumerate(grid_values):
            u_inf, m_inf = run_final_state(p_strong, u0=float(u0), m0=float(m0))
            basin_grid[i, j] = combined_share(u_inf, m_inf)

    plt.figure(figsize=(7.2, 5.8))
    im = plt.imshow(
        basin_grid,
        origin="lower",
        extent=[0.1, 0.9, 0.1, 0.9],
        aspect="auto",
        cmap="coolwarm",
        vmin=0,
        vmax=1,
    )
    plt.colorbar(im, label="平台 A 最终平均份额 L_A")
    boundary_x = np.linspace(0.1, 0.9, 200)
    plt.plot(boundary_x, 1 - boundary_x, color="black", linestyle="--", linewidth=1.3, label=r"$u_A(0)+m_A(0)=1$")
    plt.scatter([0.7], [0.3], color="yellow", edgecolor="black", zorder=3, label="(0.7, 0.3)")
    plt.xlabel(r"初始用户份额 $u_A(0)$")
    plt.ylabel(r"初始商户份额 $m_A(0)$")
    plt.grid(alpha=0.18)
    plt.legend(loc="upper right", fontsize=8)
    save_figure(fig_dir / "exp3_initial_condition_basin_heatmap.png")

    critical_cases = []
    boundary_cases = [
        ("反对角线点", 0.6, 0.4),
        ("反对角线点", 0.7, 0.3),
        ("反对角线点", 0.8, 0.2),
        ("反对角线点", 0.3, 0.7),
        ("A侧微小扰动", 0.7001, 0.3001),
        ("B侧微小扰动", 0.6999, 0.2999),
        ("A侧微小扰动", 0.8001, 0.2001),
        ("B侧微小扰动", 0.7999, 0.1999),
    ]
    for label, u0, m0 in boundary_cases:
        u_inf, m_inf = run_final_state(p_strong, u0=u0, m0=m0)
        l_a = combined_share(u_inf, m_inf)
        total = u0 + m0
        if abs(total - 1.0) < 1e-12:
            exact_state = "临界分界线：精确对称模型下收敛到共存均衡"
        elif total > 1.0:
            exact_state = "A侧：平台A锁定"
        else:
            exact_state = "B侧：平台B锁定"
        critical_cases.append([label, u0, m0, total, u_inf, m_inf, l_a, directional_market_state(u_inf, m_inf), exact_state])
    _write_csv(
        table_dir / "exp3_critical_boundary_cases.csv",
        ["case", "u0", "m0", "u0_plus_m0", "u_inf_numeric", "m_inf_numeric", "L_A_numeric", "numeric_state", "exact_symmetric_model_state"],
        critical_cases,
    )


def exp4_quality_threshold(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    dqs = np.linspace(0, 5, 101)
    rows = []
    l_values = []
    q_star = None

    for dq in dqs:
        p = params.with_updates(alpha=3.0, beta=3.0, qAU=float(dq), qAM=float(dq), qBU=0.0, qBM=0.0)
        u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
        l_a = combined_share(u_inf, m_inf)
        l_values.append(l_a)
        success = l_a > 0.6
        if success and q_star is None:
            q_star = float(dq)
        rows.append([dq, u_inf, m_inf, l_a, int(success)])

    plt.figure(figsize=(8, 5))
    plt.plot(dqs, l_values, color="#1f77b4")
    plt.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="逆袭标准 L_A=0.6")
    if q_star is not None:
        plt.axvline(q_star, color="#d62728", linestyle="--", linewidth=1.5, label=f"阈值 Δq*={q_star:.2f}")
    plt.xlabel("服务质量优势 Δq")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验4：服务质量优势与弱势平台逆袭阈值")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp4_quality_threshold.png")

    _write_csv(table_dir / "exp4_quality_threshold.csv", ["delta_q", "u_inf", "m_inf", "L_A", "success"], rows)

    compare_rows = []
    plt.figure(figsize=(8, 5))
    for label, k, color in [
        ("中等网络效应 alpha=beta=1.0", 1.0, "#4C78A8"),
        ("强网络效应 alpha=beta=3.0", 3.0, "#E45756"),
    ]:
        values = []
        q_star_k = None
        for dq in dqs:
            p = params.with_updates(alpha=k, beta=k, qAU=float(dq), qAM=float(dq), qBU=0.0, qBM=0.0)
            u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
            l_a = combined_share(u_inf, m_inf)
            values.append(l_a)
            success = l_a > 0.6
            if success and q_star_k is None:
                q_star_k = float(dq)
            compare_rows.append([label, k, dq, u_inf, m_inf, l_a, int(success)])
        plt.plot(dqs, values, label=f"{label}，阈值={q_star_k:.2f}" if q_star_k is not None else label, color=color)
    plt.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="逆袭标准 L_A=0.6")
    plt.xlabel("服务质量优势 Δq")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验4补充：不同网络效应下的服务质量逆袭对比")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp4_quality_network_compare.png")

    _write_csv(
        table_dir / "exp4_quality_network_compare.csv",
        ["setting", "network_strength", "delta_q", "u_inf", "m_inf", "L_A", "success"],
        compare_rows,
    )


def exp5_subsidy_basic(params: PlatformParams, fig_dir: Path, table_dir: Path) -> None:
    subsidies = np.linspace(0, 5, 101)
    rows = []
    l_values = []

    for b in subsidies:
        p = params.with_updates(alpha=3.0, beta=3.0, bAU=float(b), bAM=float(b), bBU=0.0, bBM=0.0)
        u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
        l_a = combined_share(u_inf, m_inf)
        l_values.append(l_a)
        rows.append([b, u_inf, m_inf, l_a, concentration(u_inf, m_inf), market_state(concentration(u_inf, m_inf))])

    selected = [0.0, 0.5, 1.0, 1.5, 2.0]
    plt.figure(figsize=(8, 5))
    for b in selected:
        p = params.with_updates(alpha=3.0, beta=3.0, bAU=b, bAM=b, bBU=0.0, bBM=0.0)
        t, u, m = run_simulation(p, u0=0.3, m0=0.3)
        plt.plot(t, u, label=f"b={b:.1f} 用户")
        plt.plot(t, m, linestyle="--", label=f"b={b:.1f} 商户")
    plt.xlabel("时间 t")
    plt.ylabel("平台 A 份额")
    plt.title("实验5：不同补贴强度下的动态演化")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=8)
    save_figure(fig_dir / "exp5_subsidy_dynamics.png")

    plt.figure(figsize=(8, 5))
    plt.plot(subsidies, l_values, color="#2ca02c")
    plt.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="L_A=0.6")
    plt.xlabel("统一补贴强度 b")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验5：补贴强度与最终市场份额")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp5_subsidy_final_share.png")

    _write_csv(table_dir / "exp5_subsidy_basic.csv", ["b", "u_inf", "m_inf", "L_A", "C", "state"], rows)

    strategy_rows = []
    strategies = [
        ("只补贴用户", lambda b: {"bAU": float(b), "bAM": 0.0}),
        ("只补贴商户", lambda b: {"bAU": 0.0, "bAM": float(b)}),
        ("双边统一补贴", lambda b: {"bAU": float(b), "bAM": float(b)}),
    ]

    plt.figure(figsize=(8, 5))
    for label, update_func in strategies:
        values = []
        threshold = None
        for b in subsidies:
            updates = update_func(b)
            p = params.with_updates(alpha=3.0, beta=3.0, bBU=0.0, bBM=0.0, **updates)
            u_inf, m_inf = run_final_state(p, u0=0.3, m0=0.3)
            l_a = combined_share(u_inf, m_inf)
            c = concentration(u_inf, m_inf)
            values.append(l_a)
            success = l_a > 0.6
            if success and threshold is None:
                threshold = float(b)
            strategy_rows.append([label, b, updates["bAU"], updates["bAM"], u_inf, m_inf, l_a, c, market_state(c), int(success)])
        threshold_text = f"，阈值={threshold:.2f}" if threshold is not None else ""
        plt.plot(subsidies, values, label=f"{label}{threshold_text}")
    plt.axhline(0.6, color="gray", linestyle="--", linewidth=1, label="L_A=0.6")
    plt.xlabel("补贴强度 b")
    plt.ylabel("平台 A 最终综合份额 L_A")
    plt.title("实验5补充：不同基础补贴对象的效果对比")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_figure(fig_dir / "exp5_subsidy_strategy_compare.png")

    _write_csv(
        table_dir / "exp5_subsidy_strategy_compare.csv",
        ["strategy", "b", "bAU", "bAM", "u_inf", "m_inf", "L_A", "C", "state", "success"],
        strategy_rows,
    )


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
