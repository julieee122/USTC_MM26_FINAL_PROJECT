import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.config import Params
from src.model import simulate_path
from src.utils import ensure_dir, setup_matplotlib
from src.metrics import (
    concentration,
    combined_share,
    market_state,
    directional_market_state,
)


def save_rows_csv(rows, path):
    if len(rows) == 0:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def classify_concentration(c: float) -> int:
    """
    0: 双平台共存
    1: 市场倾斜
    2: 市场锁定
    """
    if c < 0.2:
        return 0
    if c < 0.8:
        return 1
    return 2


# ============================================================
# 实验 7.1：市场锁定临界区域
# ============================================================

def experiment_lockin_critical_region(output_dir):
    """
    扫描 alpha 和 beta，识别共存区、倾斜区、锁定区。

    该实验用于补充阶段1：
    阶段1展示了网络效应增强会带来锁定；
    本实验进一步给出临界区域和临界阈值。
    """
    out = os.path.join(output_dir, "exp1_lockin_critical_region")
    ensure_dir(out)

    alpha_values = np.linspace(0.0, 2.5, 51)
    beta_values = np.linspace(0.0, 2.5, 51)

    c_grid = np.zeros((len(beta_values), len(alpha_values)))
    class_grid = np.zeros_like(c_grid, dtype=int)
    final_x_grid = np.zeros_like(c_grid)
    final_y_grid = np.zeros_like(c_grid)

    rows = []

    x0 = 0.55
    y0 = 0.55

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            p = Params(alpha=float(alpha), beta=float(beta))
            res = simulate_path(x0, y0, p, T=70.0, dt=0.05)

            xf = float(res["x"][-1])
            yf = float(res["y"][-1])
            c = concentration(xf, yf)

            final_x_grid[i, j] = xf
            final_y_grid[i, j] = yf
            c_grid[i, j] = c
            class_grid[i, j] = classify_concentration(c)

            rows.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "final_x": xf,
                "final_y": yf,
                "combined_share": combined_share(xf, yf),
                "concentration": c,
                "market_state": market_state(c),
                "directional_state": directional_market_state(xf, yf),
            })

    save_rows_csv(rows, os.path.join(out, "lockin_region_grid.csv"))

    # 连续集中度热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        c_grid,
        origin="lower",
        extent=[
            alpha_values.min(), alpha_values.max(),
            beta_values.min(), beta_values.max()
        ],
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis"
    )
    plt.colorbar(im, label="市场集中度 C")
    plt.xlabel(r"$\alpha$：商户规模对用户的影响")
    plt.ylabel(r"$\beta$：用户规模对商户的影响")
    plt.title("阶段7-实验1：市场集中度连续热力图")

    plt.savefig(
        os.path.join(out, "lockin_concentration_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 分类热力图
    cmap = ListedColormap(["#7FC97F", "#FDB462", "#E15759"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        class_grid,
        origin="lower",
        extent=[
            alpha_values.min(), alpha_values.max(),
            beta_values.min(), beta_values.max()
        ],
        aspect="auto",
        cmap=cmap,
        norm=norm
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["双平台共存", "市场倾斜", "市场锁定"])

    plt.xlabel(r"$\alpha$：商户规模对用户的影响")
    plt.ylabel(r"$\beta$：用户规模对商户的影响")
    plt.title("阶段7-实验1：共存区、倾斜区与锁定区")

    plt.savefig(
        os.path.join(out, "lockin_region_classification.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 对称路径 alpha=beta=k 的临界点
    k_values = np.linspace(0.0, 2.5, 101)

    diag_rows = []
    diag_c = []
    k_tilt = None
    k_lock = None

    for k in k_values:
        p = Params(alpha=float(k), beta=float(k))
        res = simulate_path(x0, y0, p, T=70.0, dt=0.05)

        xf = float(res["x"][-1])
        yf = float(res["y"][-1])
        c = concentration(xf, yf)

        diag_c.append(c)

        if c >= 0.2 and k_tilt is None:
            k_tilt = float(k)

        if c >= 0.8 and k_lock is None:
            k_lock = float(k)

        diag_rows.append({
            "k": float(k),
            "final_x": xf,
            "final_y": yf,
            "concentration": c,
            "market_state": market_state(c),
            "directional_state": directional_market_state(xf, yf),
        })

    save_rows_csv(diag_rows, os.path.join(out, "diagonal_k_scan.csv"))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, diag_c, linewidth=2)
    plt.axhline(0.2, linestyle="--", linewidth=1, label="共存-倾斜阈值 C=0.2")
    plt.axhline(0.8, linestyle="--", linewidth=1, label="倾斜-锁定阈值 C=0.8")

    if k_tilt is not None:
        plt.axvline(k_tilt, linestyle=":", linewidth=1.5)
        plt.text(k_tilt + 0.03, 0.25, rf"$k_1\approx {k_tilt:.2f}$")

    if k_lock is not None:
        plt.axvline(k_lock, linestyle=":", linewidth=1.5)
        plt.text(k_lock + 0.03, 0.85, rf"$k_2\approx {k_lock:.2f}$")

    plt.xlabel(r"对称网络效应强度 $k=\alpha=\beta$")
    plt.ylabel("市场集中度 C")
    plt.title("阶段7-实验1：对称网络效应路径下的临界跃迁")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "diagonal_k_critical_curve.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    summary_rows = [{
        "coexist_count": int((class_grid == 0).sum()),
        "tilt_count": int((class_grid == 1).sum()),
        "lockin_count": int((class_grid == 2).sum()),
        "diagonal_tilt_threshold_k": "" if k_tilt is None else k_tilt,
        "diagonal_lockin_threshold_k": "" if k_lock is None else k_lock,
    }]

    save_rows_csv(summary_rows, os.path.join(out, "lockin_region_summary.csv"))


# ============================================================
# 实验 7.2：惯性/切换成本对服务质量逆袭阈值的影响
# ============================================================

def experiment_inertia_quality_threshold(output_dir):
    """
    分析同侧锁定效应 eta 对服务质量逆袭阈值的影响。

    在你的模型中，eta_u 和 eta_m 可解释为同侧锁定/惯性/切换成本。
    eta 越大，用户和商户越不容易离开原有优势平台。
    """
    out = os.path.join(output_dir, "exp2_inertia_quality_threshold")
    ensure_dir(out)

    network_strengths = [0.5, 1.0, 1.5, 2.0]
    eta_values = [0.0, 0.2, 0.5, 0.8, 1.0]
    q_values = np.linspace(0.0, 1.5, 151)

    x0 = 0.35
    y0 = 0.35

    threshold_grid = np.full((len(eta_values), len(network_strengths)), np.nan)
    rows = []

    for i, eta in enumerate(eta_values):
        for j, k in enumerate(network_strengths):
            q_star = np.nan
            xf_star = np.nan
            yf_star = np.nan

            for q in q_values:
                p = Params(
                    alpha=k,
                    beta=k,
                    eta_u=eta,
                    eta_m=eta,
                    dq_u_base=float(q),
                    dq_m_base=float(q),
                )

                res = simulate_path(x0, y0, p, T=80.0, dt=0.04)

                xf = float(res["x"][-1])
                yf = float(res["y"][-1])

                if xf > 0.5 and yf > 0.5:
                    q_star = float(q)
                    xf_star = xf
                    yf_star = yf
                    break

            threshold_grid[i, j] = q_star

            rows.append({
                "network_strength_k": k,
                "eta": eta,
                "quality_threshold": q_star,
                "final_x_at_threshold": xf_star,
                "final_y_at_threshold": yf_star,
            })

    save_rows_csv(rows, os.path.join(out, "inertia_quality_threshold.csv"))

    plt.figure(figsize=(8, 6))
    masked = np.ma.masked_invalid(threshold_grid)

    im = plt.imshow(
        masked,
        origin="lower",
        aspect="auto",
        extent=[
            min(network_strengths), max(network_strengths),
            min(eta_values), max(eta_values)
        ],
        cmap="magma_r",
        vmin=0,
        vmax=1.5
    )

    plt.colorbar(im, label=r"最小服务质量优势 $q_c$")
    plt.xticks(network_strengths)
    plt.yticks(eta_values)
    plt.xlabel(r"网络效应强度 $k=\alpha=\beta$")
    plt.ylabel(r"同侧锁定/惯性强度 $\eta$")
    plt.title("阶段7-实验2：惯性与网络效应对服务质量逆袭阈值的影响")

    for i, eta in enumerate(eta_values):
        for j, k in enumerate(network_strengths):
            value = threshold_grid[i, j]
            text = "无" if np.isnan(value) else f"{value:.2f}"
            plt.text(k, eta, text, ha="center", va="center", fontsize=8)

    plt.savefig(
        os.path.join(out, "inertia_quality_threshold_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 实验 7.3：固定预算下的双边补贴分配
# ============================================================

def make_budget_policy(budget, rho, decay=0.06):
    """
    budget: 总补贴预算强度
    rho: 用户侧补贴比例
    """
    def policy(t, state, p):
        su = rho * budget * np.exp(-decay * t)
        sm = (1.0 - rho) * budget * np.exp(-decay * t)

        return {
            "ds_u": float(su),
            "ds_m": float(sm),
            "inv_u": 0.0,
            "inv_m": 0.0,
        }

    return policy


def experiment_budget_allocation(output_dir):
    """
    固定总预算 B，扫描用户补贴比例 rho。

    b_U = rho * B
    b_M = (1-rho) * B
    """
    out = os.path.join(output_dir, "exp3_budget_allocation")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    budgets = np.round(np.arange(0.2, 1.61, 0.05), 2)
    rho_values = np.linspace(0.0, 1.0, 51)

    l_grid = np.zeros((len(budgets), len(rho_values)))
    profit_grid = np.zeros_like(l_grid)

    rows = []

    p_base = Params(alpha=1.2, beta=1.0)

    for i, budget in enumerate(budgets):
        for j, rho in enumerate(rho_values):
            policy = make_budget_policy(budget=budget, rho=rho, decay=0.06)

            res = simulate_path(x0, y0, p_base, T=80.0, dt=0.05, policy=policy)

            xf = float(res["x"][-1])
            yf = float(res["y"][-1])
            l_a = combined_share(xf, yf)
            profit = float(res["cum_profit"][-1])

            l_grid[i, j] = l_a
            profit_grid[i, j] = profit

            rows.append({
                "budget": float(budget),
                "rho": float(rho),
                "user_subsidy_initial": float(rho * budget),
                "merchant_subsidy_initial": float((1.0 - rho) * budget),
                "final_x": xf,
                "final_y": yf,
                "combined_share": l_a,
                "concentration": concentration(xf, yf),
                "cum_profit": profit,
                "directional_state": directional_market_state(xf, yf),
            })

    save_rows_csv(rows, os.path.join(out, "budget_allocation_grid.csv"))

    # 预算-比例热力图：最终综合份额
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        l_grid,
        origin="lower",
        aspect="auto",
        extent=[
            rho_values.min(), rho_values.max(),
            budgets.min(), budgets.max()
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    plt.colorbar(im, label=r"最终综合份额 $L_A$")
    plt.xlabel(r"用户补贴比例 $\rho$")
    plt.ylabel(r"总补贴预算 $B$")
    plt.title("阶段7-实验3：固定预算下补贴分配对最终份额的影响")

    plt.savefig(
        os.path.join(out, "budget_rho_share_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 预算-比例热力图：累计净收益
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        profit_grid,
        origin="lower",
        aspect="auto",
        extent=[
            rho_values.min(), rho_values.max(),
            budgets.min(), budgets.max()
        ],
        cmap="viridis"
    )

    plt.colorbar(im, label="累计净收益")
    plt.xlabel(r"用户补贴比例 $\rho$")
    plt.ylabel(r"总补贴预算 $B$")
    plt.title("阶段7-实验3：固定预算下补贴分配对净收益的影响")

    plt.savefig(
        os.path.join(out, "budget_rho_profit_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 临界预算附近曲线
    selected_budgets = [0.4, 0.6, 0.8, 1.0, 1.2]

    plt.figure(figsize=(9, 6))

    for budget in selected_budgets:
        idx = int(np.argmin(np.abs(budgets - budget)))
        best_idx = int(np.argmax(l_grid[idx]))

        plt.plot(
            rho_values,
            l_grid[idx],
            label=rf"$B={budgets[idx]:.2f}$，最优 $\rho={rho_values[best_idx]:.2f}$"
        )

    plt.xlabel(r"用户补贴比例 $\rho$")
    plt.ylabel(r"最终综合份额 $L_A$")
    plt.title("阶段7-实验3：不同预算下的最优补贴方向")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "selected_budget_rho_curves.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 不同网络效应结构下的最优 rho
    scenarios = [
        ("对称网络效应", 1.2, 1.2),
        ("用户更依赖商户", 1.6, 0.5),
        ("商户更依赖用户", 0.5, 1.6),
    ]

    scenario_budget = 0.8
    scenario_rows = []

    plt.figure(figsize=(9, 6))

    for label, alpha, beta in scenarios:
        values = []
        profits = []

        p_case = Params(alpha=alpha, beta=beta)

        for rho in rho_values:
            policy = make_budget_policy(
                budget=scenario_budget,
                rho=rho,
                decay=0.06
            )

            res = simulate_path(x0, y0, p_case, T=80.0, dt=0.05, policy=policy)

            xf = float(res["x"][-1])
            yf = float(res["y"][-1])
            l_a = combined_share(xf, yf)
            profit = float(res["cum_profit"][-1])

            values.append(l_a)
            profits.append(profit)

            scenario_rows.append({
                "scenario": label,
                "alpha": alpha,
                "beta": beta,
                "budget": scenario_budget,
                "rho": float(rho),
                "final_x": xf,
                "final_y": yf,
                "combined_share": l_a,
                "cum_profit": profit,
                "directional_state": directional_market_state(xf, yf),
            })

        best_idx = int(np.argmax(values))

        plt.plot(
            rho_values,
            values,
            label=rf"{label}，最优 $\rho={rho_values[best_idx]:.2f}$"
        )

    save_rows_csv(
        scenario_rows,
        os.path.join(out, "network_scenario_budget_allocation.csv")
    )

    plt.xlabel(r"用户补贴比例 $\rho$")
    plt.ylabel(r"最终综合份额 $L_A$")
    plt.title("阶段7-实验3：不同网络效应结构下的补贴比例选择")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "network_scenario_budget_allocation.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 实验 7.4：补贴退出与利润最优补贴强度
# ============================================================

def make_step_subsidy_policy(b0, duration, split=0.5):
    """
    阶段性补贴：
    0 <= t <= duration 时给予补贴；
    t > duration 后补贴退出。

    split: 用户侧补贴比例
    """
    def policy(t, state, p):
        if t <= duration:
            su = split * b0
            sm = (1.0 - split) * b0
        else:
            su = 0.0
            sm = 0.0

        return {
            "ds_u": float(su),
            "ds_m": float(sm),
            "inv_u": 0.0,
            "inv_m": 0.0,
        }

    return policy


def experiment_subsidy_exit_and_profit(output_dir):
    """
    分析补贴强度 b0 和补贴持续时间 duration 对最终份额和利润的影响。
    """
    out = os.path.join(output_dir, "exp4_subsidy_exit_profit")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    p_base = Params(alpha=1.2, beta=1.0)

    b0_values = np.linspace(0.0, 1.5, 41)
    duration_values = np.linspace(0.0, 30.0, 31)

    share_grid = np.zeros((len(duration_values), len(b0_values)))
    profit_grid = np.zeros_like(share_grid)
    success_grid = np.zeros_like(share_grid)

    rows = []

    for i, duration in enumerate(duration_values):
        for j, b0 in enumerate(b0_values):
            policy = make_step_subsidy_policy(
                b0=float(b0),
                duration=float(duration),
                split=0.5
            )

            res = simulate_path(x0, y0, p_base, T=80.0, dt=0.05, policy=policy)

            xf = float(res["x"][-1])
            yf = float(res["y"][-1])
            l_a = combined_share(xf, yf)
            profit = float(res["cum_profit"][-1])
            success = 1 if (xf >= 0.8 and yf >= 0.8) else 0

            share_grid[i, j] = l_a
            profit_grid[i, j] = profit
            success_grid[i, j] = success

            rows.append({
                "b0": float(b0),
                "duration": float(duration),
                "final_x": xf,
                "final_y": yf,
                "combined_share": l_a,
                "cum_profit": profit,
                "success": success,
                "directional_state": directional_market_state(xf, yf),
            })

    save_rows_csv(rows, os.path.join(out, "subsidy_exit_grid.csv"))

    # 最终份额热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        share_grid,
        origin="lower",
        aspect="auto",
        extent=[
            b0_values.min(), b0_values.max(),
            duration_values.min(), duration_values.max()
        ],
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    plt.colorbar(im, label=r"最终综合份额 $L_A$")
    plt.xlabel(r"补贴强度 $b_0$")
    plt.ylabel(r"补贴持续时间 $T_s$")
    plt.title("阶段7-实验4：补贴强度与退出时间对最终份额的影响")

    plt.savefig(
        os.path.join(out, "subsidy_exit_share_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 利润热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        profit_grid,
        origin="lower",
        aspect="auto",
        extent=[
            b0_values.min(), b0_values.max(),
            duration_values.min(), duration_values.max()
        ],
        cmap="viridis"
    )

    plt.colorbar(im, label="累计净收益")
    plt.xlabel(r"补贴强度 $b_0$")
    plt.ylabel(r"补贴持续时间 $T_s$")
    plt.title("阶段7-实验4：补贴强度与退出时间对累计净收益的影响")

    plt.savefig(
        os.path.join(out, "subsidy_exit_profit_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 固定退出时间下的份额-利润双轴图
    fixed_duration = 10.0
    idx_duration = int(np.argmin(np.abs(duration_values - fixed_duration)))

    l_values = share_grid[idx_duration]
    p_values = profit_grid[idx_duration]

    best_profit_idx = int(np.argmax(p_values))
    first_success_idx = None

    for idx, value in enumerate(l_values):
        if value >= 0.8:
            first_success_idx = idx
            break

    fig, ax1 = plt.subplots(figsize=(9, 6))

    ax1.plot(b0_values, l_values, marker="o", markersize=3, label="最终综合份额")
    ax1.set_xlabel(r"补贴强度 $b_0$")
    ax1.set_ylabel(r"最终综合份额 $L_A$")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(b0_values, p_values, marker="s", markersize=3, linestyle="--", label="累计净收益")
    ax2.set_ylabel("累计净收益")

    if first_success_idx is not None:
        ax1.axvline(b0_values[first_success_idx], linestyle=":", linewidth=1.5)
        ax1.text(
            b0_values[first_success_idx] + 0.02,
            0.15,
            rf"最低成功补贴 $\approx {b0_values[first_success_idx]:.2f}$"
        )

    ax1.axvline(b0_values[best_profit_idx], linestyle=":", linewidth=1.5)
    ax2.text(
        b0_values[best_profit_idx] + 0.02,
        max(p_values) * 0.85,
        rf"利润最优补贴 $\approx {b0_values[best_profit_idx]:.2f}$"
    )

    plt.title(rf"阶段7-实验4：固定补贴持续时间 $T_s={duration_values[idx_duration]:.1f}$ 下的份额-利润权衡")

    fig.savefig(
        os.path.join(out, "fixed_duration_share_profit_tradeoff.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 阶段 7 总入口
# ============================================================

def run_stage7(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage7_critical_policy")
    ensure_dir(out)

    print("  阶段7-实验1：市场锁定临界区域...")
    experiment_lockin_critical_region(out)

    print("  阶段7-实验2：惯性/切换成本对服务质量逆袭阈值的影响...")
    experiment_inertia_quality_threshold(out)

    print("  阶段7-实验3：固定预算下的双边补贴分配...")
    experiment_budget_allocation(out)

    print("  阶段7-实验4：补贴退出与利润最优补贴强度...")
    experiment_subsidy_exit_and_profit(out)

    print("  阶段7全部完成。")