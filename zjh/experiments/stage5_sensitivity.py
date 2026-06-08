import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from src.config import Params
from src.model import simulate_path
from src.policies import (
    zero_policy,
    make_decay_subsidy_policy,
    greedy_quality_policy,
    long_term_quality_policy,
    dynamic_quality_policy,
)
from src.utils import (
    ensure_dir,
    setup_matplotlib,
    save_metrics_csv,
    summarize_result,
)


# ============================================================
# 通用工具函数
# ============================================================

def save_list_dict_csv(rows, path):
    """
    保存 list[dict] 类型的数据。
    """
    if len(rows) == 0:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_stage2_strategies():
    """
    阶段 2 中使用的补贴策略。
    """
    return {
        "无补贴": zero_policy,
        "用户补贴": make_decay_subsidy_policy(
            su0=0.85,
            sm0=0.00,
            decay_u=0.06,
            decay_m=0.06,
        ),
        "商户补贴": make_decay_subsidy_policy(
            su0=0.00,
            sm0=0.85,
            decay_u=0.06,
            decay_m=0.06,
        ),
        "双边补贴": make_decay_subsidy_policy(
            su0=0.55,
            sm0=0.55,
            decay_u=0.06,
            decay_m=0.06,
        ),
    }


def make_stage3_strategies():
    """
    阶段 3 中考虑供给不足时使用的补贴策略。
    """
    return {
        "无补贴": zero_policy,
        "用户补贴": make_decay_subsidy_policy(
            su0=1.10,
            sm0=0.00,
            decay_u=0.04,
            decay_m=0.06,
        ),
        "商户补贴": make_decay_subsidy_policy(
            su0=0.15,
            sm0=0.95,
            decay_u=0.07,
            decay_m=0.03,
        ),
        "双边协调补贴": make_decay_subsidy_policy(
            su0=0.65,
            sm0=0.75,
            decay_u=0.07,
            decay_m=0.035,
        ),
    }


# ============================================================
# 拓展 1：非对称网络效应下的补贴方向选择
# ============================================================

def experiment_asymmetric_network_subsidy_direction(output_dir):
    """
    分析 alpha != beta 时，不同补贴策略的收益差异。

    重点回答：
    1. 用户更依赖商户时，商户补贴是否更优？
    2. 商户更依赖用户时，用户补贴是否更优？
    """

    out = os.path.join(output_dir, "exp1_asymmetric_subsidy_direction")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    network_cases = [
        ("双边弱网络效应", 0.4, 0.4),
        ("对称强网络效应", 1.2, 1.2),
        ("用户更依赖商户", 1.6, 0.5),
        ("商户更依赖用户", 0.5, 1.6),
        ("极强锁定市场", 1.8, 1.8),
    ]

    strategies = make_stage2_strategies()

    rows = []

    for case_name, alpha, beta in network_cases:
        for strategy_name, policy in strategies.items():
            p = Params(alpha=alpha, beta=beta)

            res = simulate_path(
                x0,
                y0,
                p,
                T=80.0,
                dt=0.04,
                policy=policy
            )

            summary = summarize_result(strategy_name, res)
            summary["case_name"] = case_name
            summary["alpha"] = alpha
            summary["beta"] = beta
            rows.append(summary)

    save_list_dict_csv(
        rows,
        os.path.join(out, "asymmetric_network_subsidy_metrics.csv")
    )

    # ------------------------------------------------------------
    # 图 1：不同网络效应结构下的累计净收益比较
    # ------------------------------------------------------------

    case_names = [c[0] for c in network_cases]
    strategy_names = list(strategies.keys())

    profit_matrix = np.zeros((len(case_names), len(strategy_names)))
    final_x_matrix = np.zeros_like(profit_matrix)

    for i, case_name in enumerate(case_names):
        for j, strategy_name in enumerate(strategy_names):
            matched = [
                r for r in rows
                if r["case_name"] == case_name and r["name"] == strategy_name
            ][0]
            profit_matrix[i, j] = matched["cum_profit"]
            final_x_matrix[i, j] = matched["final_x"]

    x = np.arange(len(case_names))
    width = 0.18

    plt.figure(figsize=(13, 6))

    for j, strategy_name in enumerate(strategy_names):
        plt.bar(
            x + (j - 1.5) * width,
            profit_matrix[:, j],
            width,
            label=strategy_name
        )

    plt.xticks(x, case_names, rotation=15)
    plt.ylabel("累计净收益")
    plt.title("阶段5-拓展1：非对称网络效应下的补贴策略净收益比较")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "asymmetric_network_subsidy_profit_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ------------------------------------------------------------
    # 图 2：不同网络效应结构下的最终用户份额比较
    # ------------------------------------------------------------

    plt.figure(figsize=(13, 6))

    for j, strategy_name in enumerate(strategy_names):
        plt.bar(
            x + (j - 1.5) * width,
            final_x_matrix[:, j],
            width,
            label=strategy_name
        )

    plt.xticks(x, case_names, rotation=15)
    plt.ylabel(r"最终用户份额 $x^*$")
    plt.title("阶段5-拓展1：非对称网络效应下的补贴策略最终份额比较")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "asymmetric_network_subsidy_final_share_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ------------------------------------------------------------
    # 图 3：alpha-beta 网格下的最优补贴策略
    # ------------------------------------------------------------

    alpha_values = np.linspace(0.3, 1.8, 21)
    beta_values = np.linspace(0.3, 1.8, 21)

    subsidy_strategies = {
        "用户补贴": strategies["用户补贴"],
        "商户补贴": strategies["商户补贴"],
        "双边补贴": strategies["双边补贴"],
    }

    strategy_list = list(subsidy_strategies.keys())

    best_strategy_index = np.zeros((len(beta_values), len(alpha_values)))
    best_profit = np.zeros_like(best_strategy_index)
    best_final_x = np.zeros_like(best_strategy_index)

    grid_rows = []

    for i, beta in enumerate(beta_values):
        for j, alpha in enumerate(alpha_values):
            profits = []
            final_x_list = []

            for strategy_name, policy in subsidy_strategies.items():
                p = Params(alpha=alpha, beta=beta)

                res = simulate_path(
                    x0,
                    y0,
                    p,
                    T=70.0,
                    dt=0.06,
                    policy=policy
                )

                profits.append(res["cum_profit"][-1])
                final_x_list.append(res["x"][-1])

            best_idx = int(np.argmax(profits))

            best_strategy_index[i, j] = best_idx
            best_profit[i, j] = profits[best_idx]
            best_final_x[i, j] = final_x_list[best_idx]

            grid_rows.append({
                "alpha": alpha,
                "beta": beta,
                "best_strategy": strategy_list[best_idx],
                "best_profit": profits[best_idx],
                "best_final_x": final_x_list[best_idx],
                "user_subsidy_profit": profits[0],
                "merchant_subsidy_profit": profits[1],
                "bilateral_subsidy_profit": profits[2],
            })

    save_list_dict_csv(
        grid_rows,
        os.path.join(out, "alpha_beta_best_subsidy_strategy_grid.csv")
    )

    cmap = ListedColormap(["#4C72B0", "#55A868", "#C44E52"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        best_strategy_index,
        extent=[
            alpha_values.min(), alpha_values.max(),
            beta_values.min(), beta_values.max()
        ],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(strategy_list)

    plt.title("阶段5-拓展1：不同网络效应结构下的最优补贴方向")
    plt.xlabel(r"$\alpha$：商户规模对用户的影响")
    plt.ylabel(r"$\beta$：用户规模对商户的影响")

    plt.savefig(
        os.path.join(out, "alpha_beta_best_subsidy_strategy_map.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# 拓展 2：最低有效补贴组合分析
# ============================================================

def experiment_minimum_effective_subsidy(output_dir):
    """
    寻找使平台 A 达到锁定判据的最低有效补贴组合。

    判据：
        x* >= 0.8 且 y* >= 0.8

    目标：
        在网格搜索中寻找 su0 + sm0 最小的成功组合。
    """

    out = os.path.join(output_dir, "exp2_minimum_effective_subsidy")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    alpha_beta_cases = [
        ("对称强网络效应", 1.2, 1.2),
        ("用户更依赖商户", 1.6, 0.5),
        ("商户更依赖用户", 0.5, 1.6),
        ("极强锁定市场", 1.8, 1.8),
    ]

    su_values = np.linspace(0.0, 1.2, 41)
    sm_values = np.linspace(0.0, 1.2, 41)

    all_rows = []

    for case_name, alpha, beta in alpha_beta_cases:
        success = np.zeros((len(sm_values), len(su_values)))
        final_x = np.zeros_like(success)
        final_y = np.zeros_like(success)
        profit = np.zeros_like(success)
        total_initial_subsidy = np.zeros_like(success)

        best = None

        for i, sm in enumerate(sm_values):
            for j, su in enumerate(su_values):
                p = Params(alpha=alpha, beta=beta)

                policy = make_decay_subsidy_policy(
                    su0=su,
                    sm0=sm,
                    decay_u=0.06,
                    decay_m=0.06,
                )

                res = simulate_path(
                    x0,
                    y0,
                    p,
                    T=80.0,
                    dt=0.06,
                    policy=policy
                )

                xf = res["x"][-1]
                yf = res["y"][-1]
                pf = res["cum_profit"][-1]

                final_x[i, j] = xf
                final_y[i, j] = yf
                profit[i, j] = pf

                total = su + sm
                total_initial_subsidy[i, j] = total

                is_success = (xf >= 0.8) and (yf >= 0.8)
                success[i, j] = 1.0 if is_success else 0.0

                all_rows.append({
                    "case_name": case_name,
                    "alpha": alpha,
                    "beta": beta,
                    "su0": su,
                    "sm0": sm,
                    "total_initial_subsidy": total,
                    "final_x": xf,
                    "final_y": yf,
                    "cum_profit": pf,
                    "success": int(is_success),
                })

                if is_success:
                    if best is None:
                        best = (total, su, sm, xf, yf, pf)
                    else:
                        if total < best[0]:
                            best = (total, su, sm, xf, yf, pf)

        # --------------------------------------------------------
        # 每个网络效应情形画一张成功区域图
        # --------------------------------------------------------

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            success,
            extent=[
                su_values.min(), su_values.max(),
                sm_values.min(), sm_values.max()
            ],
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=1,
            cmap="viridis"
        )

        plt.colorbar(im, label="是否达到锁定判据")
        plt.xlabel("初始用户补贴强度")
        plt.ylabel("初始商户补贴强度")
        plt.title(f"阶段5-拓展2：最低有效补贴区域 - {case_name}")

        if best is not None:
            total, su_best, sm_best, xf_best, yf_best, pf_best = best
            plt.scatter([su_best], [sm_best], marker="*", s=180)
            plt.text(
                su_best + 0.03,
                sm_best + 0.03,
                rf"$s_U+s_M={total:.2f}$",
                fontsize=11
            )

        plt.savefig(
            os.path.join(
                out,
                f"minimum_effective_subsidy_{case_name}.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    save_list_dict_csv(
        all_rows,
        os.path.join(out, "minimum_effective_subsidy_grid.csv")
    )

    # ------------------------------------------------------------
    # 汇总每个情形的最低有效补贴组合
    # ------------------------------------------------------------

    summary_rows = []

    for case_name, alpha, beta in alpha_beta_cases:
        rows = [
            r for r in all_rows
            if r["case_name"] == case_name and r["success"] == 1
        ]

        if len(rows) == 0:
            summary_rows.append({
                "case_name": case_name,
                "alpha": alpha,
                "beta": beta,
                "min_total_subsidy": None,
                "best_su0": None,
                "best_sm0": None,
                "final_x": None,
                "final_y": None,
                "cum_profit": None,
            })
        else:
            rows_sorted = sorted(rows, key=lambda r: r["total_initial_subsidy"])
            best = rows_sorted[0]

            summary_rows.append({
                "case_name": case_name,
                "alpha": alpha,
                "beta": beta,
                "min_total_subsidy": best["total_initial_subsidy"],
                "best_su0": best["su0"],
                "best_sm0": best["sm0"],
                "final_x": best["final_x"],
                "final_y": best["final_y"],
                "cum_profit": best["cum_profit"],
            })

    save_list_dict_csv(
        summary_rows,
        os.path.join(out, "minimum_effective_subsidy_summary.csv")
    )


# ============================================================
# 拓展 3：供给不足惩罚强度 rho 的敏感性分析
# ============================================================

def experiment_shortage_penalty_sensitivity(output_dir):
    """
    分析供给不足惩罚强度 rho 对补贴策略效果的影响。

    重点回答：
    当 rho 增大时，用户补贴是否仍然有效？
    """

    out = os.path.join(output_dir, "exp3_shortage_penalty_sensitivity")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    rho_values = np.linspace(0.0, 10.0, 31)

    strategies = {
        "用户补贴": make_stage3_strategies()["用户补贴"],
        "商户补贴": make_stage3_strategies()["商户补贴"],
        "双边协调补贴": make_stage3_strategies()["双边协调补贴"],
    }

    rows = []

    for rho in rho_values:
        for strategy_name, policy in strategies.items():
            p = Params(
                alpha=1.2,
                beta=1.0,
                shortage_enabled=True,
                shortage_rho=float(rho),
                shortage_buffer=0.02,
            )

            res = simulate_path(
                x0,
                y0,
                p,
                T=90.0,
                dt=0.04,
                policy=policy
            )

            rows.append({
                "rho": float(rho),
                "strategy": strategy_name,
                "final_x": float(res["x"][-1]),
                "final_y": float(res["y"][-1]),
                "max_shortage_A": float(np.max(res["shortage_A"])),
                "avg_shortage_A": float(np.mean(res["shortage_A"])),
                "cum_profit": float(res["cum_profit"][-1]),
                "cum_total_cost": float(res["cum_total_cost"][-1]),
            })

    save_list_dict_csv(
        rows,
        os.path.join(out, "shortage_penalty_sensitivity.csv")
    )

    # ------------------------------------------------------------
    # 图：final_x、max_shortage、cum_profit 随 rho 变化
    # ------------------------------------------------------------

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for strategy_name in strategies.keys():
        sub = [r for r in rows if r["strategy"] == strategy_name]
        rhos = [r["rho"] for r in sub]
        final_x = [r["final_x"] for r in sub]
        max_shortage = [r["max_shortage_A"] for r in sub]
        profits = [r["cum_profit"] for r in sub]

        axes[0].plot(rhos, final_x, marker="o", markersize=3, label=strategy_name)
        axes[1].plot(rhos, max_shortage, marker="o", markersize=3, label=strategy_name)
        axes[2].plot(rhos, profits, marker="o", markersize=3, label=strategy_name)

    axes[0].set_title(r"阶段5-拓展3：供给不足惩罚强度 $\rho$ 对最终份额的影响")
    axes[0].set_ylabel(r"最终用户份额 $x^*$")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(r"供给不足惩罚强度 $\rho$ 对最大供给不足的影响")
    axes[1].set_ylabel("最大供给不足")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title(r"供给不足惩罚强度 $\rho$ 对累计净收益的影响")
    axes[2].set_xlabel(r"供给不足惩罚强度 $\rho$")
    axes[2].set_ylabel("累计净收益")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(
        os.path.join(out, "shortage_penalty_sensitivity_lines.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 拓展 4：服务质量投资效率与衰减率敏感性分析
# ============================================================

def experiment_quality_investment_sensitivity(output_dir):
    """
    分析服务质量投资效率 mu 和服务质量衰减率 delta 的影响。

    包含：
    1. 动态策略下 mu-delta 二维扫描；
    2. 不同 mu 下三种投资策略的净收益比较。
    """

    out = os.path.join(output_dir, "exp4_quality_investment_sensitivity")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    # ------------------------------------------------------------
    # 4.1 动态策略下 mu-delta 二维扫描
    # ------------------------------------------------------------

    mu_values = np.linspace(0.10, 0.55, 19)
    delta_values = np.linspace(0.01, 0.10, 19)

    final_x_grid = np.zeros((len(delta_values), len(mu_values)))
    final_y_grid = np.zeros_like(final_x_grid)
    profit_grid = np.zeros_like(final_x_grid)
    invest_cost_grid = np.zeros_like(final_x_grid)

    grid_rows = []

    for i, delta in enumerate(delta_values):
        for j, mu in enumerate(mu_values):
            p = Params(
                alpha=1.2,
                beta=1.0,
                shortage_enabled=True,
                shortage_rho=4.5,
                shortage_buffer=0.02,
                invest_eff_u=float(mu),
                invest_eff_m=float(mu),
                quality_decay=float(delta),
            )

            res = simulate_path(
                x0,
                y0,
                p,
                T=100.0,
                dt=0.05,
                policy=dynamic_quality_policy
            )

            final_x_grid[i, j] = res["x"][-1]
            final_y_grid[i, j] = res["y"][-1]
            profit_grid[i, j] = res["cum_profit"][-1]
            invest_cost_grid[i, j] = res["cum_invest_cost"][-1]

            grid_rows.append({
                "mu": float(mu),
                "delta": float(delta),
                "final_x": float(res["x"][-1]),
                "final_y": float(res["y"][-1]),
                "cum_profit": float(res["cum_profit"][-1]),
                "cum_invest_cost": float(res["cum_invest_cost"][-1]),
                "max_shortage_A": float(np.max(res["shortage_A"])),
            })

    save_list_dict_csv(
        grid_rows,
        os.path.join(out, "quality_investment_mu_delta_grid.csv")
    )

    # 最终用户份额热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        final_x_grid,
        extent=[
            mu_values.min(), mu_values.max(),
            delta_values.min(), delta_values.max()
        ],
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis"
    )

    plt.colorbar(im, label=r"最终用户份额 $x^*$")
    plt.title(r"阶段5-拓展4：投资效率 $\mu$ 与衰减率 $\delta$ 对最终份额的影响")
    plt.xlabel(r"投资效率 $\mu$")
    plt.ylabel(r"服务质量衰减率 $\delta$")

    plt.savefig(
        os.path.join(out, "quality_mu_delta_final_share_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # 累计净收益热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        profit_grid,
        extent=[
            mu_values.min(), mu_values.max(),
            delta_values.min(), delta_values.max()
        ],
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )

    plt.colorbar(im, label="累计净收益")
    plt.title(r"阶段5-拓展4：投资效率 $\mu$ 与衰减率 $\delta$ 对净收益的影响")
    plt.xlabel(r"投资效率 $\mu$")
    plt.ylabel(r"服务质量衰减率 $\delta$")

    plt.savefig(
        os.path.join(out, "quality_mu_delta_profit_heatmap.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ------------------------------------------------------------
    # 4.2 不同投资效率下三种服务质量策略比较
    # ------------------------------------------------------------

    mu_list = [0.12, 0.18, 0.25, 0.32, 0.40, 0.50]

    strategies = {
        "贪心策略": greedy_quality_policy,
        "长期策略": long_term_quality_policy,
        "动态策略": dynamic_quality_policy,
    }

    rows = []

    for mu in mu_list:
        for strategy_name, policy in strategies.items():
            p = Params(
                alpha=1.2,
                beta=1.0,
                shortage_enabled=True,
                shortage_rho=4.5,
                shortage_buffer=0.02,
                invest_eff_u=float(mu),
                invest_eff_m=float(mu),
                quality_decay=0.045,
            )

            res = simulate_path(
                x0,
                y0,
                p,
                T=100.0,
                dt=0.04,
                policy=policy
            )

            rows.append({
                "mu": float(mu),
                "strategy": strategy_name,
                "final_x": float(res["x"][-1]),
                "final_y": float(res["y"][-1]),
                "max_shortage_A": float(np.max(res["shortage_A"])),
                "cum_profit": float(res["cum_profit"][-1]),
                "cum_invest_cost": float(res["cum_invest_cost"][-1]),
            })

    save_list_dict_csv(
        rows,
        os.path.join(out, "quality_strategy_mu_compare.csv")
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for strategy_name in strategies.keys():
        sub = [r for r in rows if r["strategy"] == strategy_name]

        mus = [r["mu"] for r in sub]
        final_x = [r["final_x"] for r in sub]
        profits = [r["cum_profit"] for r in sub]
        shortages = [r["max_shortage_A"] for r in sub]

        axes[0].plot(mus, final_x, marker="o", label=strategy_name)
        axes[1].plot(mus, profits, marker="o", label=strategy_name)
        axes[2].plot(mus, shortages, marker="o", label=strategy_name)

    axes[0].set_title(r"投资效率 $\mu$ 对最终用户份额的影响")
    axes[0].set_ylabel(r"$x^*$")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title(r"投资效率 $\mu$ 对累计净收益的影响")
    axes[1].set_ylabel("累计净收益")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title(r"投资效率 $\mu$ 对供给不足风险的影响")
    axes[2].set_xlabel(r"投资效率 $\mu$")
    axes[2].set_ylabel("最大供给不足")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(
        os.path.join(out, "quality_strategy_mu_compare_lines.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


# ============================================================
# 阶段 5 总入口
# ============================================================

def run_stage5(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage5_sensitivity")
    ensure_dir(out)

    print("  阶段5-拓展1：非对称网络效应下的补贴方向选择...")
    experiment_asymmetric_network_subsidy_direction(out)

    print("  阶段5-拓展2：最低有效补贴组合分析...")
    experiment_minimum_effective_subsidy(out)

    print("  阶段5-拓展3：供给不足惩罚强度敏感性分析...")
    experiment_shortage_penalty_sensitivity(out)

    print("  阶段5-拓展4：服务质量投资效率与衰减率敏感性分析...")
    experiment_quality_investment_sensitivity(out)

    print("  阶段5全部拓展实验完成。")