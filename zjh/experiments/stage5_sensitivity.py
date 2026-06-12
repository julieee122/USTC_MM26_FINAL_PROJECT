"""
Stage 5: 参数敏感性与临界条件

本文件对应实验报告 5.7 节“参数敏感性与临界条件”。


"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.config import Params
from experiments.report_common import (
    ensure_dir,
    setup_matplotlib,
    save_rows_csv,
    make_params,
    call_simulate,
    get_x,
    get_y,
    get_profit,
    get_shortage_B,
    get_quality,
    combined_share,
    platform_state,
    static_policy_B,
)


# ============================================================
# 报告统一参数
# ============================================================

DT = 0.05

# 平台 A 初始占优；平台 B 为挑战者。
X0_A = 0.8
Y0_A = 0.8

T_DEFAULT = 200.0
T_QUALITY = 300.0

MEDIUM_ALPHA = 0.8
MEDIUM_BETA = 0.8
STRONG_ALPHA = 1.5
STRONG_BETA = 1.5

ASYMMETRIC_SCENARIOS = [
    ("对称强网络效应", 1.2, 1.2),
    ("用户更依赖商户", 1.6, 0.5),
    ("商户更依赖用户", 0.5, 1.6),
    ("极强锁定市场", 1.8, 1.8),
]

N_U = 1000
N_M = 50
RHO = 10.0
EPS = 1e-6

LAMBDA_Q_BASE = 0.05
D_BASE = 0.01
Q_MAX = 3.0

BUDGET = 0.8
DISCOUNT = 0.98


def _final_B_from_result(result):
    """由模型输出的 A 份额反推出平台 B 最终份额。"""
    final_A_u = get_x(result)
    final_A_m = get_y(result)
    final_B_u = 1.0 - final_A_u
    final_B_m = 1.0 - final_A_m
    final_B_avg = combined_share(final_B_u, final_B_m)
    return final_A_u, final_A_m, final_B_u, final_B_m, final_B_avg


def _quality_B_from_result(result) -> float:
    """
    model.py 中 q_u/q_m 表示平台 A 相对平台 B 的质量优势。
    平台 B 的质量优势/质量存量用 -q_u 表示。
    """
    return -get_quality(result)


# ============================================================
# 实验 1：非对称网络效应下的补贴方向
# ============================================================

def experiment_asymmetric_subsidy_direction(output_dir: str) -> None:
    """
    分析 alpha != beta 时，用户补贴、商户补贴、均衡补贴的效果差异。

    预期：
    - alpha > beta：用户更依赖商户，商户补贴更有效；
    - beta > alpha：商户更依赖用户，用户补贴更有效。
    """
    out = os.path.join(output_dir, "exp1_asymmetric_subsidy_direction")
    ensure_dir(out)

    scenarios = [
        ("对称网络效应", 1.2, 1.2),
        ("用户更依赖商户", 1.6, 0.5),
        ("商户更依赖用户", 0.5, 1.6),
    ]

    strategies = {
        "用户补贴": (BUDGET, 0.0),
        "商户补贴": (0.0, BUDGET),
        "均衡补贴": (0.5 * BUDGET, 0.5 * BUDGET),
    }

    rows: list[dict[str, Any]] = []

    for scenario_name, alpha, beta in scenarios:
        for strategy_name, (sub_u, sub_m) in strategies.items():
            params = make_params(alpha=alpha, beta=beta, discount=DISCOUNT)

            result = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=T_DEFAULT,
                dt=DT,
                policy=static_policy_B(sub_u, sub_m, 0.0),
            )

            final_A_u, final_A_m, final_B_u, final_B_m, final_B_avg = _final_B_from_result(result)
            profit = get_profit(result)

            rows.append({
                "scenario": scenario_name,
                "alpha": alpha,
                "beta": beta,
                "strategy": strategy_name,
                "subsidy_user": sub_u,
                "subsidy_merchant": sub_m,
                "final_A_user_share": final_A_u,
                "final_A_merchant_share": final_A_m,
                "final_B_user_share": final_B_u,
                "final_B_merchant_share": final_B_m,
                "final_B_average_share": final_B_avg,
                "profit": profit,
                "state": platform_state(final_B_u, final_B_m),
            })

    save_rows_csv(rows, os.path.join(out, "asymmetric_subsidy_direction.csv"))

    scenario_names = [s[0] for s in scenarios]
    strategy_names = list(strategies.keys())
    x = np.arange(len(scenario_names))
    width = 0.24

    plt.figure(figsize=(9, 5))
    for idx, strategy_name in enumerate(strategy_names):
        values = []
        for scenario_name in scenario_names:
            match = [r for r in rows if r["scenario"] == scenario_name and r["strategy"] == strategy_name]
            values.append(match[0]["profit"] if match else np.nan)
        plt.bar(x + (idx - 1) * width, values, width=width, label=strategy_name)
    plt.xticks(x, scenario_names)
    plt.ylabel("累计利润 / 贴现利润")
    plt.title("阶段5-实验1：非对称网络效应下的补贴方向比较")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "asymmetric_subsidy_profit_bar.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    for idx, strategy_name in enumerate(strategy_names):
        values = []
        for scenario_name in scenario_names:
            match = [r for r in rows if r["scenario"] == scenario_name and r["strategy"] == strategy_name]
            values.append(match[0]["final_B_average_share"] if match else np.nan)
        plt.bar(x + (idx - 1) * width, values, width=width, label=strategy_name)
    plt.xticks(x, scenario_names)
    plt.ylabel(r"挑战者最终平均份额 $L_B$")
    plt.title("阶段5-实验1：非对称网络效应下的最终份额比较")
    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "asymmetric_subsidy_share_bar.png"), dpi=300)
    plt.close()


# ============================================================
# 实验 2：最低有效补贴强度
# ============================================================

def experiment_minimum_effective_subsidy(output_dir: str) -> None:
    """
    搜索挑战者平台满足 final_user >= 0.8 且 final_merchant >= 0.8 所需的最低补贴组合。
    """
    out = os.path.join(output_dir, "exp2_minimum_effective_subsidy")
    ensure_dir(out)

    subsidy_values = np.round(np.arange(0.0, 1.21, 0.03), 2)

    rows: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []

    for scenario_name, alpha, beta in ASYMMETRIC_SCENARIOS:
        best: dict[str, Any] | None = None
        share_grid = np.zeros((len(subsidy_values), len(subsidy_values)))

        for i, b_m in enumerate(subsidy_values):
            for j, b_u in enumerate(subsidy_values):
                params = make_params(alpha=alpha, beta=beta, discount=DISCOUNT)

                result = call_simulate(
                    X0_A,
                    Y0_A,
                    params,
                    T=T_DEFAULT,
                    dt=DT,
                    policy=static_policy_B(float(b_u), float(b_m), 0.0),
                )

                final_A_u, final_A_m, final_B_u, final_B_m, avg_share = _final_B_from_result(result)
                total_subsidy = float(b_u + b_m)
                success = int(final_B_u >= 0.8 and final_B_m >= 0.8)

                share_grid[i, j] = avg_share

                row = {
                    "scenario": scenario_name,
                    "alpha": alpha,
                    "beta": beta,
                    "b_user": float(b_u),
                    "b_merchant": float(b_m),
                    "total_subsidy": total_subsidy,
                    "final_A_user_share": final_A_u,
                    "final_A_merchant_share": final_A_m,
                    "final_B_user_share": final_B_u,
                    "final_B_merchant_share": final_B_m,
                    "final_B_average_share": avg_share,
                    "success": success,
                    "state": platform_state(final_B_u, final_B_m),
                }
                rows.append(row)

                if success:
                    if best is None:
                        best = row
                    elif total_subsidy < best["total_subsidy"]:
                        best = row
                    elif np.isclose(total_subsidy, best["total_subsidy"]):
                        if avg_share > best["final_B_average_share"]:
                            best = row

        plt.figure(figsize=(7, 5.8))
        im = plt.imshow(
            share_grid,
            origin="lower",
            aspect="auto",
            extent=[subsidy_values.min(), subsidy_values.max(), subsidy_values.min(), subsidy_values.max()],
            vmin=0,
            vmax=1,
            cmap="viridis",
        )
        plt.colorbar(im, label=r"挑战者最终平均份额 $L_B$")
        plt.xlabel(r"用户侧补贴 $b_B^U$")
        plt.ylabel(r"商户侧补贴 $b_B^M$")
        plt.title(f"阶段5-实验2：最低有效补贴扫描 - {scenario_name}")
        plt.tight_layout()
        safe_name = scenario_name.replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(out, f"minimum_subsidy_heatmap_{safe_name}.png"), dpi=300)
        plt.close()

        if best is not None:
            if best["b_user"] > best["b_merchant"]:
                direction = "用户侧为主"
            elif best["b_user"] < best["b_merchant"]:
                direction = "商户侧为主"
            else:
                direction = "双边均衡"

            summary.append({
                "scenario": scenario_name,
                "alpha": alpha,
                "beta": beta,
                "min_total_subsidy": best["total_subsidy"],
                "best_b_user": best["b_user"],
                "best_b_merchant": best["b_merchant"],
                "main_direction": direction,
                "final_B_user_share": best["final_B_user_share"],
                "final_B_merchant_share": best["final_B_merchant_share"],
                "final_B_average_share": best["final_B_average_share"],
            })
        else:
            summary.append({
                "scenario": scenario_name,
                "alpha": alpha,
                "beta": beta,
                "min_total_subsidy": np.nan,
                "best_b_user": np.nan,
                "best_b_merchant": np.nan,
                "main_direction": "未达到阈值",
                "final_B_user_share": np.nan,
                "final_B_merchant_share": np.nan,
                "final_B_average_share": np.nan,
            })

    save_rows_csv(rows, os.path.join(out, "minimum_effective_subsidy_grid.csv"))
    save_rows_csv(summary, os.path.join(out, "minimum_effective_subsidy_summary.csv"))

    plt.figure(figsize=(9, 5))
    labels = [r["scenario"] for r in summary]
    values = [r["min_total_subsidy"] for r in summary]
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(r"最低总补贴强度 $b_B^U+b_B^M$")
    plt.title("阶段5-实验2：不同网络效应结构下的最低有效补贴")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "minimum_effective_subsidy_bar.png"), dpi=300)
    plt.close()


# ============================================================
# 实验 3：供给不足惩罚强度敏感性
# ============================================================

def experiment_shortage_penalty_sensitivity(output_dir: str) -> None:
    """
    扫描供给不足惩罚强度 theta，比较不同补贴策略的稳健性。
    """
    out = os.path.join(output_dir, "exp3_shortage_penalty_sensitivity")
    ensure_dir(out)

    theta_values = np.linspace(0.0, 2.0, 21)

    strategies = {
        "只补贴用户": (BUDGET, 0.0, 0.0),
        "只补贴商户": (0.0, BUDGET, 0.0),
        "均衡补贴": (0.5 * BUDGET, 0.5 * BUDGET, 0.0),
        "偏商户补贴": (0.25 * BUDGET, 0.75 * BUDGET, 0.0),
    }

    rows: list[dict[str, Any]] = []

    for theta in theta_values:
        for strategy_name, (sub_u, sub_m, inv) in strategies.items():
            params = make_params(
                alpha=MEDIUM_ALPHA,
                beta=MEDIUM_BETA,
                N_U=N_U,
                N_M=N_M,
                rho=RHO,
                theta=float(theta),
                epsilon=EPS,
                discount=DISCOUNT,
                shortage_enabled=True,
                shortage_rho=RHO,
                shortage_buffer=EPS,
            )

            result = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=T_DEFAULT,
                dt=DT,
                policy=static_policy_B(sub_u, sub_m, inv),
            )

            final_A_u, final_A_m, final_B_u, final_B_m, final_B_avg = _final_B_from_result(result)
            max_shortage, avg_shortage = get_shortage_B(result)
            profit = get_profit(result)

            rows.append({
                "theta": float(theta),
                "strategy": strategy_name,
                "subsidy_user": sub_u,
                "subsidy_merchant": sub_m,
                "final_A_user_share": final_A_u,
                "final_A_merchant_share": final_A_m,
                "final_B_user_share": final_B_u,
                "final_B_merchant_share": final_B_m,
                "final_B_average_share": final_B_avg,
                "max_shortage": max_shortage,
                "avg_shortage": avg_shortage,
                "profit": profit,
                "state": platform_state(final_B_u, final_B_m),
            })

    save_rows_csv(rows, os.path.join(out, "shortage_penalty_sensitivity.csv"))

    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        plt.plot([r["theta"] for r in sub], [r["profit"] for r in sub], marker="o", label=strategy_name)
    plt.xlabel(r"供给不足惩罚强度 $\theta$")
    plt.ylabel("累计利润 / 贴现利润")
    plt.title("阶段5-实验3：供给不足惩罚强度对利润的影响")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "shortage_penalty_profit_lines.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        plt.plot([r["theta"] for r in sub], [r["final_B_average_share"] for r in sub], marker="o", label=strategy_name)
    plt.xlabel(r"供给不足惩罚强度 $\theta$")
    plt.ylabel(r"挑战者最终平均份额 $L_B$")
    plt.title("阶段5-实验3：供给不足惩罚强度对最终份额的影响")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "shortage_penalty_share_lines.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        plt.plot([r["theta"] for r in sub], [r["max_shortage"] for r in sub], marker="o", label=strategy_name)
    plt.xlabel(r"供给不足惩罚强度 $\theta$")
    plt.ylabel("最大供给不足惩罚")
    plt.title("阶段5-实验3：供给不足惩罚强度对最大短缺的影响")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "shortage_penalty_max_shortage_lines.png"), dpi=300)
    plt.close()


# ============================================================
# 实验 4：服务质量投资效率与衰减率敏感性
# ============================================================

def _dynamic_quality_policy(t: float, state, p: Params):
    """
    报告 5.6 中的动态策略，基于平台 B 的用户-商户比例进行分配。
    """
    if isinstance(state, dict):
        x_A = float(state.get("x", state.get("u", X0_A)))
        y_A = float(state.get("y", state.get("m", Y0_A)))
    else:
        x_A = float(state[0])
        y_A = float(state[1])

    x_B = 1.0 - x_A
    y_B = 1.0 - y_A
    ratio = N_U * x_B / (N_M * y_B + EPS)

    if ratio > RHO:
        su, sm, inv = 0.2 * BUDGET, 0.6 * BUDGET, 0.2 * BUDGET
    elif ratio < 0.8 * RHO:
        su, sm, inv = 0.6 * BUDGET, 0.2 * BUDGET, 0.2 * BUDGET
    else:
        su, sm, inv = 0.3 * BUDGET, 0.3 * BUDGET, 0.4 * BUDGET

    return {
        "ds_u_B": su,
        "ds_m_B": sm,
        "inv_u_B": inv,
        "inv_m_B": inv,
    }


def experiment_quality_investment_sensitivity(output_dir: str) -> None:
    """
    扫描服务质量投资效率 lambda_q 与质量衰减率 d。
    """
    out = os.path.join(output_dir, "exp4_quality_investment_sensitivity")
    ensure_dir(out)

    lambda_q_values = [0.02, 0.05, 0.08, 0.10]
    decay_values = [0.005, 0.01, 0.02, 0.04]

    fixed_strategies = {
        "贪心策略": (0.8 * BUDGET, 0.1 * BUDGET, 0.1 * BUDGET),
        "长期策略": (0.3 * BUDGET, 0.4 * BUDGET, 0.3 * BUDGET),
        "纯质量投资": (0.0, 0.0, BUDGET),
    }

    strategy_names = ["贪心策略", "长期策略", "动态策略", "纯质量投资"]
    rows: list[dict[str, Any]] = []

    for lambda_q in lambda_q_values:
        for decay in decay_values:
            for strategy_name in strategy_names:
                if strategy_name == "动态策略":
                    policy = _dynamic_quality_policy
                else:
                    su, sm, inv = fixed_strategies[strategy_name]
                    policy = static_policy_B(su, sm, inv)

                params = make_params(
                    alpha=MEDIUM_ALPHA,
                    beta=MEDIUM_BETA,
                    shortage_enabled=True,
                    N_U=N_U,
                    N_M=N_M,
                    rho=RHO,
                    theta=1.0,
                    epsilon=EPS,
                    shortage_rho=RHO,
                    shortage_buffer=EPS,
                    lambda_q=float(lambda_q),
                    quality_decay=float(decay),
                    qmax=Q_MAX,
                    discount=DISCOUNT,
                )

                result = call_simulate(
                    X0_A,
                    Y0_A,
                    params,
                    T=T_QUALITY,
                    dt=DT,
                    policy=policy,
                )

                final_A_u, final_A_m, final_B_u, final_B_m, final_B_avg = _final_B_from_result(result)
                final_q_B = _quality_B_from_result(result)
                max_shortage, avg_shortage = get_shortage_B(result)
                profit = get_profit(result)

                rows.append({
                    "lambda_q": float(lambda_q),
                    "decay": float(decay),
                    "strategy": strategy_name,
                    "final_A_user_share": final_A_u,
                    "final_A_merchant_share": final_A_m,
                    "final_B_user_share": final_B_u,
                    "final_B_merchant_share": final_B_m,
                    "final_B_average_share": final_B_avg,
                    "final_quality_B": final_q_B,
                    "max_shortage": max_shortage,
                    "avg_shortage": avg_shortage,
                    "profit": profit,
                    "state": platform_state(final_B_u, final_B_m),
                })

    save_rows_csv(rows, os.path.join(out, "quality_investment_sensitivity.csv"))

    for strategy_name in strategy_names:
        grid = np.full((len(decay_values), len(lambda_q_values)), np.nan)
        for i, decay in enumerate(decay_values):
            for j, lambda_q in enumerate(lambda_q_values):
                match = [
                    r for r in rows
                    if r["strategy"] == strategy_name
                    and np.isclose(r["lambda_q"], lambda_q)
                    and np.isclose(r["decay"], decay)
                ]
                if match:
                    grid[i, j] = match[0]["profit"]

        plt.figure(figsize=(7, 5))
        im = plt.imshow(
            grid,
            origin="lower",
            aspect="auto",
            extent=[min(lambda_q_values), max(lambda_q_values), min(decay_values), max(decay_values)],
            cmap="viridis",
        )
        plt.colorbar(im, label="累计利润 / 贴现利润")
        plt.xlabel(r"投资转化效率 $\lambda_q$")
        plt.ylabel(r"质量衰减率 $d$")
        plt.title(f"阶段5-实验4：{strategy_name}的质量投资参数敏感性")

        for i, decay in enumerate(decay_values):
            for j, lambda_q in enumerate(lambda_q_values):
                value = grid[i, j]
                if not np.isnan(value):
                    plt.text(lambda_q, decay, f"{value:.1f}", ha="center", va="center", fontsize=8)

        plt.tight_layout()
        safe_name = strategy_name.replace("/", "_").replace("\\", "_")
        plt.savefig(os.path.join(out, f"quality_investment_profit_heatmap_{safe_name}.png"), dpi=300)
        plt.close()

    baseline_rows = [
        r for r in rows
        if np.isclose(r["lambda_q"], LAMBDA_Q_BASE)
        and np.isclose(r["decay"], D_BASE)
    ]

    plt.figure(figsize=(9, 5))
    labels = [r["strategy"] for r in baseline_rows]
    values = [r["profit"] for r in baseline_rows]
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels)
    plt.ylabel("累计利润 / 贴现利润")
    plt.title(r"阶段5-实验4：基准参数 $\lambda_q=0.05,d=0.01$ 下策略利润对比")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "quality_investment_baseline_profit_bar.png"), dpi=300)
    plt.close()


# ============================================================
# 实验 5：临界条件汇总表
# ============================================================

def write_critical_condition_summary(output_dir: str) -> None:
    """
    写出报告 5.7 中使用的临界条件汇总表。
    这些数值来自前面阶段的实验结论，不在 stage5 中重复仿真。
    """
    out = os.path.join(output_dir, "exp5_critical_condition_summary")
    ensure_dir(out)

    rows = [
        {
            "type": "网络效应锁定门槛",
            "condition": "弱网络效应 alpha=beta=0.3, u_A(0)=m_A(0)=0.85",
            "result": "最终用户份额约 0.502，锁定指数约 0.252",
            "meaning": "弱网络效应无法维持高初始份额，系统回到共存附近",
        },
        {
            "type": "网络效应锁定门槛",
            "condition": "中等网络效应 alpha=beta=0.8, u_A(0)=m_A(0)=0.85",
            "result": "最终用户份额约 0.675，锁定指数约 0.455",
            "meaning": "中等网络效应会放大初始优势，但不足以形成严格锁定",
        },
        {
            "type": "网络效应锁定门槛",
            "condition": "强网络效应 alpha=beta=1.5, u_A(0)=m_A(0)=0.55",
            "result": "最终用户份额约 0.956，锁定指数约 0.913",
            "meaning": "强网络效应会把小幅初始领先放大为市场锁定",
        },
        {
            "type": "服务质量突破门槛",
            "condition": "中等网络效应 alpha=beta=0.8",
            "result": "平台 B 打破锁定需要约 Delta q = 0.06",
            "meaning": "中等网络效应下，较小质量优势即可打破初始领先",
        },
        {
            "type": "服务质量突破门槛",
            "condition": "强网络效应 alpha=beta=1.5",
            "result": "平台 B 打破锁定需要约 Delta q = 0.35",
            "meaning": "强网络效应显著提高挑战者质量突破门槛",
        },
        {
            "type": "切换成本影响",
            "condition": "中等网络效应，s 从 0 增加到 1.5",
            "result": "质量突破阈值从 0.06 上升到 0.69",
            "meaning": "切换成本强化已有平台锁定",
        },
        {
            "type": "切换成本影响",
            "condition": "强网络效应，s 从 0 增加到 1.5",
            "result": "质量突破阈值从 0.35 上升到 1.11",
            "meaning": "强网络效应和高切换成本叠加，显著提高逆袭难度",
        },
        {
            "type": "低预算补贴",
            "condition": "B=0.10",
            "result": "只补贴用户可使 B 用户侧先跨过 0.5，略优于均衡补贴",
            "meaning": "低预算下集中撬动一侧可能略有启动优势",
        },
        {
            "type": "低预算补贴",
            "condition": "B=0.20",
            "result": "均衡补贴最终平均份额略优于单侧补贴",
            "meaning": "预算提高后，同步推动两侧更有利于形成双边正反馈",
        },
        {
            "type": "补贴退出",
            "condition": "B=1.20, T_s=8.95",
            "result": "贴现利润约 164.49，略高于持续均衡补贴 164.29",
            "meaning": "适时退出可减少低边际收益阶段的补贴浪费",
        },
        {
            "type": "供给约束",
            "condition": "N_U=1000, N_M=50, rho=10.0, theta=1.0, budget=0.8",
            "result": "只补贴商户利润最高，只补贴用户利润最低",
            "meaning": "用户规模远大于商户规模时，应优先补足供给侧",
        },
        {
            "type": "质量投资",
            "condition": "lambda_q=0.05, d=0.01, qmax=3.0",
            "result": "动态策略和纯质量投资策略在长期表现较稳健",
            "meaning": "质量投资具有累积效应，但也受转化效率和衰减率约束",
        },
    ]

    save_rows_csv(rows, os.path.join(out, "critical_condition_summary.csv"))


# ============================================================
# Stage 5 总入口
# ============================================================

def run_stage5(output_root: str) -> None:
    """
    stage5 总入口。
    """
    setup_matplotlib()

    out = os.path.join(output_root, "stage5_sensitivity")
    ensure_dir(out)

    print("  阶段5-实验1：非对称网络效应下的补贴方向...")
    experiment_asymmetric_subsidy_direction(out)

    print("  阶段5-实验2：最低有效补贴强度...")
    experiment_minimum_effective_subsidy(out)

    print("  阶段5-实验3：供给不足惩罚敏感性...")
    experiment_shortage_penalty_sensitivity(out)

    print("  阶段5-实验4：服务质量投资效率与衰减率敏感性...")
    experiment_quality_investment_sensitivity(out)

    print("  阶段5-实验5：临界条件汇总表...")
    write_critical_condition_summary(out)

    print("  阶段5全部完成。")
