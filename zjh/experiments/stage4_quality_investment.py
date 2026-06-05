import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Params
from src.model import simulate_path
from src.policies import (
    greedy_quality_policy,
    long_term_quality_policy,
    dynamic_quality_policy,
)
from src.utils import (
    ensure_dir,
    setup_matplotlib,
    save_metrics_csv,
    save_timeseries_csv,
    summarize_result,
)


def run_stage4(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage4_quality_investment")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    p = Params(
        alpha=1.2,
        beta=1.0,
        shortage_enabled=True,
        shortage_rho=4.5,
        shortage_buffer=0.02,
        quality_decay=0.045,
        invest_eff_u=0.28,
        invest_eff_m=0.28,
    )

    strategies = {
        "贪心策略": greedy_quality_policy,
        "长期策略": long_term_quality_policy,
        "动态策略": dynamic_quality_policy,
    }

    results = {}
    metrics = []

    for name, policy in strategies.items():
        res = simulate_path(x0, y0, p, T=100.0, dt=0.03, policy=policy)
        results[name] = res
        metrics.append(summarize_result(name, res))
        save_timeseries_csv(res, os.path.join(out, f"stage4_{name}_timeseries.csv"))

    save_metrics_csv(metrics, os.path.join(out, "stage4_metrics.csv"))

    # ============================================================
    # 4.1 用户份额与商户份额
    # ============================================================

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for name, res in results.items():
        axes[0].plot(res["t"], res["x"], label=f"{name}, x*={res['x'][-1]:.2f}")
        axes[1].plot(res["t"], res["y"], label=f"{name}, y*={res['y'][-1]:.2f}")

    axes[0].set_title("阶段4：服务质量投资策略下的用户份额演化")
    axes[0].set_ylabel(r"$x(t)$")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("阶段4：服务质量投资策略下的商户份额演化")
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel(r"$y(t)$")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(
        os.path.join(out, "stage4_quality_investment_share_dynamics.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 4.2 服务质量优势动态
    # ============================================================

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    for name, res in results.items():
        axes[0].plot(res["t"], res["q_u"], label=name)
        axes[1].plot(res["t"], res["q_m"], label=name)

    axes[0].set_title("阶段4：用户侧服务质量优势动态")
    axes[0].set_ylabel(r"$q_U(t)$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("阶段4：商户侧服务质量优势动态")
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel(r"$q_M(t)$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(
        os.path.join(out, "stage4_quality_stock_dynamics.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 4.3 成本收益比较
    # ============================================================

    names = [m["name"] for m in metrics]
    final_x = [m["final_x"] for m in metrics]
    final_y = [m["final_y"] for m in metrics]
    invest_cost = [m["cum_invest_cost"] for m in metrics]
    profits = [m["cum_profit"] for m in metrics]

    x = np.arange(len(names))
    width = 0.2

    plt.figure(figsize=(11, 6))
    plt.bar(x - 1.5 * width, final_x, width, label=r"最终用户份额 $x^*$")
    plt.bar(x - 0.5 * width, final_y, width, label=r"最终商户份额 $y^*$")
    plt.bar(x + 0.5 * width, invest_cost, width, label="累计投资成本")
    plt.bar(x + 1.5 * width, profits, width, label="累计净收益")

    plt.xticks(x, names)
    plt.title("阶段4：服务质量投资策略综合比较")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "stage4_strategy_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()