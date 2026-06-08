import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Params
from src.model import simulate_path
from src.policies import zero_policy, make_decay_subsidy_policy
from src.utils import (
    ensure_dir,
    setup_matplotlib,
    save_metrics_csv,
    save_timeseries_csv,
    summarize_result,
)


def run_stage3(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage3_shortage")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    p = Params(
        alpha=1.2,
        beta=1.0,
        shortage_enabled=True,
        shortage_rho=5.0,
        shortage_buffer=0.02,
    )

    strategies = {
        "无补贴": zero_policy,
        "用户补贴_需求快速增长": make_decay_subsidy_policy(
            su0=1.10,
            sm0=0.00,
            decay_u=0.04,
            decay_m=0.06,
        ),
        "商户补贴_长期稳定": make_decay_subsidy_policy(
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

    results = {}
    metrics = []

    for name, policy in strategies.items():
        res = simulate_path(x0, y0, p, T=90.0, dt=0.03, policy=policy)
        results[name] = res
        metrics.append(summarize_result(name, res))
        save_timeseries_csv(res, os.path.join(out, f"stage3_{name}_timeseries.csv"))

    save_metrics_csv(metrics, os.path.join(out, "stage3_metrics.csv"))

    # ============================================================
    # 3.1 用户份额、商户份额、供给不足程度
    # ============================================================

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for name, res in results.items():
        axes[0].plot(res["t"], res["x"], label=f"{name}, x*={res['x'][-1]:.2f}")
        axes[1].plot(res["t"], res["y"], label=f"{name}, y*={res['y'][-1]:.2f}")
        axes[2].plot(
            res["t"],
            res["shortage_A"],
            label=f"{name}, max={np.max(res['shortage_A']):.2f}"
        )

    axes[0].set_title("阶段3：供给不足惩罚下的用户份额演化")
    axes[0].set_ylabel(r"$x(t)$")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("阶段3：供给不足惩罚下的商户份额演化")
    axes[1].set_ylabel(r"$y(t)$")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("阶段3：平台 A 的供给不足程度")
    axes[2].set_xlabel("时间")
    axes[2].set_ylabel("shortage_A")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.savefig(
        os.path.join(out, "stage3_shortage_dynamics.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 3.2 成本收益与供给不足比较
    # ============================================================

    names = [m["name"] for m in metrics]
    profits = [m["cum_profit"] for m in metrics]
    max_shortage = [m["max_shortage_A"] for m in metrics]
    final_x = [m["final_x"] for m in metrics]

    x = np.arange(len(names))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].bar(x, final_x)
    axes[0].set_title("最终用户份额")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, max_shortage)
    axes[1].set_title("最大供给不足程度")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20)
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, profits)
    axes[2].set_title("累计净收益")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20)
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.suptitle("阶段3：供给不足约束下的策略比较", fontsize=16)
    fig.savefig(
        os.path.join(out, "stage3_strategy_comparison.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)