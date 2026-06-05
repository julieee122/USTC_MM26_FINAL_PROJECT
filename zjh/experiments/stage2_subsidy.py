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


def run_stage2(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage2_subsidy")
    ensure_dir(out)

    x0 = 0.35
    y0 = 0.35

    p = Params(alpha=1.2, beta=1.0)

    strategies = {
        "无补贴": zero_policy,
        "用户补贴": make_decay_subsidy_policy(
            su0=0.85,
            sm0=0.0,
            decay_u=0.06,
            decay_m=0.06,
        ),
        "商户补贴": make_decay_subsidy_policy(
            su0=0.0,
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

    results = {}
    metrics = []

    for name, policy in strategies.items():
        res = simulate_path(x0, y0, p, T=80.0, dt=0.03, policy=policy)
        results[name] = res
        metrics.append(summarize_result(name, res))
        save_timeseries_csv(res, os.path.join(out, f"stage2_{name}_timeseries.csv"))

    save_metrics_csv(metrics, os.path.join(out, "stage2_metrics.csv"))

    # ============================================================
    # 2.1 不同补贴策略下的用户份额和商户份额轨迹
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, res in results.items():
        axes[0].plot(res["t"], res["x"], label=f"{name}, x*={res['x'][-1]:.2f}")
        axes[1].plot(res["t"], res["y"], label=f"{name}, y*={res['y'][-1]:.2f}")

    axes[0].set_title("阶段2：不同补贴策略下的用户份额演化")
    axes[0].set_xlabel("时间")
    axes[0].set_ylabel(r"平台 A 用户份额 $x(t)$")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("阶段2：不同补贴策略下的商户份额演化")
    axes[1].set_xlabel("时间")
    axes[1].set_ylabel(r"平台 A 商户份额 $y(t)$")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.savefig(
        os.path.join(out, "stage2_1_2_3_subsidy_trajectories.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 2.2 补贴成本收益比较
    # ============================================================

    names = [m["name"] for m in metrics]
    profits = [m["cum_profit"] for m in metrics]
    revenues = [m["cum_revenue"] for m in metrics]
    costs = [m["cum_total_cost"] for m in metrics]

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, revenues, width, label="累计收益")
    plt.bar(x, costs, width, label="累计成本")
    plt.bar(x + width, profits, width, label="累计净收益")
    plt.xticks(x, names)
    plt.title("阶段2-实验4：不同补贴策略的成本收益比较")
    plt.ylabel("累计值")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "stage2_4_cost_benefit_bar.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ============================================================
    # 2.3 用户补贴与商户补贴强度二维扫描
    # ============================================================

    su_values = np.linspace(0.0, 1.2, 41)
    sm_values = np.linspace(0.0, 1.2, 41)

    final_x = np.zeros((len(sm_values), len(su_values)))
    total_profit = np.zeros_like(final_x)

    for i, sm in enumerate(sm_values):
        for j, su in enumerate(su_values):
            policy = make_decay_subsidy_policy(
                su0=su,
                sm0=sm,
                decay_u=0.06,
                decay_m=0.06,
            )

            res = simulate_path(x0, y0, p, T=80.0, dt=0.05, policy=policy)

            final_x[i, j] = res["x"][-1]
            total_profit[i, j] = res["cum_profit"][-1]

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        final_x,
        extent=[su_values.min(), su_values.max(), sm_values.min(), sm_values.max()],
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis"
    )
    plt.colorbar(im, label=r"最终用户份额 $x^*$")
    plt.title("阶段2：用户补贴与商户补贴对最终份额的影响")
    plt.xlabel("初始用户补贴强度")
    plt.ylabel("初始商户补贴强度")

    plt.savefig(
        os.path.join(out, "stage2_subsidy_grid_final_share.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        total_profit,
        extent=[su_values.min(), su_values.max(), sm_values.min(), sm_values.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis"
    )
    plt.colorbar(im, label="累计净收益")
    plt.title("阶段2：补贴组合的成本收益扫描")
    plt.xlabel("初始用户补贴强度")
    plt.ylabel("初始商户补贴强度")

    plt.savefig(
        os.path.join(out, "stage2_subsidy_grid_profit.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()