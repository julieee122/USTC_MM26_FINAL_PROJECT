import os
import numpy as np
import matplotlib.pyplot as plt

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
    combined_share,
    static_policy_B,
)


DT = 0.05
T = 200.0

# 统一报告口径：x,y 表示平台 A 份额；平台 A 初始占优，平台 B 为挑战者。
X0_A = 0.8
Y0_A = 0.8

ALPHA = 0.8
BETA = 0.8

BUDGET = 0.8

N_U = 1000
N_M = 50
RHO = 10.0
THETA = 1.0
EPS = 1e-6

REPORT_MODEL_KWARGS = {
    "lambda_u": 2.6,
    "lambda_m": 2.6,
    "discount": 0.98,
    "use_report_profit": True,
    "profit_mu": 5.0,
    "shortage_mode": "absolute_B",
    "quality_base_effect_scale": 1.0,
    "quality_stock_effect_scale": 1.0,
}

def _final_B_from_result(res):
    xA = get_x(res)
    yA = get_y(res)
    xB = 1.0 - xA
    yB = 1.0 - yA
    return xA, yA, xB, yB, combined_share(xB, yB)


def run_stage3(output_root: str):
    """
    阶段3：供给约束下的补贴策略修正。

    对应报告 5.5：
    - 中等网络效应 alpha=beta=0.8；
    - 平台 A 初始占优 x0=y0=0.8；
    - 平台 B 为挑战者并采取补贴策略；
    - N_U=1000, N_M=50, rho=10.0, theta=1.0；
    - 预算固定为 B=0.8。
    """
    setup_matplotlib()

    out = os.path.join(output_root, "stage3_shortage")
    ensure_dir(out)

    params = make_params(
        alpha=ALPHA,
        beta=BETA,
        shortage_enabled=True,
        N_U=N_U,
        N_M=N_M,
        rho=RHO,
        theta=THETA,
        epsilon=EPS,
        shortage_rho=RHO,
        shortage_buffer=EPS,
        **REPORT_MODEL_KWARGS,
    )

    strategies = {
        "只补贴用户": static_policy_B(BUDGET, 0.0, 0.0),
        "只补贴商户": static_policy_B(0.0, BUDGET, 0.0),
        "均衡补贴": static_policy_B(0.5 * BUDGET, 0.5 * BUDGET, 0.0),
        "偏商户补贴": static_policy_B(0.25 * BUDGET, 0.75 * BUDGET, 0.0),
    }

    results = {}
    rows = []

    for name, policy in strategies.items():
        res = call_simulate(
            X0_A,
            Y0_A,
            params,
            T=T,
            dt=DT,
            policy=policy,
        )

        results[name] = res

        final_A_u, final_A_m, final_B_u, final_B_m, LB = _final_B_from_result(res)
        max_shortage, avg_shortage = get_shortage_B(res)
        profit = get_profit(res)

        rows.append({
            "strategy": name,
            "final_A_user_share": final_A_u,
            "final_A_merchant_share": final_A_m,
            "final_B_user_share": final_B_u,
            "final_B_merchant_share": final_B_m,
            "final_B_average_share": LB,
            "max_shortage_B": max_shortage,
            "avg_shortage_B": avg_shortage,
            "profit": profit,
        })

    save_rows_csv(rows, os.path.join(out, "stage3_shortage_metrics.csv"))

    # 图 1：平台 B 平均份额动态
    plt.figure(figsize=(9, 5))
    for name, res in results.items():
        xA = np.asarray(res["x"], dtype=float)
        yA = np.asarray(res["y"], dtype=float)
        t = np.asarray(res["t"], dtype=float)
        xB = 1.0 - xA
        yB = 1.0 - yA
        avg = 0.5 * (xB + yB)
        plt.plot(t, avg, label=f"{name}, L_B={avg[-1]:.3f}")

    plt.axhline(0.5, linestyle="--", linewidth=1.0, label="反超阈值 0.5")
    plt.xlabel("时间")
    plt.ylabel(r"平台 B 平均份额 $L_B$")
    plt.title("阶段3：供给不足惩罚下的平台 B 平均份额变化")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage3_shortage_average_share.png"), dpi=300)
    plt.close()

    # 图 2：平台 B 供给不足惩罚动态
    plt.figure(figsize=(9, 5))
    for name, res in results.items():
        shortage = np.asarray(res.get("shortage_B", np.zeros_like(res["t"])), dtype=float)
        t = np.asarray(res["t"], dtype=float)
        plt.plot(t, shortage, label=f"{name}, max={np.max(shortage):.2f}")

    plt.xlabel("时间")
    plt.ylabel(r"供给不足惩罚 $C_B$")
    plt.title("阶段3：不同补贴策略下的供给不足惩罚")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage3_shortage_penalty.png"), dpi=300)
    plt.close()

    # 图 3：最终结果柱状图
    labels = [r["strategy"] for r in rows]
    x_pos = np.arange(len(labels))
    width = 0.22

    plt.figure(figsize=(10, 5))
    plt.bar(
        x_pos - width,
        [r["final_B_average_share"] for r in rows],
        width,
        label=r"最终平均份额 $L_B$",
    )
    plt.bar(
        x_pos,
        [r["avg_shortage_B"] for r in rows],
        width,
        label=r"平均供给不足 $C_B$",
    )
    plt.bar(
        x_pos + width,
        [r["profit"] for r in rows],
        width,
        label="贴现利润",
    )

    plt.xticks(x_pos, labels)
    plt.title("阶段3：供给约束下的策略比较")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage3_shortage_strategy_comparison.png"), dpi=300)
    plt.close()

    print("  阶段3完成。")
