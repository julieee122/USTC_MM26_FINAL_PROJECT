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
    get_shortage,
    get_quality,
    combined_share,
    static_policy,
)


DT = 0.05
T = 300.0

X0_B = 0.2
Y0_B = 0.2

BUDGET = 0.8

N_U = 1000
N_M = 50
RHO = 10.0
THETA = 1.0
EPS = 1e-6

LAMBDA_Q = 0.05
QUALITY_DECAY = 0.01
Q_MAX = 3.0


def dynamic_report_policy(t, state, p):
    """
    报告 5.6 中的动态策略：
    - 供给不足较强：偏商户补贴；
    - 用户侧不足：偏用户补贴；
    - 否则提高质量投资比例。
    """
    if isinstance(state, dict):
        x = float(state.get("x", state.get("u", X0_B)))
        y = float(state.get("y", state.get("m", Y0_B)))
    else:
        x = float(state[0])
        y = float(state[1])

    ratio = N_U * x / (N_M * y + EPS)

    if ratio > RHO:
        su, sm, inv = 0.2 * BUDGET, 0.6 * BUDGET, 0.2 * BUDGET
    elif ratio < 0.8 * RHO:
        su, sm, inv = 0.6 * BUDGET, 0.2 * BUDGET, 0.2 * BUDGET
    else:
        su, sm, inv = 0.3 * BUDGET, 0.3 * BUDGET, 0.4 * BUDGET

    return {
        "ds_u": su,
        "ds_m": sm,
        "inv_u": inv,
        "inv_m": inv,
    }

def run_stage4(output_root: str):
    """
    阶段4：服务质量投资与混合策略。

    对应报告 5.6：
    - 加入供给不足惩罚和服务质量更新；
    - N_U=1000, N_M=50, rho=10.0；
    - B=0.8；
    - lambda_q=0.05, d=0.01, qmax=3.0；
    - T=300；
    - 比较贪心、长期、动态、纯质量投资。
    """
    setup_matplotlib()

    out = os.path.join(output_root, "stage4_quality_investment")
    ensure_dir(out)

    network_cases = [
        ("中等", 0.8, 0.8),
        ("强", 1.5, 1.5),
    ]

    strategies = {
        "贪心策略": static_policy(0.8 * BUDGET, 0.1 * BUDGET, 0.1 * BUDGET),
        "长期策略": static_policy(0.3 * BUDGET, 0.4 * BUDGET, 0.3 * BUDGET),
        "动态策略": dynamic_report_policy,
        "纯质量投资": static_policy(0.0, 0.0, BUDGET),
    }

    all_rows = []
    all_results = {}

    for network_name, alpha, beta in network_cases:
        params = make_params(
            alpha=alpha,
            beta=beta,
            shortage_enabled=True,
            N_U=N_U,
            N_M=N_M,
            rho=RHO,
            theta=THETA,
            epsilon=EPS,
            shortage_rho=RHO,
            shortage_buffer=EPS,
            lambda_q=LAMBDA_Q,
            quality_decay=QUALITY_DECAY,
            qmax=Q_MAX,
        )

        for strategy_name, policy in strategies.items():
            res = call_simulate(
                X0_B,
                Y0_B,
                params,
                T=T,
                dt=DT,
                policy=policy,
            )

            all_results[(network_name, strategy_name)] = res

            final_u = get_x(res)
            final_m = get_y(res)
            final_q = get_quality(res)
            max_shortage, avg_shortage = get_shortage(res)
            profit = get_profit(res)

            all_rows.append({
                "network": network_name,
                "alpha": alpha,
                "beta": beta,
                "strategy": strategy_name,
                "final_B_user_share": final_u,
                "final_B_merchant_share": final_m,
                "final_B_average_share": combined_share(final_u, final_m),
                "final_quality": final_q,
                "max_shortage_B": max_shortage,
                "avg_shortage_B": avg_shortage,
                "profit": profit,
            })

    save_rows_csv(all_rows, os.path.join(out, "stage4_quality_investment_metrics.csv"))

    # 图 1：中等网络效应下平台 B 平均份额
    plt.figure(figsize=(9, 5))
    for strategy_name in strategies:
        res = all_results[("中等", strategy_name)]
        t = np.asarray(res["t"], dtype=float)
        x = np.asarray(res["x"], dtype=float)
        y = np.asarray(res["y"], dtype=float)
        avg = 0.5 * (x + y)
        plt.plot(t, avg, label=f"{strategy_name}, L_B={avg[-1]:.3f}")

    plt.axhline(0.5, linestyle="--", linewidth=1.0, label="反超阈值 0.5")
    plt.xlabel("时间")
    plt.ylabel(r"平台 B 平均份额 $L_B$")
    plt.title("阶段4：服务质量投资策略下的平台 B 平均份额")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage4_quality_share_dynamics.png"), dpi=300)
    plt.close()

    # 图 2：服务质量动态
    plt.figure(figsize=(9, 5))
    for strategy_name in strategies:
        res = all_results[("中等", strategy_name)]
        t = np.asarray(res["t"], dtype=float)

        if "q_u" in res:
            q = np.asarray(res["q_u"], dtype=float)
        elif "quality_u" in res:
            q = np.asarray(res["quality_u"], dtype=float)
        else:
            q = np.zeros_like(t)

        plt.plot(t, q, label=f"{strategy_name}, q(T)={q[-1]:.3f}")

    plt.xlabel("时间")
    plt.ylabel(r"用户侧服务质量 $q_B^U(t)$")
    plt.title("阶段4：服务质量投资策略下的质量积累")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage4_quality_stock_dynamics.png"), dpi=300)
    plt.close()

    # 图 3：中等与强网络效应下策略利润比较
    for network_name, _, _ in network_cases:
        rows = [r for r in all_rows if r["network"] == network_name]
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
            [r["final_quality"] for r in rows],
            width,
            label="最终质量",
        )
        plt.bar(
            x_pos + width,
            [r["profit"] for r in rows],
            width,
            label="贴现利润",
        )

        plt.xticks(x_pos, labels)
        plt.title(f"阶段4：{network_name}网络效应下策略综合比较")
        plt.grid(axis="y", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(out, f"stage4_strategy_comparison_{network_name}.png"),
            dpi=300,
        )
        plt.close()

    print("  阶段4完成。")