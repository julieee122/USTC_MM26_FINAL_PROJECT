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
    get_quality,
    combined_share,
    static_policy_B,
)


DT = 0.05
T = 300.0

# 统一报告口径：x,y 表示平台 A 份额；平台 A 初始占优，平台 B 为挑战者。
X0_A = 0.8
Y0_A = 0.8

BUDGET = 0.8

N_U = 1000
N_M = 50
RHO = 10.0
THETA = 1.0
EPS = 1e-6

LAMBDA_Q = 0.05
QUALITY_DECAY = 0.01
Q_MAX = 3.0

REPORT_MODEL_KWARGS = {
    "lambda_u": 2.6,
    "lambda_m": 2.6,
    "discount": 0.98,
    "use_report_profit": True,
    "profit_mu": 5.0,
    "shortage_mode": "absolute_B",
    "quality_base_effect_scale": 1.0,
    "quality_stock_effect_scale": 8.0,
}

def _final_B_from_result(res):
    xA = get_x(res)
    yA = get_y(res)
    xB = 1.0 - xA
    yB = 1.0 - yA
    return xA, yA, xB, yB, combined_share(xB, yB)


def _quality_B_from_result(res):
    """
    model.py 中 q_u/q_m 表示 A 相对 B 的质量优势；
    因此 B 的质量优势/质量存量展示为 -q_u。
    """
    q = get_quality(res)
    return -q


def dynamic_report_policy(t, state, p):
    """
    报告 5.6 中的动态策略。
    根据平台 B 的实际用户-商户比例调整预算分配：
    - 供给不足较强：偏商户补贴；
    - 用户侧不足：偏用户补贴；
    - 否则提高质量投资比例。
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


def run_stage4(output_root: str):
    """
    阶段4：服务质量投资与混合策略。

    对应报告 5.6：
    - 平台 A 初始占优，平台 B 为挑战者；
    - 加入供给不足惩罚和服务质量更新；
    - N_U=1000, N_M=50, rho=10.0；
    - B=0.8；
    - lambda_q=0.05, d=0.01, qmax=3.0；
    - T=300。
    """
    setup_matplotlib()

    out = os.path.join(output_root, "stage4_quality_investment")
    ensure_dir(out)

    network_cases = [
        ("中等", 0.8, 0.8),
        ("强", 1.5, 1.5),
    ]

    strategies = {
        "贪心策略": static_policy_B(0.8 * BUDGET, 0.1 * BUDGET, 0.1 * BUDGET),
        "长期策略": static_policy_B(0.3 * BUDGET, 0.4 * BUDGET, 0.3 * BUDGET),
        "动态策略": dynamic_report_policy,
        "纯质量投资": static_policy_B(0.0, 0.0, BUDGET),
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
            **REPORT_MODEL_KWARGS,
        )

        for strategy_name, policy in strategies.items():
            res = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=T,
                dt=DT,
                policy=policy,
            )

            all_results[(network_name, strategy_name)] = res

            final_A_u, final_A_m, final_B_u, final_B_m, LB = _final_B_from_result(res)
            final_q_B = _quality_B_from_result(res)
            max_shortage, avg_shortage = get_shortage_B(res)
            profit = get_profit(res)

            all_rows.append({
                "network": network_name,
                "alpha": alpha,
                "beta": beta,
                "strategy": strategy_name,
                "final_A_user_share": final_A_u,
                "final_A_merchant_share": final_A_m,
                "final_B_user_share": final_B_u,
                "final_B_merchant_share": final_B_m,
                "final_B_average_share": LB,
                "final_quality_B": final_q_B,
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
        x_A = np.asarray(res["x"], dtype=float)
        y_A = np.asarray(res["y"], dtype=float)
        x_B = 1.0 - x_A
        y_B = 1.0 - y_A
        avg = 0.5 * (x_B + y_B)
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

    # 图 2：服务质量动态。q_u 为 A 相对 B 优势，因此 B 质量存量取 -q_u。
    plt.figure(figsize=(9, 5))
    for strategy_name in strategies:
        res = all_results[("中等", strategy_name)]
        t = np.asarray(res["t"], dtype=float)
        q_B = -np.asarray(res.get("q_u", np.zeros_like(t)), dtype=float)
        plt.plot(t, q_B, label=f"{strategy_name}, q_B(T)={q_B[-1]:.3f}")

    plt.xlabel("时间")
    plt.ylabel(r"用户侧服务质量 $q_B^U(t)$")
    plt.title("阶段4：服务质量投资策略下的质量积累")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "stage4_quality_stock_dynamics.png"), dpi=300)
    plt.close()

    # 图 3：中等与强网络效应下策略综合比较
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
            [r["final_quality_B"] for r in rows],
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
    
        # 新增：报告图 12，强网络效应下纯质量投资的延迟反超路径
    stage4_report_pure_quality_delay_path(output_root)

    # 新增：报告图 13，纯质量投资强度的最终份额与贴现利润
    stage4_report_pure_quality_budget_scan(output_root)

    print("  阶段4完成。")

# ============================================================
# 5.6 补充实验 1：强网络效应下纯质量投资的延迟反超路径
# ============================================================

def stage4_report_pure_quality_delay_path(output_root: str) -> None:
    """
    对应报告图 12：
    强网络效应下纯质量投资的延迟反超路径。

    设定：
    - 平台 A 初始占优 x0=y0=0.8；
    - 平台 B 为挑战者；
    - 强网络效应 alpha=beta=1.5；
    - 只进行服务质量投资，不进行用户/商户补贴；
    - 比较 B=0.1, 0.2, 0.8 三种质量投资强度。
    """
    out = os.path.join(
        output_root,
        "stage4_quality_investment",
        "report_pure_quality_delay_path",
    )
    ensure_dir(out)

    investment_values = [0.1, 0.2, 0.8]

    params = make_params(
        alpha=1.5,
        beta=1.5,
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
        **REPORT_MODEL_KWARGS,
    )

    rows = []
    results = {}

    for B in investment_values:
        res = call_simulate(
            X0_A,
            Y0_A,
            params,
            T=T,
            dt=DT,
            policy=static_policy_B(0.0, 0.0, B),
        )

        results[B] = res

        t_values = np.asarray(res["t"], dtype=float)
        x_B = 1.0 - np.asarray(res["x"], dtype=float)
        y_B = 1.0 - np.asarray(res["y"], dtype=float)
        L_B = 0.5 * (x_B + y_B)

        if "q_u" in res:
            q_B = -np.asarray(res["q_u"], dtype=float)
        else:
            q_B = np.zeros_like(t_values)

        for t, uB, mB, lb, q in zip(t_values, x_B, y_B, L_B, q_B):
            rows.append({
                "investment_B": float(B),
                "t": float(t),
                "B_user_share": float(uB),
                "B_merchant_share": float(mB),
                "B_average_share": float(lb),
                "B_quality_stock": float(q),
            })

    save_rows_csv(
        rows,
        os.path.join(out, "pure_quality_delay_path_timeseries.csv"),
    )

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    for B in investment_values:
        res = results[B]
        t_values = np.asarray(res["t"], dtype=float)
        x_B = 1.0 - np.asarray(res["x"], dtype=float)
        y_B = 1.0 - np.asarray(res["y"], dtype=float)
        L_B = 0.5 * (x_B + y_B)

        if "q_u" in res:
            q_B = -np.asarray(res["q_u"], dtype=float)
        else:
            q_B = np.zeros_like(t_values)

        axes[0].plot(t_values, L_B, label=rf"$B={B}$")
        axes[1].plot(t_values, q_B, label=rf"$B={B}$")

    axes[0].axhline(0.5, linestyle="--", linewidth=1.0, label="反超阈值 0.5")
    axes[0].set_ylabel(r"平台 B 平均份额 $L_B$")
    axes[0].set_title("强网络效应下纯质量投资的延迟反超路径")
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].set_xlabel("时间")
    axes[1].set_ylabel(r"平台 B 服务质量存量 $q_B$")
    axes[1].set_title("纯质量投资下的平台 B 服务质量积累")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "pure_quality_delay_overtake_path.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# ============================================================
# 5.6 补充实验 2：纯质量投资强度的最终份额与贴现利润
# ============================================================

def stage4_report_pure_quality_budget_scan(output_root: str) -> None:
    """
    对应报告图 13：
    纯质量投资强度的最终份额与贴现利润。

    设定：
    - 平台 A 初始占优 x0=y0=0.8；
    - 平台 B 为挑战者；
    - 中等网络效应 alpha=beta=0.8；
    - 只进行服务质量投资；
    - 扫描质量投资强度 B in [0, 1.5]。
    """
    out = os.path.join(
        output_root,
        "stage4_quality_investment",
        "report_pure_quality_budget_scan",
    )
    ensure_dir(out)

    investment_values = np.round(np.linspace(0.0, 1.5, 61), 4)

    rows = []

    for B in investment_values:
        params = make_params(
            alpha=0.8,
            beta=0.8,
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
            **REPORT_MODEL_KWARGS,
        )

        res = call_simulate(
            X0_A,
            Y0_A,
            params,
            T=T,
            dt=DT,
            policy=static_policy_B(0.0, 0.0, float(B)),
        )

        xA, yA, xB, yB, LB = _final_B_from_result(res)
        profit = get_profit(res)
        q_B = _quality_B_from_result(res)

        rows.append({
            "investment_B": float(B),
            "final_A_user_share": xA,
            "final_A_merchant_share": yA,
            "final_B_user_share": xB,
            "final_B_merchant_share": yB,
            "final_B_average_share": LB,
            "final_B_quality": q_B,
            "profit": profit,
        })

    save_rows_csv(
        rows,
        os.path.join(out, "pure_quality_budget_scan.csv"),
    )

    B_values = [r["investment_B"] for r in rows]
    shares = [r["final_B_average_share"] for r in rows]
    profits = [r["profit"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(
        B_values,
        shares,
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=r"最终平均份额 $L_B$",
    )
    ax1.axhline(0.5, linestyle="--", linewidth=1.0, label="反超阈值 0.5")
    ax1.set_xlabel("纯质量投资强度 B")
    ax1.set_ylabel(r"平台 B 最终平均份额 $L_B$")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        B_values,
        profits,
        marker="s",
        markersize=3,
        linewidth=1.8,
        linestyle="--",
        label="贴现利润",
    )
    ax2.set_ylabel("贴现利润")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("纯质量投资强度对平台 B 最终份额与贴现利润的影响")
    fig.tight_layout()
    fig.savefig(
        os.path.join(out, "pure_quality_budget_share_profit.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)