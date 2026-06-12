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
    combined_share,
    platform_state,
    static_policy_B,
    step_subsidy_policy_B,
)


DT_REPORT = 0.05

# 统一报告口径：x,y 始终表示平台 A 份额；平台 A 初始占优，平台 B 为挑战者。
X0_A = 0.8
Y0_A = 0.8


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
    """由模型输出的 A 份额反推出平台 B 最终份额。"""
    xA = get_x(res)
    yA = get_y(res)
    xB = 1.0 - xA
    yB = 1.0 - yA
    return xA, yA, xB, yB, combined_share(xB, yB)


def run_stage2(output_root: str):
    setup_matplotlib()
    out = os.path.join(output_root, "stage2_subsidy")
    ensure_dir(out)

    stage2_report_subsidy_direction_and_budget(output_root)
    stage2_report_subsidy_timeseries_budget_cases(output_root)
    stage2_report_low_budget_single_side_advantage(output_root)
    stage2_report_subsidy_exit(output_root)


# ============================================================
# 5.4 主实验 1：补贴方向与预算强度
# ============================================================

def stage2_report_subsidy_direction_and_budget(output_root: str) -> None:
    """
    对应报告 5.4 主实验 1：补贴方向与预算强度。

    统一 A/B 口径：
    - x,y 表示平台 A 份额；
    - 平台 A 初始占优 x0=y0=0.8；
    - 平台 B 采取补贴策略，补贴通过 static_policy_B 进入模型。
    """
    out = os.path.join(output_root, "stage2_subsidy", "report_subsidy_direction_budget")
    ensure_dir(out)

    network_levels = [
        ("中等网络效应", 0.8, 0.8),
        ("强网络效应", 1.5, 1.5),
    ]

    budgets = [0.0, 0.1, 0.2, 0.4, 1.2]

    strategies = {
        "只补贴用户": (1.0, 0.0),
        "只补贴商户": (0.0, 1.0),
        "均衡补贴": (0.5, 0.5),
        "偏用户补贴": (0.75, 0.25),
        "偏商户补贴": (0.25, 0.75),
    }

    rows = []

    for level_name, alpha, beta in network_levels:
        for B in budgets:
            for strategy_name, (rho_u, rho_m) in strategies.items():
                sub_u = rho_u * B
                sub_m = rho_m * B

                params = make_params(
                    alpha=alpha,
                    beta=beta,
                    
                    **REPORT_MODEL_KWARGS,
                )

                res = call_simulate(
                    X0_A,
                    Y0_A,
                    params,
                    T=200.0,
                    dt=DT_REPORT,
                    policy=static_policy_B(sub_u, sub_m, 0.0),
                )

                xA, yA, xB, yB, LB = _final_B_from_result(res)
                profit = get_profit(res)

                rows.append({
                    "network_level": level_name,
                    "alpha": alpha,
                    "beta": beta,
                    "budget_B": B,
                    "strategy": strategy_name,
                    "subsidy_user": sub_u,
                    "subsidy_merchant": sub_m,
                    "final_A_user_share": xA,
                    "final_A_merchant_share": yA,
                    "final_B_user_share": xB,
                    "final_B_merchant_share": yB,
                    "final_B_average_share": LB,
                    "profit": profit,
                    "state": platform_state(xB, yB),
                })

    save_rows_csv(rows, os.path.join(out, "subsidy_direction_budget.csv"))

    # 图：中等网络效应下的最终平均份额
    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        xs = []
        ys = []
        for B in budgets:
            match = [
                r for r in rows
                if r["network_level"] == "中等网络效应"
                and r["budget_B"] == B
                and r["strategy"] == strategy_name
            ]
            if match:
                xs.append(B)
                ys.append(match[0]["final_B_average_share"])
        plt.plot(xs, ys, marker="o", label=strategy_name)

    plt.xlabel("预算强度 B")
    plt.ylabel(r"平台 B 最终平均份额 $L_B$")
    plt.title("阶段2：补贴方向和预算强度对平台 B 最终平均份额的影响")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subsidy_direction_budget_share.png"), dpi=300)
    plt.close()

    # 图：中等网络效应下的贴现利润
    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        xs = []
        ys = []
        for B in budgets:
            match = [
                r for r in rows
                if r["network_level"] == "中等网络效应"
                and r["budget_B"] == B
                and r["strategy"] == strategy_name
            ]
            if match:
                xs.append(B)
                ys.append(match[0]["profit"])
        plt.plot(xs, ys, marker="o", label=strategy_name)

    plt.xlabel("预算强度 B")
    plt.ylabel("贴现利润")
    plt.title("阶段2：补贴方向和预算强度对贴现利润的影响")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subsidy_direction_budget_profit.png"), dpi=300)
    plt.close()

# ============================================================
# 5.4 主实验 1 补充图：不同预算下的动态轨迹
# ============================================================

def stage2_report_subsidy_timeseries_budget_cases(output_root: str) -> None:
    
    out = os.path.join(
        output_root,
        "stage2_subsidy",
        "report_subsidy_direction_budget",
    )
    ensure_dir(out)

    budget_cases = [0.1, 1.2]

    strategies = {
        "只补贴用户": (1.0, 0.0),
        "只补贴商户": (0.0, 1.0),
        "均衡补贴": (0.5, 0.5),
        "偏用户补贴": (0.75, 0.25),
        "偏商户补贴": (0.25, 0.75),
    }

    rows = []
    results = {}

    for B in budget_cases:
        for strategy_name, (rho_u, rho_m) in strategies.items():
            sub_u = rho_u * B
            sub_m = rho_m * B

            params = make_params(
                alpha=0.8,
                beta=0.8,
                
                **REPORT_MODEL_KWARGS,
            )

            res = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=200.0,
                dt=DT_REPORT,
                policy=static_policy_B(sub_u, sub_m, 0.0),
            )

            results[(B, strategy_name)] = res

            t_values = np.asarray(res["t"], dtype=float)
            x_A = np.asarray(res["x"], dtype=float)
            y_A = np.asarray(res["y"], dtype=float)

            x_B = 1.0 - x_A
            y_B = 1.0 - y_A
            L_B = 0.5 * (x_B + y_B)

            for t, uB, mB, lb in zip(t_values, x_B, y_B, L_B):
                rows.append({
                    "budget_B": float(B),
                    "strategy": strategy_name,
                    "t": float(t),
                    "B_user_share": float(uB),
                    "B_merchant_share": float(mB),
                    "B_average_share": float(lb),
                })

    save_rows_csv(
        rows,
        os.path.join(out, "subsidy_direction_budget_timeseries.csv"),
    )

    fig, axes = plt.subplots(
        len(budget_cases),
        2,
        figsize=(14, 7),
        sharex=True,
        sharey=True,
    )

    for row_idx, B in enumerate(budget_cases):
        ax_user = axes[row_idx, 0]
        ax_merchant = axes[row_idx, 1]

        for strategy_name in strategies:
            res = results[(B, strategy_name)]
            t_values = np.asarray(res["t"], dtype=float)
            x_B = 1.0 - np.asarray(res["x"], dtype=float)
            y_B = 1.0 - np.asarray(res["y"], dtype=float)

            ax_user.plot(t_values, x_B, label=strategy_name)
            ax_merchant.plot(t_values, y_B, label=strategy_name)

        ax_user.axhline(0.5, linestyle="--", linewidth=1.0)
        ax_merchant.axhline(0.5, linestyle="--", linewidth=1.0)

        ax_user.set_title(rf"预算 $B={B}$：平台 B 用户份额")
        ax_merchant.set_title(rf"预算 $B={B}$：平台 B 商户份额")

        ax_user.set_ylabel("市场份额")
        ax_user.grid(alpha=0.3)
        ax_merchant.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("时间")
    axes[-1, 1].set_xlabel("时间")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "阶段2图7：不同补贴方向和预算强度下平台 B 用户/商户份额演化",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    fig.savefig(
        os.path.join(out, "subsidy_direction_budget_timeseries.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
# ============================================================
# 5.4 补充实验：低预算下的单侧补贴优势
# ============================================================

def stage2_report_low_budget_single_side_advantage(output_root: str) -> None:
    """
    对应报告 5.4 补充实验：低预算区间内单侧补贴与均衡补贴对比。
    """
    out = os.path.join(output_root, "stage2_subsidy", "report_low_budget_single_side")
    ensure_dir(out)

    budgets = np.round(np.arange(0.0, 0.201, 0.01), 3)

    strategies = {
        "只补贴用户": (1.0, 0.0),
        "偏用户补贴": (0.75, 0.25),
        "均衡补贴": (0.5, 0.5),
    }

    rows = []

    for B in budgets:
        for strategy_name, (rho_u, rho_m) in strategies.items():
            sub_u = rho_u * B
            sub_m = rho_m * B

            params = make_params(
                alpha=0.8,
                beta=0.8,
                
                **REPORT_MODEL_KWARGS,
            )

            res = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=200.0,
                dt=DT_REPORT,
                policy=static_policy_B(sub_u, sub_m, 0.0),
            )

            xA, yA, xB, yB, LB = _final_B_from_result(res)
            profit = get_profit(res)

            rows.append({
                "budget_B": float(B),
                "strategy": strategy_name,
                "subsidy_user": sub_u,
                "subsidy_merchant": sub_m,
                "final_A_user_share": xA,
                "final_A_merchant_share": yA,
                "final_B_user_share": xB,
                "final_B_merchant_share": yB,
                "final_B_average_share": LB,
                "profit": profit,
            })

    save_rows_csv(rows, os.path.join(out, "low_budget_single_side.csv"))

    # 画绝对值曲线
    plt.figure(figsize=(8, 5))
    for strategy_name in strategies:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        plt.plot(
            [r["budget_B"] for r in sub],
            [r["final_B_average_share"] for r in sub],
            label=strategy_name,
        )

    plt.xlabel("预算强度 B")
    plt.ylabel(r"平台 B 最终平均份额 $L_B$")
    plt.title("阶段2补充：低预算区间内最终平均份额对比")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "low_budget_share_compare.png"), dpi=300)
    plt.close()

    # 画相对均衡补贴的份额差
    balanced = {
        r["budget_B"]: r["final_B_average_share"]
        for r in rows
        if r["strategy"] == "均衡补贴"
    }

    plt.figure(figsize=(8, 5))
    for strategy_name in ["只补贴用户", "偏用户补贴"]:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        xs = [r["budget_B"] for r in sub]
        ys = [r["final_B_average_share"] - balanced[r["budget_B"]] for r in sub]
        plt.plot(xs, ys, label=f"{strategy_name} - 均衡补贴")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("预算强度 B")
    plt.ylabel(r"相对均衡补贴的份额差")
    plt.title("阶段2补充：低预算下单侧补贴相对均衡补贴的份额优势")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "low_budget_share_advantage.png"), dpi=300)
    plt.close()

    # 利润差
    balanced_profit = {
        r["budget_B"]: r["profit"]
        for r in rows
        if r["strategy"] == "均衡补贴"
    }

    plt.figure(figsize=(8, 5))
    for strategy_name in ["只补贴用户", "偏用户补贴"]:
        sub = [r for r in rows if r["strategy"] == strategy_name]
        xs = [r["budget_B"] for r in sub]
        ys = [r["profit"] - balanced_profit[r["budget_B"]] for r in sub]
        plt.plot(xs, ys, label=f"{strategy_name} - 均衡补贴")

    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("预算强度 B")
    plt.ylabel("相对均衡补贴的利润差")
    plt.title("阶段2补充：低预算下单侧补贴相对均衡补贴的利润优势")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "low_budget_profit_advantage.png"), dpi=300)
    plt.close()


# ============================================================
# 5.4 主实验 2：阶段性补贴退出
# ============================================================

def stage2_report_subsidy_exit(output_root: str) -> None:
    """
    对应报告 5.4 主实验 2：阶段性补贴退出。
    """
    out = os.path.join(output_root, "stage2_subsidy", "report_subsidy_exit")
    ensure_dir(out)

    B_values = np.round(np.linspace(0.0, 1.2, 49), 4)
    Ts_values = np.round(np.linspace(0.0, 10.0, 41), 4)

    share_grid = np.zeros((len(Ts_values), len(B_values)))
    profit_grid = np.zeros_like(share_grid)

    rows = []

    for i, Ts in enumerate(Ts_values):
        for j, B in enumerate(B_values):
            params = make_params(
                alpha=0.8,
                beta=0.8,
                
                **REPORT_MODEL_KWARGS,
            )

            res = call_simulate(
                X0_A,
                Y0_A,
                params,
                T=200.0,
                dt=DT_REPORT,
                policy=step_subsidy_policy_B(B, Ts, split=0.5),
            )

            xA, yA, xB, yB, LB = _final_B_from_result(res)
            profit = get_profit(res)

            share_grid[i, j] = LB
            profit_grid[i, j] = profit

            rows.append({
                "budget_B": float(B),
                "subsidy_duration_Ts": float(Ts),
                "final_A_user_share": xA,
                "final_A_merchant_share": yA,
                "final_B_user_share": xB,
                "final_B_merchant_share": yB,
                "final_B_average_share": LB,
                "profit": profit,
                "state": platform_state(xB, yB),
            })

    save_rows_csv(rows, os.path.join(out, "subsidy_exit_grid.csv"))

    # 份额热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        share_grid,
        origin="lower",
        aspect="auto",
        extent=[B_values.min(), B_values.max(), Ts_values.min(), Ts_values.max()],
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(im, label=r"平台 B 最终平均份额 $L_B$")
    plt.xlabel("补贴强度 B")
    plt.ylabel(r"补贴持续时间 $T_s$")
    plt.title("阶段2：阶段性补贴退出后的最终平均份额")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subsidy_exit_share_heatmap.png"), dpi=300)
    plt.close()

    # 利润热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        profit_grid,
        origin="lower",
        aspect="auto",
        extent=[B_values.min(), B_values.max(), Ts_values.min(), Ts_values.max()],
        cmap="viridis",
    )
    plt.colorbar(im, label="贴现利润")
    plt.xlabel("补贴强度 B")
    plt.ylabel(r"补贴持续时间 $T_s$")
    plt.title("阶段2：阶段性补贴退出后的贴现利润")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "subsidy_exit_profit_heatmap.png"), dpi=300)
    plt.close()

    # 找利润最优点
    best_idx = np.unravel_index(np.nanargmax(profit_grid), profit_grid.shape)
    best_Ts = Ts_values[best_idx[0]]
    best_B = B_values[best_idx[1]]
    best_profit = profit_grid[best_idx]
    best_share = share_grid[best_idx]

    summary = [{
        "best_budget_B": float(best_B),
        "best_duration_Ts": float(best_Ts),
        "best_profit": float(best_profit),
        "final_B_average_share_at_best": float(best_share),
    }]

    save_rows_csv(summary, os.path.join(out, "subsidy_exit_summary.csv"))
