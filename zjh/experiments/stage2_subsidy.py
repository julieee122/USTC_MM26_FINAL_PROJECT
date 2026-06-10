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
    static_policy,
    step_subsidy_policy,
)


DT_REPORT = 0.05
X0_B = 0.2
Y0_B = 0.2
DISCOUNT = 0.98

def run_stage2(output_root: str):
    setup_matplotlib()
    out = os.path.join(output_root, "stage2_subsidy")
    ensure_dir(out)

    # 5.4 主实验1
    stage2_report_subsidy_direction_and_budget(output_root)

    # 5.4 补充实验
    stage2_report_low_budget_single_side_advantage(output_root)

    # 5.4 主实验2
    stage2_report_subsidy_exit(output_root)

# ============================================================
# 报告版补充实验：从原 stage7 拆回 stage2
# ============================================================

def stage2_report_subsidy_direction_and_budget(output_root: str) -> None:
    """
    对应报告 5.4 主实验 1：
    补贴方向与预算强度。

    报告参数：
    - 平台 A 初始占优，平台 B 为挑战者，所以代码中 B 初始份额 0.2
    - 中等网络效应 alpha=beta=0.8
    - 强网络效应 alpha=beta=1.5
    - B in {0,0.1,0.2,0.4,1.2}
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
                    discount=DISCOUNT,
                )

                res = call_simulate(
                    X0_B,
                    Y0_B,
                    params,
                    T=200.0,
                    dt=DT_REPORT,
                    policy=static_policy(sub_u, sub_m, 0.0),
                )

                xf = get_x(res)
                yf = get_y(res)
                profit = get_profit(res)
                LB = combined_share(xf, yf)

                rows.append({
                    "network_level": level_name,
                    "alpha": alpha,
                    "beta": beta,
                    "budget_B": B,
                    "strategy": strategy_name,
                    "subsidy_user": sub_u,
                    "subsidy_merchant": sub_m,
                    "final_B_user_share": xf,
                    "final_B_merchant_share": yf,
                    "final_B_average_share": LB,
                    "profit": profit,
                    "state": platform_state(xf, yf),
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


def stage2_report_low_budget_single_side_advantage(output_root: str) -> None:
    """
    对应报告 5.4 补充实验：
    低预算区间内单侧补贴与均衡补贴的对比。

    扫描：
    - B in [0,0.2], step=0.01
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
                discount=DISCOUNT,
            )

            res = call_simulate(
                X0_B,
                Y0_B,
                params,
                T=200.0,
                dt=DT_REPORT,
                policy=static_policy(sub_u, sub_m, 0.0),
            )

            xf = get_x(res)
            yf = get_y(res)
            LB = combined_share(xf, yf)
            profit = get_profit(res)

            rows.append({
                "budget_B": float(B),
                "strategy": strategy_name,
                "subsidy_user": sub_u,
                "subsidy_merchant": sub_m,
                "final_B_user_share": xf,
                "final_B_merchant_share": yf,
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


def stage2_report_subsidy_exit(output_root: str) -> None:
    """
    对应报告 5.4 主实验 2：
    阶段性补贴退出。

    报告参数：
    - 中等网络效应 alpha=beta=0.8
    - B in [0,1.2]
    - Ts in [0,10]
    - T=200
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
                discount=DISCOUNT,
            )

            res = call_simulate(
                X0_B,
                Y0_B,
                params,
                T=200.0,
                dt=DT_REPORT,
                policy=step_subsidy_policy(B, Ts, split=0.5),
            )

            xf = get_x(res)
            yf = get_y(res)
            LB = combined_share(xf, yf)
            profit = get_profit(res)

            share_grid[i, j] = LB
            profit_grid[i, j] = profit

            rows.append({
                "budget_B": float(B),
                "subsidy_duration_Ts": float(Ts),
                "final_B_user_share": xf,
                "final_B_merchant_share": yf,
                "final_B_average_share": LB,
                "profit": profit,
                "state": platform_state(xf, yf),
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