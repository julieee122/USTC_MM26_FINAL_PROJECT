import os
import numpy as np
import matplotlib.pyplot as plt



from matplotlib.colors import ListedColormap, BoundaryNorm

from experiments.report_common import (
    ensure_dir,
    setup_matplotlib,
    save_rows_csv,
    make_params,
    call_simulate,
    get_x,
    get_y,
    concentration,
    combined_share,
    lock_index,
    market_state_by_c,
    platform_state,
)


DT_REPORT = 0.05

STAGE1_SOLVER = "rk45"


def call_stage1_simulate(*args, **kwargs):
    """
    阶段 1 基础机制实验使用 RK45。
    后续阶段策略实验仍然默认使用欧拉法。
    """
    kwargs.setdefault("method", STAGE1_SOLVER)
    return call_simulate(*args, **kwargs)

def run_stage1(output_root: str):
    setup_matplotlib()
    out = os.path.join(output_root, "stage1_basic")
    ensure_dir(out)

    # 5.2 主实验：网络效应与初始规模耦合
    stage1_report_main_network_initial_size(output_root)

    # 5.2 补充实验1：alpha-beta 二维扫描
    stage1_report_network_effect_critical_region(output_root)

    # 5.2 补充实验2：非同步初始优势
    # 如果报告中保留了这部分，就也要保留
    stage1_report_async_initial_advantage(output_root)

    # 5.3 主实验：服务质量优势突破初始锁定
    stage1_report_quality_advantage_break_lockin(output_root)

    # 5.3 补充实验：切换成本下服务质量突破阈值
    stage1_report_quality_threshold_with_switching_cost(output_root)

def stage1_report_main_network_initial_size(output_root: str) -> None:
    """
    对应报告 5.2 主实验：
    网络效应与初始规模耦合。

    报告参数：
    - 弱网络效应：(alpha,beta)=(0.3,0.3)
    - 中等网络效应：(alpha,beta)=(0.8,0.8)
    - 强网络效应：(alpha,beta)=(1.5,1.5)
    - 初始份额：u_A(0)=m_A(0)=0.5,0.55,0.65,0.75,0.85
    - 锁定指数：L_T=max(x_T,1-x_T)max(y_T,1-y_T)
    - 锁定判据：L_T>0.8
    """
    out = os.path.join(output_root, "stage1_basic", "report_main_network_initial_size")
    ensure_dir(out)

    network_levels = [
        ("弱网络效应", 0.3, 0.3),
        ("中等网络效应", 0.8, 0.8),
        ("强网络效应", 1.5, 1.5),
    ]

    initial_values = [0.5, 0.55, 0.65, 0.75, 0.85]

    rows = []
    summary_rows = []

    lock_grid = np.zeros((len(network_levels), len(initial_values)))
    final_x_grid = np.zeros_like(lock_grid)

    # 用于画典型轨迹：报告图 1a 主要展示 x0=y0=0.55 时不同网络效应下的演化
    plt.figure(figsize=(8, 5))

    for i, (level_name, alpha, beta) in enumerate(network_levels):
        min_lock_initial = None
        selected_summary = None

        for j, x0 in enumerate(initial_values):
            y0 = x0

            params = make_params(alpha=alpha, beta=beta)

            res = call_stage1_simulate(
                x0,
                y0,
                params,
                T=80.0,
                dt=DT_REPORT,
            )

            xf = get_x(res)
            yf = get_y(res)

            LT = lock_index(xf, yf)
            final_x_grid[i, j] = xf
            lock_grid[i, j] = LT

            state = "锁定" if LT > 0.8 else "未锁定"

            row = {
                "network_level": level_name,
                "alpha": alpha,
                "beta": beta,
                "initial_u_A": x0,
                "initial_m_A": y0,
                "initial_lead": x0 - 0.5,
                "final_u_A": xf,
                "final_m_A": yf,
                "lock_index_LT": LT,
                "state": state,
            }
            rows.append(row)

            if LT > 0.8 and min_lock_initial is None:
                min_lock_initial = x0
                selected_summary = row

        # 如果没有锁定，则取最大初始份额对应的结果，和报告表述一致
        if selected_summary is None:
            candidates = [
                r for r in rows
                if r["network_level"] == level_name
                and abs(r["initial_u_A"] - max(initial_values)) < 1e-12
            ]
            selected_summary = candidates[0]

            summary_rows.append({
                "network_level": level_name,
                "selection_rule": "未锁定，取最大初始份额",
                "initial_share": selected_summary["initial_u_A"],
                "initial_lead": selected_summary["initial_lead"],
                "final_u_A": selected_summary["final_u_A"],
                "final_m_A": selected_summary["final_m_A"],
                "lock_index_LT": selected_summary["lock_index_LT"],
                "state": selected_summary["state"],
            })
        else:
            summary_rows.append({
                "network_level": level_name,
                "selection_rule": "最小锁定初始份额",
                "initial_share": selected_summary["initial_u_A"],
                "initial_lead": selected_summary["initial_lead"],
                "final_u_A": selected_summary["final_u_A"],
                "final_m_A": selected_summary["final_m_A"],
                "lock_index_LT": selected_summary["lock_index_LT"],
                "state": selected_summary["state"],
            })

        # 典型轨迹：x0=y0=0.55
        params = make_params(alpha=alpha, beta=beta)
        res = call_stage1_simulate(
            0.55,
            0.55,
            params,
            T=80.0,
            dt=DT_REPORT,
        )

        plt.plot(
            res["t"],
            res["x"],
            label=rf"{level_name} $(\alpha=\beta={alpha})$",
        )

    save_rows_csv(rows, os.path.join(out, "main_network_initial_results.csv"))
    save_rows_csv(summary_rows, os.path.join(out, "main_network_initial_summary.csv"))

    # 图 1a：不同网络效应下 x0=y0=0.55 的用户份额演化
    plt.xlabel("时间")
    plt.ylabel(r"平台 A 用户份额 $u_A(t)$")
    plt.title(r"阶段1主实验：$u_A(0)=m_A(0)=0.55$ 时不同网络效应下的平台 A 用户份额演化")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "main_network_typical_trajectories.png"),
        dpi=300,
    )
    plt.close()

    # 图 1b：网络效应和初始规模对最终锁定指数的影响
    plt.figure(figsize=(8, 4.8))
    im = plt.imshow(
        lock_grid,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )

    plt.colorbar(im, label=r"锁定指数 $L_T$")
    plt.xticks(
        np.arange(len(initial_values)),
        [f"{v:.2f}" for v in initial_values],
    )
    plt.yticks(
        np.arange(len(network_levels)),
        [x[0] for x in network_levels],
    )
    plt.xlabel(r"初始份额 $u_A(0)=m_A(0)$")
    plt.ylabel("网络效应强度")
    plt.title("阶段1主实验：网络效应和初始规模对最终锁定指数的影响")

    for i in range(len(network_levels)):
        for j in range(len(initial_values)):
            plt.text(
                j,
                i,
                f"{lock_grid[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "main_network_lock_index_heatmap.png"),
        dpi=300,
    )
    plt.close()

def stage1_report_async_initial_advantage(output_root: str) -> None:
    """
    对应报告 5.2 补充实验 2：
    非同步初始优势。

    报告逻辑：
    - 强网络效应 alpha=beta=3.0；
    - 用户侧和商户侧初始份额不完全相等；
    - 反对角线 u_A(0)+m_A(0)=1 是理论分界线；
    - 当总初始优势略偏向 A 时走向 A 锁定；
      略偏向 B 时走向 B 锁定。
    """
    out = os.path.join(output_root, "stage1_basic", "report_async_initial_advantage")
    ensure_dir(out)

    params = make_params(
        alpha=3.0,
        beta=3.0,
        eta_u=0.0,
        eta_m=0.0,
    )

    representative_cases = [
        ("双边同步领先", 0.6000, 0.6000),
        ("用户领先商户落后，A总份额略占优", 0.7001, 0.3001),
        ("用户领先商户落后，B总份额略占优", 0.6999, 0.2999),
        ("商户领先用户落后，A总份额略占优", 0.3001, 0.7001),
        ("商户领先用户落后，B总份额略占优", 0.2999, 0.6999),
        ("双边同步落后", 0.4000, 0.4000),
        ("理论共存点", 0.5000, 0.5000),
    ]

    rows = []

    for case_name, x0, y0 in representative_cases:
        res = call_stage1_simulate(
            x0,
            y0,
            params,
            T=80.0,
            dt=DT_REPORT,
        )

        xf = get_x(res)
        yf = get_y(res)
        LA = combined_share(xf, yf)

        if xf > 0.8 and yf > 0.8:
            state = "平台 A 锁定"
        elif xf < 0.2 and yf < 0.2:
            state = "平台 B 锁定"
        elif abs(xf - 0.5) + abs(yf - 0.5) < 0.05:
            state = "双平台共存"
        else:
            state = "市场倾斜"

        rows.append({
            "case": case_name,
            "initial_u_A": x0,
            "initial_m_A": y0,
            "initial_sum": x0 + y0,
            "final_u_A": xf,
            "final_m_A": yf,
            "final_average_share_A": LA,
            "state": state,
        })

    save_rows_csv(rows, os.path.join(out, "async_initial_representative_cases.csv"))

    # 二维初始条件平面扫描
    # 报告中步长为 0.01，对应 81x81 个点；如果运行太慢，可改成 41。
    grid = np.round(np.linspace(0.1, 0.9, 81), 4)

    final_avg_grid = np.zeros((len(grid), len(grid)))
    scan_rows = []

    for i, y0 in enumerate(grid):
        for j, x0 in enumerate(grid):
            res = call_stage1_simulate(
                float(x0),
                float(y0),
                params,
                T=80.0,
                dt=DT_REPORT,
            )

            xf = get_x(res)
            yf = get_y(res)
            LA = combined_share(xf, yf)

            final_avg_grid[i, j] = LA

            scan_rows.append({
                "initial_u_A": float(x0),
                "initial_m_A": float(y0),
                "final_u_A": xf,
                "final_m_A": yf,
                "final_average_share_A": LA,
            })

    save_rows_csv(scan_rows, os.path.join(out, "async_initial_grid.csv"))

    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        final_avg_grid,
        origin="lower",
        extent=[grid.min(), grid.max(), grid.min(), grid.max()],
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="coolwarm",
    )

    plt.colorbar(im, label=r"最终平均份额 $L_A$")
    plt.xlabel(r"初始用户份额 $u_A(0)$")
    plt.ylabel(r"初始商户份额 $m_A(0)$")
    plt.title("阶段1补充：初始条件平面与锁定方向")

    # 理论分界线：u_A(0)+m_A(0)=1
    xs = np.linspace(0.1, 0.9, 200)
    ys = 1.0 - xs
    plt.plot(xs, ys, "k--", linewidth=1.2, label=r"$u_A(0)+m_A(0)=1$")

    # 标注理论共存点
    plt.scatter([0.5], [0.5], marker="o", s=35, label="理论共存点")

    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(out, "async_initial_final_share_heatmap.png"),
        dpi=300,
    )
    plt.close()

# ============================================================
# 报告版补充实验：从原 stage7 拆回 stage1
# ============================================================



def stage1_report_network_effect_critical_region(output_root: str) -> None:
    """
    对应报告 5.2 补充实验 1：
    双边网络效应强度扫描。

    参数：
    - u_A(0)=m_A(0)=0.55
    - alpha,beta in [0,5]，步长 0.1
    - C=|u_A-0.5|+|m_A-0.5|
    """
    out = os.path.join(output_root, "stage1_basic", "report_network_critical_region")
    ensure_dir(out)

    x0 = 0.55
    y0 = 0.55
    alphas = np.round(np.linspace(0.0, 5.0, 51), 2)
    betas = np.round(np.linspace(0.0, 5.0, 51), 2)

    c_grid = np.zeros((len(betas), len(alphas)))
    class_grid = np.zeros_like(c_grid, dtype=int)

    rows = []

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            params = make_params(alpha=float(alpha), beta=float(beta))
            res = call_stage1_simulate(x0, y0, params, T=80.0, dt=DT_REPORT)

            xf = get_x(res)
            yf = get_y(res)
            c = concentration(xf, yf)

            if c < 0.2:
                cls = 0
            elif c < 0.8:
                cls = 1
            else:
                cls = 2

            c_grid[i, j] = c
            class_grid[i, j] = cls

            rows.append({
                "alpha": float(alpha),
                "beta": float(beta),
                "final_u_A": xf,
                "final_m_A": yf,
                "combined_share_A": combined_share(xf, yf),
                "concentration_C": c,
                "market_state": market_state_by_c(c),
            })

    save_rows_csv(rows, os.path.join(out, "network_effect_grid.csv"))

    # 图 1：市场集中度热力图
    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        c_grid,
        origin="lower",
        extent=[alphas.min(), alphas.max(), betas.min(), betas.max()],
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
    )
    plt.colorbar(im, label="市场集中度 C")
    plt.xlabel(r"$\alpha$：商户规模对用户的影响")
    plt.ylabel(r"$\beta$：用户规模对商户的影响")
    plt.title("阶段1补充：双边网络效应强度与市场集中度")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "network_effect_concentration_heatmap.png"), dpi=300)
    plt.close()

    # 图 2：共存、倾斜、锁定分类图
    cmap = ListedColormap(["#7FC97F", "#FDB462", "#E15759"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        class_grid,
        origin="lower",
        extent=[alphas.min(), alphas.max(), betas.min(), betas.max()],
        aspect="auto",
        cmap=cmap,
        norm=norm,
    )
    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["双平台共存", "市场倾斜", "市场锁定"])
    plt.xlabel(r"$\alpha$：商户规模对用户的影响")
    plt.ylabel(r"$\beta$：用户规模对商户的影响")
    plt.title("阶段1补充：共存区、倾斜区与锁定区")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "network_effect_region_classification.png"), dpi=300)
    plt.close()

    # 图 3：对称路径 k=alpha=beta 下的临界跃迁
    k_values = np.round(np.linspace(0.0, 5.0, 101), 3)
    diag_rows = []
    diag_c = []
    k_tilt = None
    k_lock = None

    for k in k_values:
        params = make_params(alpha=float(k), beta=float(k))
        res = call_stage1_simulate(x0, y0, params, T=80.0, dt=DT_REPORT)

        xf = get_x(res)
        yf = get_y(res)
        c = concentration(xf, yf)
        diag_c.append(c)

        if c >= 0.2 and k_tilt is None:
            k_tilt = float(k)
        if c >= 0.8 and k_lock is None:
            k_lock = float(k)

        diag_rows.append({
            "k": float(k),
            "final_u_A": xf,
            "final_m_A": yf,
            "concentration_C": c,
            "market_state": market_state_by_c(c),
        })

    save_rows_csv(diag_rows, os.path.join(out, "diagonal_k_scan.csv"))

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, diag_c, linewidth=2)
    plt.axhline(0.2, linestyle="--", linewidth=1.2, label="共存-倾斜阈值 C=0.2")
    plt.axhline(0.8, linestyle="--", linewidth=1.2, label="倾斜-锁定阈值 C=0.8")

    if k_tilt is not None:
        plt.axvline(k_tilt, linestyle=":", linewidth=1.2)
        plt.text(k_tilt + 0.05, 0.25, rf"$k_1\approx {k_tilt:.2f}$")

    if k_lock is not None:
        plt.axvline(k_lock, linestyle=":", linewidth=1.2)
        plt.text(k_lock + 0.05, 0.85, rf"$k_2\approx {k_lock:.2f}$")

    plt.xlabel(r"对称网络效应强度 $k=\alpha=\beta$")
    plt.ylabel("市场集中度 C")
    plt.title("阶段1补充：对称网络效应路径下的临界跃迁")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "diagonal_k_critical_curve.png"), dpi=300)
    plt.close()

    summary = [{
        "coexist_count": int((class_grid == 0).sum()),
        "tilt_count": int((class_grid == 1).sum()),
        "lockin_count": int((class_grid == 2).sum()),
        "diagonal_tilt_threshold_k": "" if k_tilt is None else k_tilt,
        "diagonal_lockin_threshold_k": "" if k_lock is None else k_lock,
    }]
    save_rows_csv(summary, os.path.join(out, "network_effect_summary.csv"))


def stage1_report_quality_advantage_break_lockin(output_root: str) -> None:
    """
    对应报告 5.3：
    平台 B 凭服务质量优势打破平台 A 初始锁定。

    代码中 x,y 代表挑战者 B 的份额，因此初始为 0.2。
    """
    out = os.path.join(output_root, "stage1_basic", "report_quality_break_lockin")
    ensure_dir(out)

    x0_A = 0.8
    y0_A = 0.8

    network_levels = [
        ("弱网络效应", 0.3, 0.3),
        ("中等网络效应", 0.8, 0.8),
        ("强网络效应", 1.5, 1.5),
    ]

    dq_values = np.round(np.arange(0.0, 1.501, 0.01), 3)

    rows = []
    threshold_rows = []

    plt.figure(figsize=(8, 5))

    for label, alpha, beta in network_levels:
        avg_values = []
        break_threshold = None
        full_threshold = None
        break_share = None
        max_share = -1.0

        for dq in dq_values:
            params = make_params(
                alpha=alpha,
                beta=beta,
                dq_u_base=-float(dq),
                dq_m_base=-float(dq),
            )

            res = call_stage1_simulate(x0_A, y0_A, params, T=80.0, dt=DT_REPORT)

            xf_A = get_x(res)
            yf_A = get_y(res)

            xf_B = 1.0 - xf_A
            yf_B = 1.0 - yf_A

            avg = combined_share(xf_B, yf_B)
            avg_values.append(avg)
            max_share = max(max_share, avg)

            if break_threshold is None and xf_B >= 0.5 and yf_B >= 0.5:
                break_threshold = float(dq)
                break_share = avg

            if full_threshold is None and xf_B >= 0.99 and yf_B >= 0.99:
                full_threshold = float(dq)

            rows.append({
                "network_level": label,
                "alpha": alpha,
                "beta": beta,
                "delta_q": float(dq),
                "final_A_user_share": xf_A,
                "final_A_merchant_share": yf_A,
                "final_B_user_share": xf_B,
                "final_B_merchant_share": yf_B,
                "final_B_average_share": avg,
                "state": platform_state(xf_B, yf_B),
            })

        threshold_rows.append({
            "network_level": label,
            "alpha": alpha,
            "beta": beta,
            "break_lockin_delta_q": break_threshold,
            "break_lockin_share": break_share,
            "full_overtake_delta_q": full_threshold,
            "max_scanned_share": max_share,
        })

        plt.plot(dq_values, avg_values, label=label)

    save_rows_csv(rows, os.path.join(out, "quality_advantage_scan.csv"))
    save_rows_csv(threshold_rows, os.path.join(out, "quality_advantage_thresholds.csv"))

    plt.axhline(0.5, linestyle="--", linewidth=1.0, label="打破锁定阈值 0.5")
    plt.axhline(0.99, linestyle=":", linewidth=1.0, label="完全反超阈值 0.99")
    plt.xlabel(r"平台 B 服务质量优势 $\Delta q$")
    plt.ylabel(r"平台 B 最终平均份额 $L_B$")
    plt.title("阶段1补充：服务质量优势对平台 B 最终份额的影响")
    plt.ylim(0, 1.03)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out, "quality_advantage_final_share.png"), dpi=300)
    plt.close()


def stage1_report_quality_threshold_with_switching_cost(output_root: str) -> None:
    """
    对应报告 5.3 补充实验：
    加入切换成本/惯性后，服务质量突破阈值如何变化。

    报告参数：
    - alpha=beta in {0.3,0.8,1.5}
    - s in {0,0.5,1.0,1.5}
    - Delta q in [0,1.5], step=0.01
    """
    out = os.path.join(output_root, "stage1_basic", "report_quality_switching_threshold")
    ensure_dir(out)

    x0_A = 0.8
    y0_A = 0.8

    network_levels = [
        ("弱网络效应", 0.3),
        ("中等网络效应", 0.8),
        ("强网络效应", 1.5),
    ]
    switching_costs = [0.0, 0.5, 1.0, 1.5]
    dq_values = np.round(np.arange(0.0, 1.501, 0.01), 3)

    threshold_grid = np.full((len(switching_costs), len(network_levels)), np.nan)
    rows = []

    for i, s in enumerate(switching_costs):
        for j, (label, k) in enumerate(network_levels):
            q_star = np.nan
            u_star = np.nan
            m_star = np.nan

            for dq in dq_values:
                params = make_params(
                    alpha=k,
                    beta=k,
                    eta_u=s,
                    eta_m=s,
                    dq_u_base=-float(dq),
                    dq_m_base=-float(dq),
                )

                res = call_stage1_simulate(x0_A, y0_A, params, T=80.0, dt=DT_REPORT)

                xf_A = get_x(res)
                yf_A = get_y(res)

                xf_B = 1.0 - xf_A
                yf_B = 1.0 - yf_A

                if xf_B >= 0.5 and yf_B >= 0.5:
                    q_star = float(dq)
                    u_star = xf_B
                    m_star = yf_B
                    break

            threshold_grid[i, j] = q_star

            rows.append({
                "network_level": label,
                "k_alpha_beta": k,
                "switching_cost_s": s,
                "quality_threshold": q_star,
                "final_B_user_share_at_threshold": u_star,
                "final_B_merchant_share_at_threshold": m_star,
            })

    save_rows_csv(rows, os.path.join(out, "quality_switching_thresholds.csv"))

    plt.figure(figsize=(7, 5))
    im = plt.imshow(
        threshold_grid,
        origin="lower",
        aspect="auto",
        extent=[
            0,
            len(network_levels) - 1,
            min(switching_costs),
            max(switching_costs),
        ],
        cmap="magma_r",
        vmin=0,
        vmax=1.5,
    )
    plt.colorbar(im, label=r"最小质量优势 $\Delta q^*$")
    plt.xticks(range(len(network_levels)), [x[0] for x in network_levels])
    plt.yticks(switching_costs)
    plt.xlabel("网络效应强度")
    plt.ylabel(r"切换成本/惯性 $s$")
    plt.title("阶段1补充：切换成本与网络效应下的服务质量突破阈值")

    for i, s in enumerate(switching_costs):
        for j, _ in enumerate(network_levels):
            value = threshold_grid[i, j]
            text = "无" if np.isnan(value) else f"{value:.2f}"
            plt.text(j, s, text, ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out, "quality_switching_threshold_heatmap.png"), dpi=300)
    plt.close()