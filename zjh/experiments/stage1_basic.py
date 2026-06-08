import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import Params
from src.model import (
    simulate_path,
    simulate_grid_initial,
    simulate_alpha_beta_scan,
)
from src.utils import ensure_dir, setup_matplotlib


def run_stage1(output_root: str):
    setup_matplotlib()

    out = os.path.join(output_root, "stage1_basic")
    ensure_dir(out)

    p = Params()

    # ============================================================
    # 1.1 alpha = beta 情形：网络效应与初始规模耦合
    # ============================================================

    strengths = [0.2, 0.6, 1.0, 1.4]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    last_im = None

    for idx, g in enumerate(strengths):
        grid, X_final, _ = simulate_grid_initial(
            p,
            alpha=g,
            beta=g,
            grid_n=51,
            T=70.0,
            dt=0.05
        )

        ax = axes[idx]
        last_im = ax.imshow(
            X_final,
            extent=[grid.min(), grid.max(), grid.min(), grid.max()],
            origin="lower",
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap="viridis"
        )

        ax.set_title(rf"$\alpha=\beta={g}$")
        ax.set_xlabel(r"平台 A 初始用户份额 $x(0)$")
        ax.set_ylabel(r"平台 A 初始商户份额 $y(0)$")
        ax.axvline(0.5, linestyle="--", linewidth=1)
        ax.axhline(0.5, linestyle="--", linewidth=1)

    fig.suptitle("阶段1-实验1：网络效应与初始规模耦合", fontsize=16)
    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.85)
    cbar.set_label(r"最终平台 A 用户份额 $x^*$")

    fig.savefig(
        os.path.join(out, "stage1_1_initial_network_equal_alpha_beta.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 1.2 alpha != beta 情形：非对称网络效应扫描
    # ============================================================

    alpha_values = np.linspace(0.1, 1.8, 61)
    beta_values = np.linspace(0.1, 1.8, 61)

    cases = [
        (0.45, 0.55, "user_low_merchant_high"),
        (0.55, 0.45, "user_high_merchant_low"),
        (0.45, 0.45, "both_low"),
    ]

    for x0, y0, name in cases:
        _, _, X_final, _ = simulate_alpha_beta_scan(
            p,
            x0,
            y0,
            alpha_values,
            beta_values,
            T=70.0,
            dt=0.05
        )

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            X_final,
            extent=[
                alpha_values.min(), alpha_values.max(),
                beta_values.min(), beta_values.max()
            ],
            origin="lower",
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap="viridis"
        )

        plt.colorbar(im, label=r"最终平台 A 用户份额 $x^*$")
        plt.title(rf"阶段1：非对称网络效应扫描，$x_0={x0}, y_0={y0}$")
        plt.xlabel(r"$\alpha$：商户规模对用户的影响")
        plt.ylabel(r"$\beta$：用户规模对商户的影响")

        plt.savefig(
            os.path.join(out, f"stage1_1_asymmetric_alpha_beta_{name}.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # ============================================================
    # 1.3 不同网络效应结构下的典型轨迹
    # ============================================================

    typical_initials = [
        (0.45, 0.45),
        (0.50, 0.50),
        (0.55, 0.55),
        (0.60, 0.50),
        (0.50, 0.60),
    ]

    network_cases = [
        (0.4, 0.4, "对称弱网络效应"),
        (1.2, 1.2, "对称强网络效应"),
        (1.2, 0.4, "用户更依赖商户"),
        (0.4, 1.2, "商户更依赖用户"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (alpha, beta, title) in zip(axes, network_cases):
        p_case = Params(alpha=alpha, beta=beta)

        for x0, y0 in typical_initials:
            res = simulate_path(x0, y0, p_case, T=60.0, dt=0.03)
            ax.plot(res["t"], res["x"], label=rf"$x_0={x0},y_0={y0}$")

        ax.set_title(title + rf"：$\alpha={alpha},\beta={beta}$")
        ax.set_xlabel("时间")
        ax.set_ylabel(r"平台 A 用户份额 $x(t)$")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("阶段1-实验1：不同网络效应结构下的典型轨迹", fontsize=16)
    fig.savefig(
        os.path.join(out, "stage1_1_typical_trajectories_asymmetric.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # ============================================================
    # 1.4 服务质量能否打破锁定
    # ============================================================

    x0 = 0.35
    y0 = 0.35

    q_values = np.linspace(0.0, 1.2, 121)

    final_x = []
    final_y = []

    for q in q_values:
        p_case = Params(
            alpha=1.2,
            beta=1.0,
            dq_u_base=q,
            dq_m_base=q,
        )

        res = simulate_path(x0, y0, p_case, T=80.0, dt=0.03)
        final_x.append(res["x"][-1])
        final_y.append(res["y"][-1])

    final_x = np.array(final_x)
    final_y = np.array(final_y)

    threshold_idx = np.where((final_x > 0.8) & (final_y > 0.8))[0]
    q_threshold = q_values[threshold_idx[0]] if len(threshold_idx) > 0 else None

    plt.figure(figsize=(9, 6))
    plt.plot(q_values, final_x, marker="o", markersize=2, label=r"最终用户份额 $x^*$")
    plt.plot(q_values, final_y, marker="s", markersize=2, label=r"最终商户份额 $y^*$")
    plt.axhline(0.5, linestyle=":", linewidth=1, label="反超判据 0.5")
    plt.axhline(0.8, linestyle="--", linewidth=1, label="锁定判据 0.8")

    if q_threshold is not None:
        plt.axvline(q_threshold, linestyle="--", linewidth=1)
        plt.text(q_threshold + 0.02, 0.12, rf"$q_c\approx {q_threshold:.2f}$")

    plt.title("阶段1-实验2：服务质量优势能否打破锁定")
    plt.xlabel(r"平台 A 服务质量优势 $q=\Delta q_U=\Delta q_M$")
    plt.ylabel("最终市场份额")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "stage1_2_quality_break_lockin_threshold.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    q_examples = [0.0, 0.2, 0.3, 0.45, 0.7]

    plt.figure(figsize=(10, 6))
    for q in q_examples:
        p_case = Params(alpha=1.2, beta=1.0, dq_u_base=q, dq_m_base=q)
        res = simulate_path(x0, y0, p_case, T=80.0, dt=0.03)
        plt.plot(
            res["t"],
            res["x"],
            label=rf"$q={q},x^*={res['x'][-1]:.2f}$"
        )

    plt.title("阶段1-实验2：不同服务质量优势下的平台 A 用户份额演化")
    plt.xlabel("时间")
    plt.ylabel(r"平台 A 用户份额 $x(t)$")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(out, "stage1_2_quality_trajectories.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()