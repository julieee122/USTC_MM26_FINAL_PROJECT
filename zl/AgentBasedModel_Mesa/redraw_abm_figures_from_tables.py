from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from run_abm_experiments import COLORS, plot_heatmap, set_plot_style


BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "results" / "figures"
TABLE_DIR = BASE_DIR / "results" / "tables"


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def save_line_plot(fig_path: Path, title: str, xlabel: str, ylabel: str) -> None:
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.72)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def draw_experiment_a() -> None:
    summary = read_table("revised_A_critical_heterogeneity_summary.csv")
    final_df = read_table("revised_A_critical_heterogeneity_final.csv")
    segment = read_table("revised_A_user_segment_summary.csv")

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    x = summary["heterogeneity_sigma"]
    ax.plot(x, summary["success_probability"], marker="o", lw=2.6, color=COLORS["blue"], label="A 占优概率")
    ax.plot(x, summary["lock_B_probability"], marker="o", lw=2.6, color=COLORS["orange"], label="B 锁定概率")
    ax.plot(x, summary["coexist_probability"], marker="o", lw=2.6, color=COLORS["teal"], label="共存概率")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best")
    save_line_plot(
        FIG_DIR / "revised_A_heterogeneity_probabilities.png",
        "实验A：临界区异质性对市场结构的影响",
        "个体异质性强度",
        "概率",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = summary["heterogeneity_sigma"].astype(float)
    mean_l = summary["final_L_mean"]
    std_l = summary["final_L_std"].fillna(0.0)
    lower = (mean_l - std_l).clip(0, 1)
    upper = (mean_l + std_l).clip(0, 1)
    ax.fill_between(x, lower, upper, color=COLORS["blue"], alpha=0.16, label="均值±标准差")
    ax.plot(x, mean_l, marker="o", lw=2.4, color=COLORS["blue"], label="最终 LA 均值")
    ax.axhline(0.6, color=COLORS["red"], lw=1.4, ls=":", label="A 占优阈值")
    ax.set_xlabel("个体异质性强度")
    ax.set_ylabel("最终平台 A 综合份额")
    ax.set_xticks(x, [f"{v:.2f}" for v in x])
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis="y", alpha=0.6)

    ax2 = ax.twinx()
    ax2.bar(x, summary["lock_B_probability"], width=0.12, color=COLORS["orange"], alpha=0.42, label="B 锁定概率")
    ax2.set_ylabel("B 锁定概率")
    ax2.set_ylim(0, 0.35)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower left")
    ax.set_title("实验A：异质性强度、最终份额与失败风险", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "revised_A_heterogeneity_boxplot.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.7, 5.2))
    high = segment[segment["heterogeneity_sigma"] == segment["heterogeneity_sigma"].max()]
    high = high.copy()
    high["estimated_A_users"] = high["share_on_A_mean"] * high["count_mean"]
    high["share_among_A_users"] = high["estimated_A_users"] / high["estimated_A_users"].sum()
    high = high.sort_values("share_among_A_users", ascending=False)
    label_map = {
        "price_sensitive": "价格敏感型",
        "quality_sensitive": "质量敏感型",
        "subsidy_sensitive": "补贴敏感型",
        "inertial": "惯性型",
    }
    labels = high["segment"].map(label_map)
    values = high["share_among_A_users"]
    colors = [COLORS["blue"], COLORS["teal"], COLORS["gold"], COLORS["orange"]]
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.76,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    for text in autotexts:
        text.set_color("#1F2933")
        text.set_fontsize(10)
    ax.text(0, 0, "平台 A\n最终用户", ha="center", va="center", fontsize=12, color="#1F2933")
    ax.set_title("实验A：强异质性下平台 A 最终用户类型构成", fontsize=13, pad=12)
    ax.legend(wedges, [f"{label}：{value:.3f}" for label, value in zip(labels, values)], loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "revised_A_user_segment_distribution.png")
    plt.close(fig)


def draw_experiment_b() -> None:
    summary = read_table("revised_B_equal_budget_subsidy_summary.csv")
    fine = read_table("paper_B_budget_fine_scan_summary.csv")
    palette = {
        "无补贴": COLORS["gray"],
        "统一补贴": COLORS["gold"],
        "随机补贴": COLORS["orange"],
        "摇摆用户": COLORS["blue"],
        "关键商户": COLORS["teal"],
        "双边精准": COLORS["purple"],
    }

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for label, group in summary.groupby("policy_label"):
        group = group.sort_values("budget")
        ax.plot(
            group["budget"],
            group["success_probability"],
            marker="o",
            lw=2.4,
            color=palette.get(label, COLORS["gray"]),
            label=label,
        )
    ax.axhline(0.5, color=COLORS["red"], lw=1.4, ls=":", label="50% 成功概率")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right")
    save_line_plot(
        FIG_DIR / "revised_B_budget_success_curves.png",
        "实验B：等预算下不同补贴策略的成功概率",
        "单期补贴预算",
        "P(L_A > 0.6)",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for label, group in summary.groupby("policy_label"):
        if label == "无补贴":
            continue
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["roi_mean"], marker="o", lw=2.4, color=palette.get(label, COLORS["gray"]), label=label)
    ax.axhline(0, color="#2F3437", lw=1.0)
    ax.legend(loc="best")
    save_line_plot(
        FIG_DIR / "revised_B_budget_roi_curves.png",
        "实验B：等预算下不同补贴策略的 ROI",
        "单期补贴预算",
        "平均 ROI",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for label, group in fine.groupby("policy_label"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["success_probability"], marker="o", lw=2.4, color=palette.get(label, COLORS["gray"]), label=label)
    ax.axhline(0.5, color=COLORS["red"], lw=1.4, ls=":", label="50% 成功概率")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="lower right")
    save_line_plot(
        FIG_DIR / "paper_B_budget_fine_success_curves.png",
        "补贴临界预算细粒度扫描",
        "单期补贴预算",
        "P(L_A > 0.6)",
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for label, group in fine.groupby("policy_label"):
        group = group.sort_values("budget")
        ax.plot(group["budget"], group["roi_mean"], marker="o", lw=2.4, color=palette.get(label, COLORS["gray"]), label=label)
    ax.axhline(0, color="#2F3437", lw=1.0)
    ax.legend(loc="best")
    save_line_plot(
        FIG_DIR / "paper_B_budget_fine_roi_curves.png",
        "临界预算区间 ROI 对比",
        "单期补贴预算",
        "平均 ROI",
    )


def draw_experiment_c() -> None:
    summary = read_table("revised_C_cold_start_heatmap_summary.csv")
    fine = read_table("paper_C_cold_start_fine_heatmap_summary.csv")
    seed_quality = read_table("revised_C_seed_quality_summary.csv")

    pivot = summary.pivot(index="u0", columns="m0", values="success_probability")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "revised_C_cold_start_success_heatmap.png",
        title="实验C：冷启动成功概率热力图",
        xlabel="初始商户份额 m0",
        ylabel="初始用户份额 u0",
        cbar_label="成功概率",
        cmap="YlGnBu",
    )
    pivot = summary.pivot(index="u0", columns="m0", values="final_L_mean")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "revised_C_cold_start_final_L_heatmap.png",
        title="实验C：最终平台 A 综合份额热力图",
        xlabel="初始商户份额 m0",
        ylabel="初始用户份额 u0",
        cbar_label="最终 L_A",
        cmap="YlOrBr",
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.3))
    for label, group in seed_quality.groupby("resource_label"):
        group = group.sort_values("seed_quality")
        ax.plot(group["seed_quality"], group["success_probability"], marker="o", lw=2.4, label=label)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best")
    save_line_plot(
        FIG_DIR / "revised_C_seed_quality_success_curves.png",
        "实验C扩展：种子质量对冷启动成功率的影响",
        "种子质量",
        "成功概率",
    )

    pivot = fine.pivot(index="u0", columns="m0", values="success_probability")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "paper_C_cold_start_fine_success_heatmap.png",
        title="冷启动成功概率细粒度热力图",
        xlabel="初始商户份额 m0",
        ylabel="初始用户份额 u0",
        cbar_label="成功概率",
        cmap="YlGnBu",
    )
    pivot = fine.pivot(index="u0", columns="m0", values="final_L_mean")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "paper_C_cold_start_fine_final_L_heatmap.png",
        title="冷启动最终 L_A 细粒度热力图",
        xlabel="初始商户份额 m0",
        ylabel="初始用户份额 u0",
        cbar_label="最终 L_A",
        cmap="YlOrBr",
    )


def draw_experiment_d() -> None:
    summary = read_table("revised_D_multi_home_threshold_summary.csv")
    fine = read_table("paper_D_multi_home_fine_heatmap_summary.csv")
    paths = read_table("revised_D_multi_home_paths_summary.csv")

    pivot = summary.pivot(index="multi_home_cost", columns="rho", values="coexist_probability")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "revised_D_multi_home_coexist_heatmap.png",
        title="实验D：多归属成本-有效供给贡献下的共存概率",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本",
        cbar_label="共存概率",
        cmap="PuBuGn",
    )
    pivot = summary.pivot(index="multi_home_cost", columns="rho", values="final_multi_share")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "revised_D_multi_home_share_heatmap.png",
        title="实验D：多归属商户比例热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本",
        cbar_label="多归属比例",
        cmap="YlGnBu",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    path_palette = {
        "merchant_A_only": COLORS["blue"],
        "merchant_B_only": COLORS["orange"],
        "merchant_multi": COLORS["teal"],
        "L_A": COLORS["purple"],
    }
    for ax, (scenario, group) in zip(axes, paths.groupby("scenario_label")):
        group = group.sort_values("step")
        ax.plot(group["step"], group["merchant_A_only"], lw=2.2, color=path_palette["merchant_A_only"], label="A-only 商户")
        ax.plot(group["step"], group["merchant_B_only"], lw=2.2, color=path_palette["merchant_B_only"], label="B-only 商户")
        ax.plot(group["step"], group["merchant_multi"], lw=2.2, color=path_palette["merchant_multi"], label="Multi-home 商户")
        ax.plot(group["step"], group["L_A"], lw=2.2, color=path_palette["L_A"], ls="--", label="L_A")
        ax.set_title(scenario, fontsize=11)
        ax.set_xlabel("时间步")
        ax.grid(True, alpha=0.72)
    axes[0].set_ylabel("比例")
    axes[1].legend(loc="center right")
    fig.suptitle("实验D扩展：商户 A-only/B-only/Multi-home 动态路径", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "revised_D_multi_home_state_paths.png")
    plt.close(fig)

    pivot = fine.pivot(index="multi_home_cost", columns="rho", values="coexist_probability")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "paper_D_multi_home_fine_coexist_heatmap.png",
        title="低成本区间多归属共存概率热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本",
        cbar_label="共存概率",
        cmap="PuBuGn",
    )
    pivot = fine.pivot(index="multi_home_cost", columns="rho", values="final_multi_share")
    plot_heatmap(
        pivot,
        fig_path=FIG_DIR / "paper_D_multi_home_fine_share_heatmap.png",
        title="低成本区间多归属商户比例热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本",
        cbar_label="多归属比例",
        cmap="YlGnBu",
    )


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    draw_experiment_a()
    draw_experiment_b()
    draw_experiment_c()
    draw_experiment_d()
    print(f"已重绘图像目录：{FIG_DIR}")


if __name__ == "__main__":
    main()

