from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.integrate import solve_ivp

from abm_experiment_model import PlatformCompetitionABM, ExperimentParams


COLORS = {
    "blue": "#3B6FB6",
    "teal": "#2A9D8F",
    "orange": "#E76F51",
    "gold": "#E9C46A",
    "purple": "#7B61A8",
    "gray": "#6C757D",
    "red": "#C44E52",
}

N_USERS = 300
N_MERCHANTS = 150
STEPS = 70


def setup_dirs() -> tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, table_dir


def set_plot_style() -> None:
    font_path = Path(r"C:\Windows\Fonts\simhei.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 130,
            "savefig.dpi": 260,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.9,
            "grid.color": "#D7DEE8",
            "legend.frameon": False,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
        }
    )


def lock_state(l_a: float) -> str:
    if l_a >= 0.8:
        return "A_lock"
    if l_a <= 0.2:
        return "B_lock"
    return "coexist"


def time_to_threshold(df: pd.DataFrame, threshold: float) -> float | None:
    reached = df[df["L_A"] >= threshold]
    if reached.empty:
        return None
    return float(reached["step"].min())


def run_one(condition: dict, seed: int, steps: int = STEPS) -> pd.DataFrame:
    model = PlatformCompetitionABM(
        n_users=condition.get("n_users", N_USERS),
        n_merchants=condition.get("n_merchants", N_MERCHANTS),
        u0=condition.get("u0", 0.5),
        m0=condition.get("m0", 0.5),
        params=condition["params"],
        seed=seed,
        subsidy_policy=condition.get("subsidy_policy", "none"),
        allow_multi_home=condition.get("allow_multi_home", False),
        multi_home_share=condition.get("multi_home_share", 1.0),
        network_type=condition.get("network_type", "none"),
        cold_start_strategy=condition.get("cold_start_strategy", "none"),
        seed_users=condition.get("seed_users", 0),
        seed_merchants=condition.get("seed_merchants", 0),
    )
    df = model.run(steps)
    df["seed"] = seed
    for key, value in condition.items():
        if key != "params":
            df[key] = value
    return df


def build_model(condition: dict, seed: int) -> PlatformCompetitionABM:
    return PlatformCompetitionABM(
        n_users=condition.get("n_users", N_USERS),
        n_merchants=condition.get("n_merchants", N_MERCHANTS),
        u0=condition.get("u0", 0.5),
        m0=condition.get("m0", 0.5),
        params=condition["params"],
        seed=seed,
        subsidy_policy=condition.get("subsidy_policy", "none"),
        allow_multi_home=condition.get("allow_multi_home", False),
        multi_home_share=condition.get("multi_home_share", 1.0),
        network_type=condition.get("network_type", "none"),
        cold_start_strategy=condition.get("cold_start_strategy", "none"),
        seed_users=condition.get("seed_users", 0),
        seed_merchants=condition.get("seed_merchants", 0),
    )


def final_rows(df: pd.DataFrame, condition_cols: list[str], success_threshold: float = 0.6) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(condition_cols + ["seed"]):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(condition_cols + ["seed"], keys))
        group = group.sort_values("step")
        first = group.iloc[0]
        last = group.iloc[-1]
        spend = float(last["total_subsidy_spend"])
        rows.append(
            key_map
            | {
                "final_L": float(last["L_A"]),
                "concentration": float(last["concentration"]),
                "lock_state": lock_state(float(last["L_A"])),
                "success": float(last["L_A"] >= success_threshold),
                "time_to_success": time_to_threshold(group, success_threshold),
                "total_subsidy_spend": spend,
                "roi": (float(last["L_A"]) - float(first["L_A"])) / spend if spend > 0 else 0.0,
                "final_multi_share": float(last["merchant_multi"]),
            }
        )
    return pd.DataFrame(rows)


def summarize_final(final_df: pd.DataFrame, condition_cols: list[str]) -> pd.DataFrame:
    summary = final_df.groupby(condition_cols, as_index=False).agg(
        final_L_mean=("final_L", "mean"),
        final_L_std=("final_L", "std"),
        concentration_mean=("concentration", "mean"),
        success_probability=("success", "mean"),
        mean_time_to_success=("time_to_success", "mean"),
        subsidy_spend_mean=("total_subsidy_spend", "mean"),
        roi_mean=("roi", "mean"),
        final_multi_share=("final_multi_share", "mean"),
        lock_A_probability=("lock_state", lambda s: float((s == "A_lock").mean())),
        lock_B_probability=("lock_state", lambda s: float((s == "B_lock").mean())),
        coexist_probability=("lock_state", lambda s: float((s == "coexist").mean())),
    )
    return summary.fillna(0.0)


def plot_heatmap(
    pivot: pd.DataFrame,
    *,
    fig_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    cbar_label: str,
    cmap: str = "YlGnBu",
) -> None:
    fig, ax = plt.subplots(figsize=(8.1, 6.2))
    im = ax.imshow(pivot.values, origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)), [f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    for y in range(len(pivot.index)):
        for x in range(len(pivot.columns)):
            value = pivot.values[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7.5, color="#1f2933")
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)


def experiment_a_critical_heterogeneity(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5100, 5130)
    heterogeneity_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    runs = []
    user_segments = []
    merchant_segments = []
    for heterogeneity in heterogeneity_values:
        condition = {
            "experiment": "A_critical_heterogeneity",
            "heterogeneity_sigma": heterogeneity,
            "u0": 0.52,
            "m0": 0.52,
            "params": ExperimentParams(
                alpha=2.3,
                beta=2.3,
                heterogeneity_sigma=heterogeneity,
                sigma_user=0.25,
                sigma_merchant=0.25,
            ),
        }
        for seed in seeds:
            model = build_model(condition, seed)
            df = model.run(STEPS)
            df["seed"] = seed
            df["heterogeneity_sigma"] = heterogeneity
            runs.append(df)

            user_seg = model.user_segment_distribution()
            user_seg["seed"] = seed
            user_seg["heterogeneity_sigma"] = heterogeneity
            user_segments.append(user_seg)

            merchant_seg = model.merchant_segment_distribution()
            merchant_seg["seed"] = seed
            merchant_seg["heterogeneity_sigma"] = heterogeneity
            merchant_segments.append(merchant_seg)
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["heterogeneity_sigma"])
    summary = summarize_final(final_df, ["heterogeneity_sigma"])

    runs_df.to_csv(table_dir / "revised_A_critical_heterogeneity_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "revised_A_critical_heterogeneity_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_A_critical_heterogeneity_summary.csv", index=False, encoding="utf-8-sig")
    user_segment_df = pd.concat(user_segments, ignore_index=True)
    merchant_segment_df = pd.concat(merchant_segments, ignore_index=True)
    user_segment_summary = user_segment_df.groupby(["heterogeneity_sigma", "segment"], as_index=False).agg(
        share_on_A_mean=("share_on_A", "mean"),
        count_mean=("count", "mean"),
    )
    merchant_segment_summary = merchant_segment_df.groupby(["heterogeneity_sigma", "segment"], as_index=False).agg(
        A_only_share_mean=("A_only_share", "mean"),
        B_only_share_mean=("B_only_share", "mean"),
        multi_share_mean=("multi_share", "mean"),
        count_mean=("count", "mean"),
    )
    user_segment_summary.to_csv(table_dir / "revised_A_user_segment_summary.csv", index=False, encoding="utf-8-sig")
    merchant_segment_summary.to_csv(table_dir / "revised_A_merchant_segment_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = summary["heterogeneity_sigma"]
    ax.stackplot(
        x,
        summary["lock_A_probability"],
        summary["lock_B_probability"],
        summary["coexist_probability"],
        labels=["A锁定", "B锁定", "共存"],
        colors=[COLORS["blue"], COLORS["orange"], COLORS["teal"]],
        alpha=0.86,
    )
    ax.set_xlabel("个体异质性强度")
    ax.set_ylabel("结果概率")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.45)
    ax.legend(loc="center right")
    ax.set_title("实验A：临界区异质性对市场结构的影响", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "revised_A_heterogeneity_probabilities.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    x = summary["heterogeneity_sigma"].astype(float)
    mean_l = summary["final_L_mean"]
    std_l = summary["final_L_std"].fillna(0.0)
    lower = (mean_l - std_l).clip(0, 1)
    upper = (mean_l + std_l).clip(0, 1)
    ax.fill_between(x, lower, upper, color=COLORS["blue"], alpha=0.16, label="均值±标准差")
    ax.plot(x, mean_l, marker="o", linewidth=2.2, color=COLORS["blue"], label="最终 LA 均值")
    ax.axhline(0.6, color=COLORS["red"], linestyle=":", linewidth=1.4, label="A占优阈值")
    ax.set_xlabel("个体异质性强度")
    ax.set_ylabel("最终平台 A 综合份额")
    ax.set_xticks(x, [f"{v:.2f}" for v in x])
    ax.set_ylim(0, 1.03)
    ax.grid(True, axis="y", alpha=0.5)

    ax2 = ax.twinx()
    ax2.bar(x, summary["lock_B_probability"], width=0.12, color=COLORS["orange"], alpha=0.42, label="B锁定概率")
    ax2.set_ylabel("B锁定概率")
    ax2.set_ylim(0, 0.35)

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="lower left")
    ax.set_title("实验A：异质性强度、最终份额与失败风险", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "revised_A_heterogeneity_boxplot.png")
    plt.close(fig)

    high_heterogeneity = user_segment_summary[user_segment_summary["heterogeneity_sigma"] == max(heterogeneity_values)].copy()
    high_heterogeneity["estimated_A_users"] = high_heterogeneity["share_on_A_mean"] * high_heterogeneity["count_mean"]
    high_heterogeneity["share_among_A_users"] = high_heterogeneity["estimated_A_users"] / high_heterogeneity["estimated_A_users"].sum()
    high_heterogeneity = high_heterogeneity.sort_values("share_among_A_users", ascending=False)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    label_map = {
        "price_sensitive": "价格敏感型",
        "quality_sensitive": "质量敏感型",
        "subsidy_sensitive": "补贴敏感型",
        "inertial": "惯性型",
    }
    labels = high_heterogeneity["segment"].map(label_map)
    values = high_heterogeneity["share_among_A_users"]
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
    fig.savefig(fig_dir / "revised_A_user_segment_distribution.png")
    plt.close(fig)


def experiment_b_equal_budget_subsidy(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5200, 5230)
    budgets = [0, 50, 100, 200, 400, 800, 1200]
    policies = [
        ("none", "无补贴"),
        ("uniform", "统一补贴"),
        ("random", "随机补贴"),
        ("swing_user", "摇摆用户"),
        ("key_merchant", "关键商户"),
        ("two_sided_targeted", "双边精准"),
    ]
    runs = []
    for budget in budgets:
        for policy, label in policies:
            condition = {
                "experiment": "B_equal_budget_subsidy",
                "budget": budget,
                "subsidy_policy": policy,
                "policy_label": label,
                "u0": 0.30,
                "m0": 0.30,
                "params": ExperimentParams(
                    alpha=2.5,
                    beta=2.5,
                    heterogeneity_sigma=0.8,
                    subsidy_budget=float(budget),
                    max_subsidy_per_agent=999.0,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                ),
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["budget", "subsidy_policy", "policy_label"])
    summary = summarize_final(final_df, ["budget", "subsidy_policy", "policy_label"])

    runs_df.to_csv(table_dir / "revised_B_equal_budget_subsidy_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "revised_B_equal_budget_subsidy_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_B_equal_budget_subsidy_summary.csv", index=False, encoding="utf-8-sig")
    critical_rows = []
    for policy, label in policies:
        part = summary[summary["subsidy_policy"] == policy].sort_values("budget")
        reached = part[part["success_probability"] >= 0.5]
        critical_rows.append(
            {
                "subsidy_policy": policy,
                "policy_label": label,
                "critical_budget_B_star": float(reached.iloc[0]["budget"]) if not reached.empty else np.nan,
                "max_success_probability": float(part["success_probability"].max()),
                "best_roi": float(part["roi_mean"].max()),
            }
        )
    critical_df = pd.DataFrame(critical_rows)
    critical_df.to_csv(table_dir / "revised_B_critical_budget_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    palette = [COLORS["gray"], COLORS["blue"], COLORS["gold"], COLORS["teal"], COLORS["purple"], COLORS["orange"]]
    for (policy, label), color in zip(policies, palette):
        part = summary[summary["subsidy_policy"] == policy].sort_values("budget")
        ax.plot(part["budget"], part["success_probability"], marker="o", linewidth=2.0, color=color, label=label)
    ax.set_xlabel("每期补贴预算 B")
    ax.set_ylabel("跨越 L_A > 0.6 的概率")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.55)
    ax.legend(ncol=2, loc="lower right")
    ax.set_title("实验B：等预算下不同补贴策略的成功概率", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "revised_B_budget_success_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.3, 5.4))
    for (policy, label), color in zip(policies[1:], palette[1:]):
        part = summary[(summary["subsidy_policy"] == policy) & (summary["budget"] > 0)].sort_values("budget")
        ax.plot(part["budget"], part["roi_mean"], marker="s", linewidth=2.0, color=color, label=label)
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xlabel("每期补贴预算 B")
    ax.set_ylabel("ROI")
    ax.grid(True, alpha=0.55)
    ax.legend(ncol=2, loc="best")
    ax.set_title("实验B：等预算下不同补贴策略的 ROI", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "revised_B_budget_roi_curves.png")
    plt.close(fig)


def experiment_c_cold_start_heatmap(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5300, 5320)
    grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    runs = []
    for u0 in grid:
        for m0 in grid:
            condition = {
                "experiment": "C_cold_start_heatmap",
                "u0": u0,
                "m0": m0,
                "params": ExperimentParams(
                    alpha=2.7,
                    beta=2.7,
                    heterogeneity_sigma=0.5,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                    initial_a_inertia_boost=1.2,
                ),
                "n_users": 260,
                "n_merchants": 130,
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["u0", "m0"], success_threshold=0.6)
    summary = summarize_final(final_df, ["u0", "m0"])

    runs_df.to_csv(table_dir / "revised_C_cold_start_heatmap_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "revised_C_cold_start_heatmap_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_C_cold_start_heatmap_summary.csv", index=False, encoding="utf-8-sig")

    success_pivot = summary.pivot(index="m0", columns="u0", values="success_probability").sort_index().sort_index(axis=1)
    final_pivot = summary.pivot(index="m0", columns="u0", values="final_L_mean").sort_index().sort_index(axis=1)
    plot_heatmap(
        success_pivot,
        fig_path=fig_dir / "revised_C_cold_start_success_heatmap.png",
        title="实验C：冷启动成功概率热力图",
        xlabel="初始用户份额 u_A(0)",
        ylabel="初始商户份额 m_A(0)",
        cbar_label="P(L_A>0.6)",
        cmap="YlGnBu",
    )
    plot_heatmap(
        final_pivot,
        fig_path=fig_dir / "revised_C_cold_start_final_L_heatmap.png",
        title="实验C：最终平台 A 综合份额热力图",
        xlabel="初始用户份额 u_A(0)",
        ylabel="初始商户份额 m_A(0)",
        cbar_label="最终 L_A 均值",
        cmap="YlOrRd",
    )


def experiment_c_seed_quality_scan(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5600, 5630)
    boosts = [0.0, 0.6, 1.2, 1.8, 2.4]
    resource_points = [
        (0.35, 0.40, "用户35%-商户40%"),
        (0.40, 0.35, "用户40%-商户35%"),
        (0.40, 0.40, "用户40%-商户40%"),
        (0.45, 0.30, "用户45%-商户30%"),
        (0.30, 0.45, "用户30%-商户45%"),
    ]
    runs = []
    for u0, m0, label in resource_points:
        for boost in boosts:
            condition = {
                "experiment": "C_seed_quality_scan",
                "u0": u0,
                "m0": m0,
                "resource_label": label,
                "seed_quality": boost,
                "params": ExperimentParams(
                    alpha=2.7,
                    beta=2.7,
                    heterogeneity_sigma=0.5,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                    initial_a_inertia_boost=boost,
                ),
                "n_users": 260,
                "n_merchants": 130,
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["u0", "m0", "resource_label", "seed_quality"], success_threshold=0.6)
    summary = summarize_final(final_df, ["u0", "m0", "resource_label", "seed_quality"])
    runs_df.to_csv(table_dir / "revised_C_seed_quality_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "revised_C_seed_quality_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_C_seed_quality_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    palette = [COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"], COLORS["gold"]]
    for (_, _, label), color in zip(resource_points, palette):
        part = summary[summary["resource_label"] == label].sort_values("seed_quality")
        ax.plot(part["seed_quality"], part["success_probability"], marker="o", linewidth=2.0, color=color, label=label)
    ax.set_xlabel("种子黏性/质量强度")
    ax.set_ylabel("冷启动成功概率")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("实验C扩展：种子质量对冷启动成功率的影响", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "revised_C_seed_quality_success_curves.png")
    plt.close(fig)


def experiment_d_multi_home_threshold(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5400, 5430)
    costs = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0]
    rhos = [0.3, 0.5, 0.7, 0.9, 1.0]
    runs = []
    for cost in costs:
        for rho in rhos:
            condition = {
                "experiment": "D_multi_home_threshold",
                "multi_home_cost": cost,
                "rho": rho,
                "u0": 0.52,
                "m0": 0.52,
                "allow_multi_home": True,
                "multi_home_share": 1.0,
                "params": ExperimentParams(
                    alpha=2.3,
                    beta=2.3,
                    heterogeneity_sigma=0.5,
                    multi_home_cost=cost,
                    multi_home_visibility=rho,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                ),
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["multi_home_cost", "rho"])
    summary = summarize_final(final_df, ["multi_home_cost", "rho"])

    runs_df.to_csv(table_dir / "revised_D_multi_home_threshold_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "revised_D_multi_home_threshold_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_D_multi_home_threshold_summary.csv", index=False, encoding="utf-8-sig")

    coexist_pivot = summary.pivot(index="multi_home_cost", columns="rho", values="coexist_probability").sort_index().sort_index(axis=1)
    multi_pivot = summary.pivot(index="multi_home_cost", columns="rho", values="final_multi_share").sort_index().sort_index(axis=1)
    plot_heatmap(
        coexist_pivot,
        fig_path=fig_dir / "revised_D_multi_home_coexist_heatmap.png",
        title="实验D：多归属成本-有效供给贡献下的共存概率",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本 k_M",
        cbar_label="共存概率",
        cmap="YlGnBu",
    )
    plot_heatmap(
        multi_pivot,
        fig_path=fig_dir / "revised_D_multi_home_share_heatmap.png",
        title="实验D：多归属商户比例热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本 k_M",
        cbar_label="多归属商户比例",
        cmap="Greens",
    )


def experiment_d_multi_home_paths(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(5700, 5730)
    scenarios = [
        (0.0, 0.5, "低成本-中等贡献"),
        (0.5, 0.5, "中成本-中等贡献"),
        (1.2, 0.5, "高成本-中等贡献"),
        (0.0, 0.9, "低成本-高贡献"),
    ]
    runs = []
    for cost, rho, label in scenarios:
        condition = {
            "experiment": "D_multi_home_paths",
            "multi_home_cost": cost,
            "rho": rho,
            "scenario_label": label,
            "u0": 0.52,
            "m0": 0.52,
            "allow_multi_home": True,
            "multi_home_share": 1.0,
            "params": ExperimentParams(
                alpha=2.3,
                beta=2.3,
                heterogeneity_sigma=0.5,
                multi_home_cost=cost,
                multi_home_visibility=rho,
                sigma_user=0.25,
                sigma_merchant=0.25,
            ),
        }
        for seed in seeds:
            runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    summary = runs_df.groupby(["scenario_label", "step"], as_index=False).agg(
        merchant_A_only=("merchant_A_only", "mean"),
        merchant_B_only=("merchant_B_only", "mean"),
        merchant_multi=("merchant_multi", "mean"),
        L_A=("L_A", "mean"),
    )
    runs_df.to_csv(table_dir / "revised_D_multi_home_paths_runs.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "revised_D_multi_home_paths_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    for ax, (_, _, label) in zip(axes.ravel(), scenarios):
        part = summary[summary["scenario_label"] == label].sort_values("step")
        ax.plot(part["step"], part["merchant_A_only"], color=COLORS["blue"], linewidth=1.9, label="A-only")
        ax.plot(part["step"], part["merchant_B_only"], color=COLORS["orange"], linewidth=1.9, label="B-only")
        ax.plot(part["step"], part["merchant_multi"], color=COLORS["teal"], linewidth=1.9, label="Multi-home")
        ax.set_title(label, fontsize=11, pad=8)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.45)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.supxlabel("时间步", y=0.06)
    fig.supylabel("商户状态比例", x=0.04)
    fig.suptitle("实验D扩展：商户 A-only/B-only/Multi-home 动态路径", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.04, 0.10, 1.0, 0.95])
    fig.savefig(fig_dir / "revised_D_multi_home_state_paths.png")
    plt.close(fig)


def experiment_b_paper_budget_fine_scan(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(6100, 6150)
    budgets = [120, 140, 160, 180, 200, 220, 240, 260, 280, 300]
    policies = [
        ("uniform", "统一补贴"),
        ("random", "随机补贴"),
        ("two_sided_targeted", "双边精准"),
    ]
    runs = []
    for budget in budgets:
        for policy, label in policies:
            condition = {
                "experiment": "paper_B_budget_fine_scan",
                "budget": budget,
                "subsidy_policy": policy,
                "policy_label": label,
                "u0": 0.30,
                "m0": 0.30,
                "params": ExperimentParams(
                    alpha=2.5,
                    beta=2.5,
                    heterogeneity_sigma=0.8,
                    subsidy_budget=float(budget),
                    max_subsidy_per_agent=999.0,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                ),
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["budget", "subsidy_policy", "policy_label"])
    summary = summarize_final(final_df, ["budget", "subsidy_policy", "policy_label"])
    runs_df.to_csv(table_dir / "paper_B_budget_fine_scan_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "paper_B_budget_fine_scan_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "paper_B_budget_fine_scan_summary.csv", index=False, encoding="utf-8-sig")

    critical_rows = []
    for policy, label in policies:
        part = summary[summary["subsidy_policy"] == policy].sort_values("budget")
        reached = part[part["success_probability"] >= 0.5]
        critical_rows.append(
            {
                "subsidy_policy": policy,
                "policy_label": label,
                "critical_budget_B_star": float(reached.iloc[0]["budget"]) if not reached.empty else np.nan,
                "max_success_probability": float(part["success_probability"].max()),
                "best_roi": float(part["roi_mean"].max()),
            }
        )
    pd.DataFrame(critical_rows).to_csv(table_dir / "paper_B_critical_budget_fine_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    palette = [COLORS["blue"], COLORS["gold"], COLORS["orange"]]
    for (policy, label), color in zip(policies, palette):
        part = summary[summary["subsidy_policy"] == policy].sort_values("budget")
        ax.plot(part["budget"], part["success_probability"], marker="o", linewidth=2.2, color=color, label=label)
    ax.axhline(0.5, color=COLORS["red"], linestyle=":", linewidth=1.3, label="50%成功阈值")
    ax.set_xlabel("每期补贴预算 B")
    ax.set_ylabel("跨越 L_A > 0.6 的概率")
    ax.set_ylim(0, 1.03)
    ax.grid(True, alpha=0.55)
    ax.legend(loc="lower right")
    ax.set_title("补贴临界预算细粒度扫描", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "paper_B_budget_fine_success_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for (policy, label), color in zip(policies, palette):
        part = summary[summary["subsidy_policy"] == policy].sort_values("budget")
        ax.plot(part["budget"], part["roi_mean"], marker="s", linewidth=2.0, color=color, label=label)
    ax.axhline(0, color="#444444", linewidth=1.0)
    ax.set_xlabel("每期补贴预算 B")
    ax.set_ylabel("ROI")
    ax.grid(True, alpha=0.55)
    ax.legend(loc="best")
    ax.set_title("临界预算区间 ROI 对比", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "paper_B_budget_fine_roi_curves.png")
    plt.close(fig)


def experiment_c_paper_cold_start_fine_heatmap(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(6200, 6250)
    grid = [0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]
    runs = []
    for u0 in grid:
        for m0 in grid:
            condition = {
                "experiment": "paper_C_cold_start_fine_heatmap",
                "u0": u0,
                "m0": m0,
                "params": ExperimentParams(
                    alpha=2.7,
                    beta=2.7,
                    heterogeneity_sigma=0.5,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                    initial_a_inertia_boost=1.2,
                ),
                "n_users": 260,
                "n_merchants": 130,
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["u0", "m0"], success_threshold=0.6)
    summary = summarize_final(final_df, ["u0", "m0"])
    runs_df.to_csv(table_dir / "paper_C_cold_start_fine_heatmap_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "paper_C_cold_start_fine_heatmap_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "paper_C_cold_start_fine_heatmap_summary.csv", index=False, encoding="utf-8-sig")

    success_pivot = summary.pivot(index="m0", columns="u0", values="success_probability").sort_index().sort_index(axis=1)
    final_pivot = summary.pivot(index="m0", columns="u0", values="final_L_mean").sort_index().sort_index(axis=1)
    plot_heatmap(
        success_pivot,
        fig_path=fig_dir / "paper_C_cold_start_fine_success_heatmap.png",
        title="冷启动成功概率细粒度热力图",
        xlabel="初始用户份额 u_A(0)",
        ylabel="初始商户份额 m_A(0)",
        cbar_label="P(L_A>0.6)",
        cmap="YlGnBu",
    )
    plot_heatmap(
        final_pivot,
        fig_path=fig_dir / "paper_C_cold_start_fine_final_L_heatmap.png",
        title="冷启动最终 L_A 细粒度热力图",
        xlabel="初始用户份额 u_A(0)",
        ylabel="初始商户份额 m_A(0)",
        cbar_label="最终 L_A 均值",
        cmap="YlOrRd",
    )


def experiment_d_paper_multi_home_fine_heatmap(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(6300, 6350)
    costs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    rhos = [0.3, 0.4, 0.5, 0.6, 0.7]
    runs = []
    for cost in costs:
        for rho in rhos:
            condition = {
                "experiment": "paper_D_multi_home_fine_heatmap",
                "multi_home_cost": cost,
                "rho": rho,
                "u0": 0.52,
                "m0": 0.52,
                "allow_multi_home": True,
                "multi_home_share": 1.0,
                "params": ExperimentParams(
                    alpha=2.3,
                    beta=2.3,
                    heterogeneity_sigma=0.5,
                    multi_home_cost=cost,
                    multi_home_visibility=rho,
                    sigma_user=0.25,
                    sigma_merchant=0.25,
                ),
            }
            for seed in seeds:
                runs.append(run_one(condition, seed))
    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["multi_home_cost", "rho"])
    summary = summarize_final(final_df, ["multi_home_cost", "rho"])
    runs_df.to_csv(table_dir / "paper_D_multi_home_fine_heatmap_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "paper_D_multi_home_fine_heatmap_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "paper_D_multi_home_fine_heatmap_summary.csv", index=False, encoding="utf-8-sig")

    coexist_pivot = summary.pivot(index="multi_home_cost", columns="rho", values="coexist_probability").sort_index().sort_index(axis=1)
    multi_pivot = summary.pivot(index="multi_home_cost", columns="rho", values="final_multi_share").sort_index().sort_index(axis=1)
    plot_heatmap(
        coexist_pivot,
        fig_path=fig_dir / "paper_D_multi_home_fine_coexist_heatmap.png",
        title="低成本区间多归属共存概率热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本 k_M",
        cbar_label="共存概率",
        cmap="YlGnBu",
    )
    plot_heatmap(
        multi_pivot,
        fig_path=fig_dir / "paper_D_multi_home_fine_share_heatmap.png",
        title="低成本区间多归属商户比例热力图",
        xlabel="多归属有效供给贡献 rho",
        ylabel="多归属成本 k_M",
        cbar_label="多归属商户比例",
        cmap="Greens",
    )


def experiment_quality_advantage(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(7300, 7315)
    deltas = [0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]
    strategies = [
        ("no_quality", "无质量优势", False, False, "none", 0.0),
        ("user_quality", "用户侧质量优势", True, False, "none", 0.0),
        ("merchant_quality", "商户侧质量优势", False, True, "none", 0.0),
        ("two_sided_quality", "双边质量优势", True, True, "none", 0.0),
        ("quality_plus_subsidy", "双边质量+少量精准补贴", True, True, "two_sided_targeted", 80.0),
    ]
    runs = []
    user_segments = []
    merchant_segments = []
    for delta_q in deltas:
        for strategy, label, user_quality, merchant_quality, subsidy_policy, budget in strategies:
            params = ExperimentParams(
                alpha=2.5,
                beta=2.5,
                heterogeneity_sigma=0.8,
                q_a_user=delta_q if user_quality else 0.0,
                q_b_user=0.0,
                q_a_merchant=delta_q if merchant_quality else 0.0,
                q_b_merchant=0.0,
                subsidy_budget=budget,
                max_subsidy_per_agent=999.0,
                sigma_user=0.25,
                sigma_merchant=0.25,
            )
            condition = {
                "experiment": "quality_advantage",
                "delta_q": delta_q,
                "quality_strategy": strategy,
                "quality_label": label,
                "u0": 0.30,
                "m0": 0.30,
                "n_users": 220,
                "n_merchants": 110,
                "params": params,
                "subsidy_policy": subsidy_policy,
            }
            for seed in seeds:
                model = build_model(condition, seed)
                df = model.run(STEPS)
                df["seed"] = seed
                df["delta_q"] = delta_q
                df["quality_strategy"] = strategy
                df["quality_label"] = label
                runs.append(df)

                if delta_q == 0.75 and strategy in {"user_quality", "two_sided_quality", "quality_plus_subsidy"}:
                    user_seg = model.user_segment_distribution()
                    user_seg["seed"] = seed
                    user_seg["delta_q"] = delta_q
                    user_seg["quality_strategy"] = strategy
                    user_seg["quality_label"] = label
                    user_segments.append(user_seg)

                    merchant_seg = model.merchant_segment_distribution()
                    merchant_seg["seed"] = seed
                    merchant_seg["delta_q"] = delta_q
                    merchant_seg["quality_strategy"] = strategy
                    merchant_seg["quality_label"] = label
                    merchant_segments.append(merchant_seg)

    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["delta_q", "quality_strategy", "quality_label"], success_threshold=0.6)
    summary = summarize_final(final_df, ["delta_q", "quality_strategy", "quality_label"])
    runs_df.to_csv(table_dir / "abm_quality_advantage_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "abm_quality_advantage_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "abm_quality_advantage_summary.csv", index=False, encoding="utf-8-sig")

    critical_rows = []
    for strategy, label, *_ in strategies:
        part = summary[summary["quality_strategy"] == strategy].sort_values("delta_q")
        reached = part[part["success_probability"] >= 0.5]
        critical_rows.append(
            {
                "quality_strategy": strategy,
                "quality_label": label,
                "critical_delta_q": float(reached.iloc[0]["delta_q"]) if not reached.empty else np.nan,
                "max_success_probability": float(part["success_probability"].max()),
                "max_final_L_mean": float(part["final_L_mean"].max()),
            }
        )
    pd.DataFrame(critical_rows).to_csv(table_dir / "abm_quality_advantage_critical.csv", index=False, encoding="utf-8-sig")

    if user_segments:
        user_segment_df = pd.concat(user_segments, ignore_index=True)
        user_segment_summary = user_segment_df.groupby(["quality_strategy", "quality_label", "segment"], as_index=False).agg(
            share_on_A_mean=("share_on_A", "mean"),
            count_mean=("count", "mean"),
        )
        user_segment_summary.to_csv(table_dir / "abm_quality_advantage_user_segments.csv", index=False, encoding="utf-8-sig")
    else:
        user_segment_summary = pd.DataFrame()

    if merchant_segments:
        merchant_segment_df = pd.concat(merchant_segments, ignore_index=True)
        merchant_segment_summary = merchant_segment_df.groupby(["quality_strategy", "quality_label", "segment"], as_index=False).agg(
            A_only_share_mean=("A_only_share", "mean"),
            B_only_share_mean=("B_only_share", "mean"),
            multi_share_mean=("multi_share", "mean"),
            count_mean=("count", "mean"),
        )
        merchant_segment_summary.to_csv(table_dir / "abm_quality_advantage_merchant_segments.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    palette = [COLORS["gray"], COLORS["blue"], COLORS["teal"], COLORS["orange"], COLORS["purple"]]
    for (strategy, label, *_), color in zip(strategies, palette):
        part = summary[summary["quality_strategy"] == strategy].sort_values("delta_q")
        ax.plot(part["delta_q"], part["success_probability"], marker="o", linewidth=2.1, color=color, label=label)
    ax.axhline(0.5, color=COLORS["red"], linestyle=":", linewidth=1.3, label="50%成功阈值")
    ax.set_xlabel("平台 A 服务质量优势 Δq")
    ax.set_ylabel("A 成功概率")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.52)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("服务质量优势对弱势平台逆袭概率的影响", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "abm_quality_advantage_success_curves.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for (strategy, label, *_), color in zip(strategies, palette):
        part = summary[summary["quality_strategy"] == strategy].sort_values("delta_q")
        ax.plot(part["delta_q"], part["final_L_mean"], marker="s", linewidth=2.1, color=color, label=label)
    ax.axhline(0.6, color=COLORS["red"], linestyle=":", linewidth=1.3, label="A占优阈值")
    ax.set_xlabel("平台 A 服务质量优势 Δq")
    ax.set_ylabel("最终 L_A 均值")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, alpha=0.52)
    ax.legend(loc="lower right", ncol=2)
    ax.set_title("服务质量优势对最终综合份额的影响", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "abm_quality_advantage_final_L.png")
    plt.close(fig)

    if not user_segment_summary.empty:
        pivot = user_segment_summary.pivot(index="segment", columns="quality_label", values="share_on_A_mean")
        preferred_cols = [label for _, label, *_ in strategies if label in pivot.columns]
        pivot = pivot[preferred_cols]
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        pivot.plot(kind="bar", ax=ax, color=[COLORS["orange"], COLORS["purple"]])
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("用户类型")
        ax.set_ylabel("平台 A 用户占比")
        ax.grid(True, axis="y", alpha=0.45)
        ax.legend(loc="lower right")
        ax.set_title("质量优势下不同用户类型的平台 A 留存比例", fontsize=13, pad=12)
        fig.tight_layout()
        fig.savefig(fig_dir / "abm_quality_advantage_user_segments.png")
        plt.close(fig)


def experiment_mechanism_ablation(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(7100, 7130)
    versions = [
        (
            "Full ABM",
            "保留全部机制",
            ExperimentParams(alpha=2.5, beta=2.5, heterogeneity_sigma=0.8, subsidy_budget=140.0, max_subsidy_per_agent=999.0, multi_home_cost=0.10, multi_home_visibility=0.40, sigma_user=0.25, sigma_merchant=0.25),
            "two_sided_targeted",
            True,
        ),
        (
            "w/o heterogeneity",
            "去掉个体异质性",
            ExperimentParams(alpha=2.5, beta=2.5, heterogeneity_sigma=0.0, subsidy_budget=140.0, max_subsidy_per_agent=999.0, multi_home_cost=0.10, multi_home_visibility=0.40, sigma_user=0.25, sigma_merchant=0.25),
            "two_sided_targeted",
            True,
        ),
        (
            "w/o inertia",
            "去掉惯性/切换成本",
            ExperimentParams(alpha=2.5, beta=2.5, heterogeneity_sigma=0.8, subsidy_budget=140.0, max_subsidy_per_agent=999.0, multi_home_cost=0.10, multi_home_visibility=0.40, sigma_user=0.25, sigma_merchant=0.25, user_inertia_mean=0.0, merchant_inertia_mean=0.0),
            "two_sided_targeted",
            True,
        ),
        (
            "w/o targeted rule",
            "精准补贴改为随机补贴",
            ExperimentParams(alpha=2.5, beta=2.5, heterogeneity_sigma=0.8, subsidy_budget=140.0, max_subsidy_per_agent=999.0, multi_home_cost=0.10, multi_home_visibility=0.40, sigma_user=0.25, sigma_merchant=0.25),
            "random",
            True,
        ),
        (
            "w/o multi-home",
            "商户不可多归属",
            ExperimentParams(alpha=2.5, beta=2.5, heterogeneity_sigma=0.8, subsidy_budget=140.0, max_subsidy_per_agent=999.0, sigma_user=0.25, sigma_merchant=0.25),
            "two_sided_targeted",
            False,
        ),
        (
            "w/o network effect",
            "去掉双边网络效应",
            ExperimentParams(alpha=0.0, beta=0.0, heterogeneity_sigma=0.8, subsidy_budget=140.0, max_subsidy_per_agent=999.0, multi_home_cost=0.10, multi_home_visibility=0.40, sigma_user=0.25, sigma_merchant=0.25),
            "two_sided_targeted",
            True,
        ),
    ]
    runs = []
    for version, removed_mechanism, params, policy, allow_multi_home in versions:
        condition = {
            "experiment": "mechanism_ablation",
            "version": version,
            "removed_mechanism": removed_mechanism,
            "u0": 0.30,
            "m0": 0.30,
            "params": params,
            "subsidy_policy": policy,
            "allow_multi_home": allow_multi_home,
            "multi_home_share": 1.0,
        }
        for seed in seeds:
            runs.append(run_one(condition, seed))

    runs_df = pd.concat(runs, ignore_index=True)
    final_df = final_rows(runs_df, ["version", "removed_mechanism"], success_threshold=0.6)
    summary = summarize_final(final_df, ["version", "removed_mechanism"])
    order = [v[0] for v in versions]
    summary["version"] = pd.Categorical(summary["version"], categories=order, ordered=True)
    summary = summary.sort_values("version")

    runs_df.to_csv(table_dir / "abm_mechanism_ablation_runs.csv", index=False, encoding="utf-8-sig")
    final_df.to_csv(table_dir / "abm_mechanism_ablation_final.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "abm_mechanism_ablation_summary.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    x = np.arange(len(summary))
    width = 0.36
    ax.bar(x - width / 2, summary["success_probability"], width, color=COLORS["blue"], label="成功概率")
    ax.bar(x + width / 2, summary["lock_A_probability"], width, color=COLORS["orange"], label="A锁定概率")
    ax.set_xticks(x, summary["version"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("概率")
    ax.grid(True, axis="y", alpha=0.45)
    ax.legend(loc="upper right")
    ax.set_title("机制消融：去掉关键机制后的结果变化", fontsize=13, pad=12)
    for idx, value in enumerate(summary["success_probability"]):
        ax.text(idx - width / 2, min(value + 0.03, 1.03), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "abm_mechanism_ablation_probabilities.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.errorbar(
        x,
        summary["final_L_mean"],
        yerr=summary["final_L_std"],
        fmt="o-",
        color=COLORS["teal"],
        ecolor="#91C9C0",
        capsize=4,
        linewidth=2.0,
    )
    ax.axhline(0.6, color=COLORS["red"], linestyle=":", linewidth=1.3, label="成功阈值")
    ax.set_xticks(x, summary["version"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("最终 L_A")
    ax.grid(True, axis="y", alpha=0.45)
    ax.legend(loc="lower right")
    ax.set_title("机制消融：最终综合份额均值与波动", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "abm_mechanism_ablation_final_L.png")
    plt.close(fig)


def ode_l_path(alpha: float, beta: float, u0: float, m0: float, steps: int = STEPS) -> pd.DataFrame:
    def rhs(_t: float, y: np.ndarray) -> list[float]:
        u, m = np.clip(y, 0.0, 1.0)
        p_u = 1.0 / (1.0 + np.exp(np.clip(-1.8 * alpha * (2.0 * m - 1.0), -700, 700)))
        p_m = 1.0 / (1.0 + np.exp(np.clip(-1.8 * beta * (2.0 * u - 1.0), -700, 700)))
        return [p_u - u, p_m - m]

    t_eval = np.arange(steps + 1, dtype=float)
    sol = solve_ivp(rhs, (0.0, float(steps)), [u0, m0], t_eval=t_eval, rtol=1e-8, atol=1e-8)
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    u = np.clip(sol.y[0], 0.0, 1.0)
    m = np.clip(sol.y[1], 0.0, 1.0)
    return pd.DataFrame({"step": t_eval.astype(int), "ode_u_A": u, "ode_m_A": m, "ode_L_A": 0.5 * (u + m)})


def experiment_abm_ode_comparison(fig_dir: Path, table_dir: Path) -> None:
    seeds = range(7200, 7230)
    scenarios = [
        ("弱网络效应", 0.8, 0.8, 0.50, 0.50),
        ("强网络效应", 2.8, 2.8, 0.55, 0.55),
        ("冷启动临界区", 2.3, 2.3, 0.52, 0.52),
    ]
    runs = []
    ode_rows = []
    for label, alpha, beta, u0, m0 in scenarios:
        ode = ode_l_path(alpha, beta, u0, m0)
        ode["scenario"] = label
        ode_rows.append(ode)
        condition = {
            "experiment": "abm_ode_comparison",
            "scenario": label,
            "u0": u0,
            "m0": m0,
            "params": ExperimentParams(
                alpha=alpha,
                beta=beta,
                heterogeneity_sigma=0.5,
                sigma_user=0.25,
                sigma_merchant=0.25,
                user_inertia_mean=0.10,
                merchant_inertia_mean=0.10,
            ),
        }
        for seed in seeds:
            runs.append(run_one(condition, seed))

    runs_df = pd.concat(runs, ignore_index=True)
    ode_df = pd.concat(ode_rows, ignore_index=True)
    abm_path = runs_df.groupby(["scenario", "step"], as_index=False).agg(
        abm_L_A_mean=("L_A", "mean"),
        abm_L_A_std=("L_A", "std"),
        abm_u_A_mean=("u_A", "mean"),
        abm_merchant_A_mean=("merchant_A_presence", "mean"),
    )
    compare_path = abm_path.merge(ode_df[["scenario", "step", "ode_L_A"]], on=["scenario", "step"], how="left")
    final_df = final_rows(runs_df, ["scenario"], success_threshold=0.6)
    summary = summarize_final(final_df, ["scenario"])
    ode_final = ode_df.sort_values("step").groupby("scenario", as_index=False).tail(1)[["scenario", "ode_L_A"]]
    summary = summary.merge(ode_final, on="scenario", how="left")
    summary["mean_gap_ABM_minus_ODE"] = summary["final_L_mean"] - summary["ode_L_A"]

    runs_df.to_csv(table_dir / "abm_ode_model_compare_runs.csv", index=False, encoding="utf-8-sig")
    ode_df.to_csv(table_dir / "abm_ode_model_compare_ode_path.csv", index=False, encoding="utf-8-sig")
    compare_path.to_csv(table_dir / "abm_ode_model_compare_path_summary.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(table_dir / "abm_ode_model_compare_summary.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), sharey=True)
    for ax, (label, _, _, _, _) in zip(axes, scenarios):
        part = compare_path[compare_path["scenario"] == label].sort_values("step")
        lower = (part["abm_L_A_mean"] - part["abm_L_A_std"]).clip(0, 1)
        upper = (part["abm_L_A_mean"] + part["abm_L_A_std"]).clip(0, 1)
        ax.fill_between(part["step"].to_numpy(), lower.to_numpy(), upper.to_numpy(), color=COLORS["blue"], alpha=0.18, label="ABM均值±标准差")
        ax.plot(part["step"], part["abm_L_A_mean"], color=COLORS["blue"], linewidth=2.0, label="ABM均值")
        ax.plot(part["step"], part["ode_L_A"], color=COLORS["orange"], linewidth=2.0, linestyle="--", label="ODE")
        ax.axhline(0.6, color=COLORS["red"], linestyle=":", linewidth=1.2)
        ax.set_title(label, fontsize=11, pad=8)
        ax.set_xlabel("时间步")
        ax.grid(True, alpha=0.45)
    axes[0].set_ylabel("平台 A 综合份额 L_A")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("ABM 与 ODE 在相同宏观机制下的动态对比", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0.12, 1, 0.93])
    fig.savefig(fig_dir / "abm_ode_model_compare_paths.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    order = [s[0] for s in scenarios]
    summary["scenario"] = pd.Categorical(summary["scenario"], categories=order, ordered=True)
    summary = summary.sort_values("scenario")
    x = np.arange(len(summary))
    ax.bar(x - 0.18, summary["final_L_mean"], 0.36, color=COLORS["blue"], label="ABM最终均值")
    ax.bar(x + 0.18, summary["ode_L_A"], 0.36, color=COLORS["orange"], label="ODE最终值")
    ax.errorbar(x - 0.18, summary["final_L_mean"], yerr=summary["final_L_std"], fmt="none", ecolor="#333333", capsize=4, linewidth=1.0)
    ax.set_xticks(x, summary["scenario"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("最终 L_A")
    ax.grid(True, axis="y", alpha=0.45)
    ax.legend(loc="upper left")
    ax.set_title("ABM 与 ODE 最终结果对比", fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "abm_ode_model_compare_final.png")
    plt.close(fig)


def main() -> None:
    fig_dir, table_dir = setup_dirs()
    set_plot_style()
    experiment_a_critical_heterogeneity(fig_dir, table_dir)
    experiment_b_equal_budget_subsidy(fig_dir, table_dir)
    experiment_c_cold_start_heatmap(fig_dir, table_dir)
    experiment_c_seed_quality_scan(fig_dir, table_dir)
    experiment_d_multi_home_threshold(fig_dir, table_dir)
    experiment_d_multi_home_paths(fig_dir, table_dir)
    experiment_quality_advantage(fig_dir, table_dir)
    experiment_mechanism_ablation(fig_dir, table_dir)
    experiment_abm_ode_comparison(fig_dir, table_dir)
    print(f"Saved revised ABM experiment figures to: {fig_dir}")
    print(f"Saved revised ABM experiment tables to: {table_dir}")


if __name__ == "__main__":
    main()

