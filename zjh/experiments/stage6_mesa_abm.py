import os
from dataclasses import replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.mesa_config import MesaABMParams
from src.mesa_abm_model import TwoSidedPlatformABM
from src.utils import ensure_dir, setup_matplotlib


STRATEGY_LABELS = {
    "none": "无策略",
    "user_subsidy": "用户补贴",
    "merchant_subsidy": "商户补贴",
    "bilateral_subsidy": "双边补贴",
    "greedy": "贪心策略",
    "long_term": "长期策略",
    "dynamic": "动态策略",
}

USER_TYPE_LABELS = {
    "price_sensitive": "价格敏感型",
    "quality_sensitive": "质量敏感型",
    "inertial": "惯性用户",
    "normal": "普通用户",
}

MERCHANT_TYPE_LABELS = {
    "large": "大商户",
    "small_medium": "中小商户",
    "new": "新商户",
    "multi_home": "多归属商户",
}


def run_one_abm(params: MesaABMParams):
    model = TwoSidedPlatformABM(params)
    model.run_model()
    return pd.DataFrame(model.history)


def run_repeated_abm(base_params: MesaABMParams, strategy: str, n_runs: int):
    all_runs = []

    for run_id in range(n_runs):
        params = replace(
            base_params,
            strategy=strategy,
            seed=base_params.seed + run_id,
        )

        df = run_one_abm(params)
        df["run_id"] = run_id
        df["strategy"] = strategy
        all_runs.append(df)

    return pd.concat(all_runs, ignore_index=True)


def summarize_final(df: pd.DataFrame):
    final_df = df.sort_values("t").groupby(["strategy", "run_id"]).tail(1)

    summary = final_df.groupby("strategy").agg(
        final_user_share_mean=("user_share_A", "mean"),
        final_user_share_std=("user_share_A", "std"),
        final_merchant_share_mean=("merchant_share_A", "mean"),
        final_merchant_share_std=("merchant_share_A", "std"),
        user_satisfaction_mean=("avg_user_utility", "mean"),
        merchant_satisfaction_mean=("avg_merchant_utility", "mean"),
        user_multi_home_mean=("user_multi_home_rate", "mean"),
        merchant_multi_home_mean=("merchant_multi_home_rate", "mean"),
        lock_in_index_mean=("lock_in_index", "mean"),
        lock_in_index_std=("lock_in_index", "std"),
        cum_profit_mean=("cum_profit_A", "mean"),
        cum_profit_std=("cum_profit_A", "std"),
    ).reset_index()

    return summary


def aggregate_time_series(df: pd.DataFrame):
    """
    按 strategy 和 t 聚合多次重复实验的时间序列均值。

    注意：
    t 是分组变量，不能再次放入 numeric_cols，
    否则 reset_index() 时会出现 cannot insert t, already exists。
    """
    group_cols = ["strategy", "t"]

    numeric_cols = [
        col for col in df.columns
        if col not in group_cols + ["run_id"]
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    grouped = df.groupby(group_cols)[numeric_cols].mean().reset_index()

    return grouped

def plot_basic_timeseries(df: pd.DataFrame, output_dir: str, filename: str, title: str):
    grouped = aggregate_time_series(df)

    strategies = grouped["strategy"].unique()

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    plot_items = [
        ("user_share_A", "平台 A 用户份额"),
        ("merchant_share_A", "平台 A 商户份额"),
        ("avg_user_utility", "用户满意度"),
        ("avg_merchant_utility", "商户满意度"),
        ("lock_in_index", "市场锁定指数"),
        ("cum_profit_A", "累计净收益"),
    ]

    for strategy in strategies:
        sub = grouped[grouped["strategy"] == strategy]
        label = STRATEGY_LABELS.get(strategy, strategy)

        for ax, (col, ylabel) in zip(axes, plot_items):
            ax.plot(sub["t"], sub[col], label=label)

    for ax, (_, ylabel) in zip(axes, plot_items):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("时间")
    axes[-2].set_xlabel("时间")

    fig.suptitle(title, fontsize=16)

    fig.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)


def plot_final_summary(summary: pd.DataFrame, output_dir: str, filename: str, title: str):
    summary = summary.copy()
    summary["label"] = summary["strategy"].map(lambda s: STRATEGY_LABELS.get(s, s))

    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    width = 0.18

    profit_norm = summary["cum_profit_mean"] / max(abs(summary["cum_profit_mean"]).max(), 1e-8)

    plt.figure(figsize=(13, 6))

    plt.bar(
        x - 1.5 * width,
        summary["final_user_share_mean"],
        width,
        yerr=summary["final_user_share_std"],
        capsize=4,
        label="平均用户份额"
    )

    plt.bar(
        x - 0.5 * width,
        summary["final_merchant_share_mean"],
        width,
        yerr=summary["final_merchant_share_std"],
        capsize=4,
        label="平均商户份额"
    )

    plt.bar(
        x + 0.5 * width,
        summary["lock_in_index_mean"],
        width,
        yerr=summary["lock_in_index_std"],
        capsize=4,
        label="市场锁定指数"
    )

    plt.bar(
        x + 1.5 * width,
        profit_norm,
        width,
        label="归一化累计净收益"
    )

    plt.xticks(x, labels, rotation=15)
    plt.ylim(0, 1.15)
    plt.ylabel("指标值")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Mesa 实验 1：异质性用户下的补贴效果
# ============================================================

def mesa_experiment_1_heterogeneous_subsidy(output_root: str):
    output_dir = os.path.join(output_root, "mesa_exp1_heterogeneous_subsidy")
    ensure_dir(output_dir)

    base_params = MesaABMParams(
        seed=2026,
        n_users=1000,
        n_merchants=300,
        max_steps=300,
        n_runs=30,
        x0=0.35,
        y0=0.35,
    )

    strategies = [
        "none",
        "user_subsidy",
        "merchant_subsidy",
        "bilateral_subsidy",
        "dynamic",
    ]

    all_results = []

    for strategy in strategies:
        print(f"  Mesa 实验1：正在运行 {strategy}")
        df = run_repeated_abm(
            base_params=base_params,
            strategy=strategy,
            n_runs=base_params.n_runs
        )
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp1_timeseries.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = summarize_final(result_df)
    summary.to_csv(
        os.path.join(output_dir, "mesa_exp1_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_basic_timeseries(
        result_df,
        output_dir,
        filename="mesa_exp1_basic_timeseries.png",
        title="Mesa 实验1：异质性用户下的补贴效果"
    )

    plot_final_summary(
        summary,
        output_dir,
        filename="mesa_exp1_final_summary.png",
        title="Mesa 实验1：不同补贴策略最终结果比较"
    )

    # 用户类型份额变化：重点看价格敏感型与质量敏感型
    grouped = aggregate_time_series(result_df)

    plt.figure(figsize=(11, 6))

    for strategy in strategies:
        sub = grouped[grouped["strategy"] == strategy]
        label = STRATEGY_LABELS.get(strategy, strategy)

        plt.plot(
            sub["t"],
            sub["user_share_A_price_sensitive"],
            linestyle="-",
            label=f"{label}-价格敏感型"
        )

        plt.plot(
            sub["t"],
            sub["user_share_A_quality_sensitive"],
            linestyle="--",
            label=f"{label}-质量敏感型"
        )

    plt.title("Mesa 实验1：价格敏感型与质量敏感型用户的份额变化")
    plt.xlabel("时间")
    plt.ylabel("平台 A 用户份额")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)

    plt.savefig(
        os.path.join(output_dir, "mesa_exp1_user_type_share.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Mesa 实验 2：商户多归属对市场锁定的影响
# ============================================================

def mesa_experiment_2_merchant_multihoming(output_root: str):
    output_dir = os.path.join(output_root, "mesa_exp2_merchant_multihoming")
    ensure_dir(output_dir)

    multi_values = [0.0, 0.25, 0.5, 0.75, 1.0]

    all_results = []

    for multi_prob in multi_values:
        print(f"  Mesa 实验2：正在运行商户多归属概率 {multi_prob}")

        base_params = MesaABMParams(
            seed=3030,
            n_users=1000,
            n_merchants=300,
            max_steps=300,
            n_runs=30,

            # 为观察锁定，设置平台 A 初始略有优势
            x0=0.50,
            y0=0.50,

            # 增大多归属阈值，使商户更容易同时入驻两个平台
            multi_home_gap=0.80,

            # 不采取主动补贴，观察多归属对自然锁定的影响
            strategy="none",

            merchant_multi_home_override=multi_prob,
        )

        df = run_repeated_abm(
            base_params=base_params,
            strategy="none",
            n_runs=base_params.n_runs
        )

        df["merchant_multi_home_prob"] = multi_prob
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp2_timeseries.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    final_df = result_df.sort_values("t").groupby(
        ["merchant_multi_home_prob", "run_id"]
    ).tail(1)

    summary = final_df.groupby("merchant_multi_home_prob").agg(
        final_user_share_mean=("user_share_A", "mean"),
        final_user_share_std=("user_share_A", "std"),
        final_merchant_share_mean=("merchant_share_A", "mean"),
        final_merchant_share_std=("merchant_share_A", "std"),
        merchant_multi_home_rate_mean=("merchant_multi_home_rate", "mean"),
        lock_in_index_mean=("lock_in_index", "mean"),
        lock_in_index_std=("lock_in_index", "std"),
        user_satisfaction_mean=("avg_user_utility", "mean"),
        merchant_satisfaction_mean=("avg_merchant_utility", "mean"),
        cum_profit_mean=("cum_profit_A", "mean"),
    ).reset_index()

    summary.to_csv(
        os.path.join(output_dir, "mesa_exp2_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # 图：多归属概率与锁定指数
    plt.figure(figsize=(10, 6))

    plt.errorbar(
        summary["merchant_multi_home_prob"],
        summary["lock_in_index_mean"],
        yerr=summary["lock_in_index_std"],
        marker="o",
        capsize=4,
        label="市场锁定指数"
    )

    plt.plot(
        summary["merchant_multi_home_prob"],
        summary["final_user_share_mean"],
        marker="s",
        label="最终用户份额"
    )

    plt.plot(
        summary["merchant_multi_home_prob"],
        summary["final_merchant_share_mean"],
        marker="^",
        label="最终商户份额"
    )

    plt.title("Mesa 实验2：商户多归属概率对市场锁定的影响")
    plt.xlabel("商户多归属概率")
    plt.ylabel("指标值")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(
        os.path.join(output_dir, "mesa_exp2_multihoming_lockin.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Mesa 实验 3：长期策略稳健性
# ============================================================

def mesa_experiment_3_long_term_robustness(output_root: str):
    output_dir = os.path.join(output_root, "mesa_exp3_long_term_robustness")
    ensure_dir(output_dir)

    base_params = MesaABMParams(
        seed=4040,
        n_users=1000,
        n_merchants=300,
        max_steps=300,
        n_runs=30,
        x0=0.35,
        y0=0.35,
    )

    strategies = [
        "greedy",
        "long_term",
        "dynamic",
    ]

    all_results = []

    for strategy in strategies:
        print(f"  Mesa 实验3：正在运行 {strategy}")

        df = run_repeated_abm(
            base_params=base_params,
            strategy=strategy,
            n_runs=base_params.n_runs
        )

        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp3_timeseries.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    summary = summarize_final(result_df)

    summary.to_csv(
        os.path.join(output_dir, "mesa_exp3_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    plot_basic_timeseries(
        result_df,
        output_dir,
        filename="mesa_exp3_basic_timeseries.png",
        title="Mesa 实验3：长期策略稳健性比较"
    )

    plot_final_summary(
        summary,
        output_dir,
        filename="mesa_exp3_final_summary.png",
        title="Mesa 实验3：贪心、长期与动态策略综合比较"
    )


# ============================================================
# 阶段 6 总入口
# ============================================================

def run_stage6(output_root: str):
    setup_matplotlib()

    output_dir = os.path.join(output_root, "stage6_mesa_abm")
    ensure_dir(output_dir)

    print("开始运行 Mesa 实验1：异质性用户下的补贴效果...")
    mesa_experiment_1_heterogeneous_subsidy(output_dir)

    print("开始运行 Mesa 实验2：商户多归属对市场锁定的影响...")
    mesa_experiment_2_merchant_multihoming(output_dir)

    print("开始运行 Mesa 实验3：长期策略稳健性...")
    mesa_experiment_3_long_term_robustness(output_dir)

    print(f"Mesa 阶段全部完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    run_stage6("outputs_full_logit")