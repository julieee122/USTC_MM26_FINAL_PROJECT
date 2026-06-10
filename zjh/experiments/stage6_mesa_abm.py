import os
from dataclasses import  fields,replace

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

STRATEGY_LABELS.update({
    "targeted_long_term": "针对性长期",
    "targeted_dynamic": "针对性动态",
    "pure_quality": "纯质量投资",

    "full_targeted_long_term": "Full targeted long-term",
    "no_targeting": "w/o targeting",
    "replace_by_dynamic": "replace by dynamic",
    "replace_by_greedy": "replace by greedy",
    "no_quality_investment": "w/o quality investment",
    "no_supply_penalty": "w/o supply penalty",
    "no_heterogeneity": "w/o heterogeneity",
    "no_network_effect": "w/o network effect",
})


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



def make_mesa_params(**kwargs):
    """
    兼容 MesaABMParams 字段。
    只传入当前 MesaABMParams 支持的参数，避免字段名不存在时报错。
    """
    valid_fields = {f.name for f in fields(MesaABMParams)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return MesaABMParams(**filtered)


REPORT_ABM_COMMON = {
    "n_users": 1000,
    "n_merchants": 50,      # 与报告中的 N_U/N_M=20 对齐
    "max_steps": 300,
    "n_runs": 30,
}

REPORT_CHALLENGER_INIT = {
    # 如果 ABM 代码中的平台 A 是“被扶持平台”，则这里的 A 对应报告中的挑战者 B
    "x0": 0.2,
    "y0": 0.2,
}

REPORT_LOCKIN_INIT = {
    # 用于多归属削弱锁定实验：小幅初始优势更容易观察多归属对锁定的削弱
    "x0": 0.55,
    "y0": 0.55,
}

ODE_PROFIT_TABLE16 = {
    ("中等", "targeted_long_term"): 178.52,
    ("中等", "targeted_dynamic"): 193.53,
    ("中等", "pure_quality"): 191.66,
    ("强", "targeted_long_term"): 124.48,
    ("强", "targeted_dynamic"): 151.57,
    ("强", "pure_quality"): 152.16,
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
def summarize_report_validation(df: pd.DataFrame, network_name: str):
    """
    对应报告表 16：
    计算最终 LB、P(LB >= 0.6)、ABM 贴现利润、ODE 贴现利润与差值。

    注意：这里把 ABM 中的平台 A 解释为“被扶持的挑战者平台”，
    因此 L_B = 0.5 * (user_share_A + merchant_share_A)。
    """
    final_df = df.sort_values("t").groupby(["strategy", "run_id"]).tail(1).copy()

    final_df["L_B"] = 0.5 * (
        final_df["user_share_A"] + final_df["merchant_share_A"]
    )

    rows = []

    for strategy, sub in final_df.groupby("strategy"):
        final_LB = sub["L_B"].mean()
        prob_success = (sub["L_B"] >= 0.6).mean()
        abm_profit = sub["cum_profit_A"].mean()

        ode_profit = ODE_PROFIT_TABLE16.get((network_name, strategy), np.nan)
        gap = abm_profit - ode_profit if not np.isnan(ode_profit) else np.nan

        rows.append({
            "network": network_name,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            "final_LB": final_LB,
            "prob_LB_ge_0_6": prob_success,
            "abm_discounted_profit": abm_profit,
            "ode_discounted_profit": ode_profit,
            "profit_gap": gap,
        })

    return pd.DataFrame(rows)

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
# Mesa 报告版公共工具：参数兼容与消融指标
# ============================================================

def make_mesa_params_safe(**kwargs) -> MesaABMParams:
    """
    只传入 MesaABMParams 支持的字段，避免不同版本参数名不一致时报错。
    """
    valid_fields = {f.name for f in fields(MesaABMParams)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return MesaABMParams(**filtered)


def replace_mesa_params_safe(base_params: MesaABMParams, **kwargs) -> MesaABMParams:
    """
    对已有 MesaABMParams 安全 replace。
    未定义字段会自动忽略。
    """
    valid_fields = {f.name for f in fields(MesaABMParams)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return replace(base_params, **filtered)


def _safe_last_value(df: pd.DataFrame, candidates: list[str], default=np.nan):
    """
    从单个 run 的时间序列中读取最后一期指标。
    """
    for col in candidates:
        if col in df.columns:
            return float(df[col].iloc[-1])
    return default


def _safe_max_value(df: pd.DataFrame, candidates: list[str], default=np.nan):
    """
    从单个 run 的时间序列中读取最大值。
    """
    for col in candidates:
        if col in df.columns:
            return float(df[col].max())
    return default


def _estimate_shortage_from_share(df: pd.DataFrame, rho: float = 10.0, eps: float = 1e-6):
    """
    如果 ABM 模型没有直接输出 C_B / shortage_B，
    则用报告中的供给不足公式近似计算：

        C_B = max(0, N_U * u_B / (N_M * m_B + eps) - rho)

    注意：这里默认代码中的平台 A 是“被扶持的挑战者平台”，
    因此 user_share_A、merchant_share_A 对应报告中的 B 份额。
    """
    if "user_share_A" not in df.columns or "merchant_share_A" not in df.columns:
        return np.nan

    u = df["user_share_A"].to_numpy(dtype=float)
    m = df["merchant_share_A"].to_numpy(dtype=float)

    # 如果历史中没有记录 n_users / n_merchants，就按本实验设定使用 1000 / 50
    n_users = 1000
    n_merchants = 50

    shortage = np.maximum(0.0, n_users * u / (n_merchants * m + eps) - rho)
    return float(np.nanmax(shortage))


def run_repeated_abm_ablation(
    base_params: MesaABMParams,
    ablation_name: str,
    actual_strategy: str,
    n_runs: int,
):
    """
    运行某个消融版本。

    ablation_name：报告中的消融版本名称；
    actual_strategy：实际传给 ABM 模型的 strategy。

    这样做是为了避免模型暂时不支持 full_targeted_long_term 等名字。
    例如：
        Full targeted long-term 可以实际用 long_term；
        replace by dynamic 可以实际用 dynamic。
    """
    all_runs = []

    for run_id in range(n_runs):
        params = replace_mesa_params_safe(
            base_params,
            strategy=actual_strategy,
            seed=base_params.seed + run_id,
        )

        df = run_one_abm(params)
        df["run_id"] = run_id
        df["strategy"] = actual_strategy
        df["ablation"] = ablation_name
        all_runs.append(df)

    return pd.concat(all_runs, ignore_index=True)


def summarize_ablation_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    对应报告表 17：
    输出最终 LB、P(LB>=0.6)、严格锁定概率、贴现利润、总支出、最大 CB。
    """
    rows = []

    for (ablation, run_id), sub in df.groupby(["ablation", "run_id"]):
        sub = sub.sort_values("t")

        final_user = _safe_last_value(
            sub,
            ["user_share_A", "final_user_share_A"],
            default=np.nan,
        )
        final_merchant = _safe_last_value(
            sub,
            ["merchant_share_A", "final_merchant_share_A"],
            default=np.nan,
        )

        final_LB = 0.5 * (final_user + final_merchant)

        profit = _safe_last_value(
            sub,
            [
                "discounted_profit_A",
                "cum_discounted_profit_A",
                "cum_profit_A",
                "profit_A",
            ],
            default=np.nan,
        )

        total_spend = _safe_last_value(
            sub,
            [
                "cum_total_cost_A",
                "cum_subsidy_cost_A",
                "cum_cost_A",
                "total_spend_A",
                "total_cost_A",
            ],
            default=np.nan,
        )

        max_cb = _safe_max_value(
            sub,
            [
                "shortage_B",
                "shortage_A",
                "C_B",
                "C_A",
                "max_shortage_B",
                "max_shortage_A",
            ],
            default=np.nan,
        )

        if np.isnan(max_cb):
            max_cb = _estimate_shortage_from_share(sub)

        rows.append({
            "ablation": ablation,
            "label": STRATEGY_LABELS.get(ablation, ablation),
            "run_id": run_id,
            "final_user_share": final_user,
            "final_merchant_share": final_merchant,
            "final_LB": final_LB,
            "success_LB_ge_0_6": int(final_LB >= 0.6),
            "strict_lock": int(final_user >= 0.8 and final_merchant >= 0.8),
            "discounted_profit": profit,
            "total_spend": total_spend,
            "max_CB": max_cb,
        })

    run_df = pd.DataFrame(rows)

    summary = run_df.groupby(["ablation", "label"]).agg(
        final_LB=("final_LB", "mean"),
        final_LB_std=("final_LB", "std"),
        prob_LB_ge_0_6=("success_LB_ge_0_6", "mean"),
        strict_lock_prob=("strict_lock", "mean"),
        discounted_profit=("discounted_profit", "mean"),
        discounted_profit_std=("discounted_profit", "std"),
        total_spend=("total_spend", "mean"),
        max_CB=("max_CB", "max"),
    ).reset_index()

    return summary
# ============================================================
# Mesa 实验 1：个体异质性强度扫描
# ============================================================

def mesa_experiment_1_heterogeneity_strength_scan(output_root: str):
    """
    对应报告 6.3 实验 1：个体异质性强度扫描。

    报告设定：
    - u_A(0)=m_A(0)=0.8
    - alpha=beta=0.8
    - sigma_theta in {0,0.25,0.5,0.75,1.0,1.25,1.5,2.0}

    目的：
    检验微观个体异质性增强后，市场锁定结果是否更不稳定。
    """
    output_dir = os.path.join(output_root, "mesa_exp1_heterogeneity_strength_scan")
    ensure_dir(output_dir)

    sigma_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    all_results = []

    for sigma in sigma_values:
        print(f"  Mesa 实验1：正在运行异质性强度 sigma_theta={sigma}")

        base_params = make_mesa_params_safe(
            seed=2026,
            n_users=1000,
            n_merchants=50,
            max_steps=300,
            n_runs=30,

            # 报告 6.3 统一初始条件：平台 A 初始占优
            x0=0.8,
            y0=0.8,

            alpha=0.8,
            beta=0.8,

            # 不同版本参数类可能字段名不同，全部传入，safe 函数会自动过滤不存在的字段
            sigma_theta=sigma,
            sigmaU=sigma,
            sigmaM=sigma,
            preference_sigma=sigma,
            user_preference_sigma=sigma,
            merchant_preference_sigma=sigma,

            strategy="none",
        )

        df = run_repeated_abm(
            base_params=base_params,
            strategy="none",
            n_runs=base_params.n_runs,
        )

        df["sigma_theta"] = sigma
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp1_heterogeneity_timeseries.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 汇总指标
    # ============================================================

    final_rows = []

    for (sigma, run_id), sub in result_df.groupby(["sigma_theta", "run_id"]):
        sub = sub.sort_values("t")

        final_user = _safe_last_value(sub, ["user_share_A"], default=np.nan)
        final_merchant = _safe_last_value(sub, ["merchant_share_A"], default=np.nan)

        final_LA = 0.5 * (final_user + final_merchant)
        concentration_C = abs(final_user - 0.5) + abs(final_merchant - 0.5)

        lock_index = _safe_last_value(sub, ["lock_in_index"], default=concentration_C)
        profit = _safe_last_value(sub, ["cum_profit_A"], default=np.nan)

        final_rows.append({
            "sigma_theta": sigma,
            "run_id": run_id,
            "final_user_share_A": final_user,
            "final_merchant_share_A": final_merchant,
            "final_LA": final_LA,
            "concentration_C": concentration_C,
            "strict_lock": int(concentration_C >= 0.8),
            "coexist": int(concentration_C < 0.2),
            "tilt": int(0.2 <= concentration_C < 0.8),
            "lock_in_index": lock_index,
            "cum_profit_A": profit,
        })

    final_df = pd.DataFrame(final_rows)

    summary = final_df.groupby("sigma_theta").agg(
        final_LA_mean=("final_LA", "mean"),
        final_LA_std=("final_LA", "std"),
        concentration_C_mean=("concentration_C", "mean"),
        concentration_C_std=("concentration_C", "std"),
        strict_lock_prob=("strict_lock", "mean"),
        coexist_prob=("coexist", "mean"),
        tilt_prob=("tilt", "mean"),
        cum_profit_mean=("cum_profit_A", "mean"),
        cum_profit_std=("cum_profit_A", "std"),
    ).reset_index()

    summary.to_csv(
        os.path.join(output_dir, "mesa_exp1_heterogeneity_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 图 1：异质性强度与最终份额、锁定概率
    # ============================================================

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(
        summary["sigma_theta"],
        summary["final_LA_mean"],
        yerr=summary["final_LA_std"],
        marker="o",
        capsize=4,
        label=r"最终平均份额 $L_A$",
    )

    ax1.set_xlabel(r"异质性强度 $\sigma_\theta$")
    ax1.set_ylabel(r"最终平均份额 $L_A$")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        summary["sigma_theta"],
        summary["strict_lock_prob"],
        marker="s",
        linestyle="--",
        label="严格锁定概率",
    )
    ax2.set_ylabel("严格锁定概率")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Mesa 实验1：个体异质性强度对市场锁定的影响")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mesa_exp1_heterogeneity_final_share_lock_prob.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ============================================================
    # 图 2：异质性强度与市场集中度 C
    # ============================================================

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        summary["sigma_theta"],
        summary["concentration_C_mean"],
        yerr=summary["concentration_C_std"],
        marker="o",
        capsize=4,
        label=r"市场集中度 $C$",
    )

    plt.axhline(0.2, linestyle="--", linewidth=1.0, label="共存/倾斜阈值 C=0.2")
    plt.axhline(0.8, linestyle="--", linewidth=1.0, label="倾斜/锁定阈值 C=0.8")

    plt.xlabel(r"异质性强度 $\sigma_\theta$")
    plt.ylabel(r"市场集中度 $C$")
    plt.title("Mesa 实验1：异质性强度与市场集中度")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "mesa_exp1_heterogeneity_concentration.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ============================================================
    # 图 3：部分 sigma 下的 LA 动态
    # ============================================================

    result_df["L_A"] = 0.5 * (
        result_df["user_share_A"] + result_df["merchant_share_A"]
    )

    grouped = result_df.groupby(["sigma_theta", "t"])["L_A"].mean().reset_index()

    plt.figure(figsize=(10, 6))

    for sigma in sigma_values:
        sub = grouped[grouped["sigma_theta"] == sigma]
        plt.plot(sub["t"], sub["L_A"], label=rf"$\sigma_\theta={sigma}$")

    plt.xlabel("时间")
    plt.ylabel(r"平均份额 $L_A(t)$")
    plt.title("Mesa 实验1：不同异质性强度下的平均份额动态")
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "mesa_exp1_heterogeneity_timeseries.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# Mesa 实验 2：商户多归属成本对市场锁定的影响
# ============================================================

def mesa_experiment_2_merchant_multihoming_cost_scan(output_root: str):
    """
    对应报告 6.3 实验 2：
    商户多归属成本 k_M 扫描。

    报告扫描：
    k_M = 0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0

    注意：
    原代码扫描的是 merchant_multi_home_override，即“强制多归属概率”；
    报告中需要扫描的是“多归属成本/门槛”，因此这里不再使用 override。
    """
    output_dir = os.path.join(output_root, "mesa_exp2_merchant_multihoming_cost")
    ensure_dir(output_dir)

    k_values = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0]

    all_results = []

    for k_m in k_values:
        print(f"  Mesa 实验2：正在运行商户多归属成本 k_M={k_m}")

        base_params = make_mesa_params_safe(
            seed=3030,
            n_users=1000,
            n_merchants=50,
            max_steps=300,
            n_runs=30,

            # 报告 6.3 统一初始条件
            x0=0.8,
            y0=0.8,

            alpha=0.8,
            beta=0.8,

            # 异质性强度固定为报告中的 0.8
            sigma_theta=0.8,
            sigmaU=0.8,
            sigmaM=0.8,
            preference_sigma=0.8,

            # 多归属成本/门槛的不同可能字段名
            k_M=k_m,
            multi_home_cost=k_m,
            merchant_multi_home_cost=k_m,
            

            # 不使用强制多归属概率
            # merchant_multi_home_override 不再传入

            strategy="none",
        )

        df = run_repeated_abm(
            base_params=base_params,
            strategy="none",
            n_runs=base_params.n_runs,
        )

        df["multi_home_cost_kM"] = k_m
        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp2_multihoming_cost_timeseries.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 汇总指标
    # ============================================================

    final_rows = []

    for (k_m, run_id), sub in result_df.groupby(["multi_home_cost_kM", "run_id"]):
        sub = sub.sort_values("t")

        final_user = _safe_last_value(sub, ["user_share_A"], default=np.nan)
        final_merchant = _safe_last_value(sub, ["merchant_share_A"], default=np.nan)

        final_LA = 0.5 * (final_user + final_merchant)
        concentration_C = abs(final_user - 0.5) + abs(final_merchant - 0.5)

        multi_home_rate = _safe_last_value(
            sub,
            ["merchant_multi_home_rate", "multi_home_rate", "merchant_multihome_rate"],
            default=np.nan,
        )

        lock_index = _safe_last_value(sub, ["lock_in_index"], default=concentration_C)

        final_rows.append({
            "multi_home_cost_kM": k_m,
            "run_id": run_id,
            "final_user_share_A": final_user,
            "final_merchant_share_A": final_merchant,
            "final_LA": final_LA,
            "concentration_C": concentration_C,
            "coexist": int(concentration_C < 0.2),
            "tilt": int(0.2 <= concentration_C < 0.8),
            "strict_lock": int(concentration_C >= 0.8),
            "merchant_multi_home_rate": multi_home_rate,
            "lock_in_index": lock_index,
        })

    final_df = pd.DataFrame(final_rows)

    summary = final_df.groupby("multi_home_cost_kM").agg(
        merchant_multi_home_rate_mean=("merchant_multi_home_rate", "mean"),
        merchant_multi_home_rate_std=("merchant_multi_home_rate", "std"),
        final_LA_mean=("final_LA", "mean"),
        final_LA_std=("final_LA", "std"),
        concentration_C_mean=("concentration_C", "mean"),
        concentration_C_std=("concentration_C", "std"),
        coexist_prob=("coexist", "mean"),
        tilt_prob=("tilt", "mean"),
        strict_lock_prob=("strict_lock", "mean"),
        lock_in_index_mean=("lock_in_index", "mean"),
        lock_in_index_std=("lock_in_index", "std"),
    ).reset_index()

    summary.to_csv(
        os.path.join(output_dir, "mesa_exp2_multihoming_cost_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 图 1：k_M 与多归属比例、锁定概率
    # ============================================================

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.errorbar(
        summary["multi_home_cost_kM"],
        summary["merchant_multi_home_rate_mean"],
        yerr=summary["merchant_multi_home_rate_std"],
        marker="o",
        capsize=4,
        label="商户多归属比例",
    )

    ax1.set_xlabel(r"商户多归属成本 $k_M$")
    ax1.set_ylabel("商户多归属比例")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        summary["multi_home_cost_kM"],
        summary["strict_lock_prob"],
        marker="s",
        linestyle="--",
        label="严格锁定概率",
    )
    ax2.set_ylabel("严格锁定概率")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(r"Mesa 实验2：多归属成本 $k_M$ 对商户多归属与锁定概率的影响")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mesa_exp2_multihoming_cost_lock_prob.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ============================================================
    # 图 2：k_M 与市场集中度 C
    # ============================================================

    plt.figure(figsize=(10, 6))

    plt.errorbar(
        summary["multi_home_cost_kM"],
        summary["concentration_C_mean"],
        yerr=summary["concentration_C_std"],
        marker="o",
        capsize=4,
        label=r"市场集中度 $C$",
    )

    plt.axhline(0.2, linestyle="--", linewidth=1.0, label="共存/倾斜阈值 C=0.2")
    plt.axhline(0.8, linestyle="--", linewidth=1.0, label="倾斜/锁定阈值 C=0.8")

    plt.xlabel(r"商户多归属成本 $k_M$")
    plt.ylabel(r"市场集中度 $C$")
    plt.title(r"Mesa 实验2：多归属成本 $k_M$ 与市场集中度")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "mesa_exp2_multihoming_cost_concentration.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ============================================================
    # 图 3：共存/倾斜/锁定概率
    # ============================================================

    x = np.arange(len(summary))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(
        x - width,
        summary["coexist_prob"],
        width,
        label="共存概率",
    )

    plt.bar(
        x,
        summary["tilt_prob"],
        width,
        label="倾斜概率",
    )

    plt.bar(
        x + width,
        summary["strict_lock_prob"],
        width,
        label="锁定概率",
    )

    plt.xticks(
        x,
        [f"{v:.1f}" for v in summary["multi_home_cost_kM"]],
    )
    plt.xlabel(r"商户多归属成本 $k_M$")
    plt.ylabel("概率")
    plt.ylim(0, 1.05)
    plt.title("Mesa 实验2：不同多归属成本下的市场状态概率")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "mesa_exp2_multihoming_market_state_prob.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# Mesa 实验 3：长期策略稳健性
# ============================================================

def mesa_experiment_3_long_term_robustness(output_root: str):
    output_dir = os.path.join(output_root, "mesa_exp3_long_term_robustness")
    ensure_dir(output_dir)

    base_params = make_mesa_params(
        seed=4040,
        **REPORT_ABM_COMMON,
        **REPORT_CHALLENGER_INIT,
        alpha=0.8,
        beta=0.8,
    )

    strategies = [
        "greedy",
        "long_term",
        "dynamic",
        "pure_quality",
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

def mesa_experiment_4_ode_strategy_validation(output_root: str):
    """
    对应报告图 17 和表 16：
    ODE 最优策略组合在 ABM 中的表现。

    比较：
    - 针对性长期
    - 针对性动态
    - 纯质量投资

    输出：
    - final LB
    - P(LB >= 0.6)
    - ABM 贴现利润
    - ODE 贴现利润
    - 差值
    """
    output_dir = os.path.join(output_root, "mesa_exp4_ode_strategy_validation")
    ensure_dir(output_dir)

    network_cases = [
        ("中等", 0.8, 0.8),
        ("强", 1.5, 1.5),
    ]

    strategies = [
        "targeted_long_term",
        "targeted_dynamic",
        "pure_quality",
    ]

    all_summary = []
    all_timeseries = []

    for network_name, alpha, beta in network_cases:
        for strategy in strategies:
            print(f"  Mesa 实验4：{network_name}网络效应，策略 {strategy}")

            base_params = make_mesa_params(
                seed=5050,
                **REPORT_ABM_COMMON,
                **REPORT_CHALLENGER_INIT,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
            )

            df = run_repeated_abm(
                base_params=base_params,
                strategy=strategy,
                n_runs=base_params.n_runs,
            )

            df["network"] = network_name
            all_timeseries.append(df)

        network_df = pd.concat(
            [d for d in all_timeseries if d["network"].iloc[0] == network_name],
            ignore_index=True,
        )

        summary = summarize_report_validation(network_df, network_name)
        all_summary.append(summary)

    result_df = pd.concat(all_timeseries, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp4_timeseries.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(
        os.path.join(output_dir, "mesa_exp4_summary_table16.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 图：ABM 与 ODE 贴现利润对比
    summary_df["label"] = summary_df.apply(
        lambda r: f"{r['network']}-{r['strategy_label']}",
        axis=1,
    )

    x = np.arange(len(summary_df))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        summary_df["abm_discounted_profit"],
        width,
        label="ABM 贴现利润",
    )

    plt.bar(
        x + width / 2,
        summary_df["ode_discounted_profit"],
        width,
        label="ODE 贴现利润",
    )

    plt.xticks(x, summary_df["label"], rotation=20, ha="right")
    plt.ylabel("贴现利润")
    plt.title("Mesa 实验4：ODE 最优策略组合在 ABM 中的贴现利润表现")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_dir, "mesa_exp4_ode_abm_profit_compare.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

# ============================================================
# Mesa 实验5：机制消融实验
# ============================================================

def mesa_experiment_5_ablation(output_root: str):
    """
    对应报告中的 Mesa 消融实验。

    消融目标：
    在 ABM 微观仿真环境下，检验完整长期混合策略中各个机制的重要性。

    消融版本包括：
    1. 完整针对性长期策略；
    2. 去掉精准补贴；
    3. 替换为动态策略；
    4. 替换为贪心策略；
    5. 去掉质量投资；
    6. 去掉供给惩罚；
    7. 去掉异质性；
    8. 去掉网络效应。

    输出：
    - mesa_exp5_ablation_timeseries.csv
    - mesa_exp5_ablation_summary_table17.csv
    - mesa_exp5_ablation_final_LB_and_CB.png
    - mesa_exp5_ablation_profit_and_success.png

    说明：
    这里把代码中的平台 A 解释为报告中的挑战者平台 B，
    因此 x0=y0=0.2 表示挑战者初始份额为 0.2。
    """
    output_dir = os.path.join(output_root, "mesa_exp5_ablation")
    ensure_dir(output_dir)

    # ============================================================
    # 1. 基础参数：尽量与报告 ABM 部分保持一致
    # ============================================================

    common_params = dict(
        seed=7070,
        n_users=1000,
        n_merchants=50,
        max_steps=300,
        n_runs=30,

        # 代码中的平台 A = 报告中的挑战者 B
        x0=0.2,
        y0=0.2,

        # 中等网络效应
        alpha=0.8,
        beta=0.8,
    )

    # 这些字段并不一定全部存在于 MesaABMParams 中；
    # make_mesa_params_safe 会自动过滤不存在的字段。
    mechanism_params = dict(
        # 异质性强度
        sigma_theta=0.8,
        sigmaU=0.8,
        sigmaM=0.8,
        preference_sigma=0.8,

        # 供给不足参数
        rho=10.0,
        shortage_rho=10.0,
        theta=1.0,
        shortage_theta=1.0,
        shortage_enabled=True,
        supply_penalty_enabled=True,

        # 质量投资参数
        lambda_q=0.05,
        quality_efficiency=0.05,
        invest_eff_u=0.05,
        invest_eff_m=0.05,
        quality_decay=0.01,
        qmax=3.0,
        q_max=3.0,
        quality_investment_enabled=True,

        # 预算参数
        budget=20.0,
        total_budget=20.0,
        budget_total=20.0,

        # 精准补贴开关
        targeted=True,
        targeting_enabled=True,
    )

    # ============================================================
    # 2. 消融实验设置
    # ============================================================
    # ablation：报告中展示的消融版本名称；
    # actual_strategy：实际传给 ABM 模型的策略名称。
    #
    # 如果你的 ABM 模型暂时不支持 full_targeted_long_term 这种名字，
    # 就让它实际使用已有的 long_term / dynamic / greedy。
    # 具体机制差异通过参数开关体现。
    # ============================================================

    ablation_cases = [
        {
            "ablation": "full_targeted_long_term",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "targeted": True,
                "targeting_enabled": True,
                "quality_investment_enabled": True,
                "shortage_enabled": True,
                "supply_penalty_enabled": True,
            },
        },
        {
            "ablation": "no_targeting",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "targeted": False,
                "targeting_enabled": False,
            },
        },
        {
            "ablation": "replace_by_dynamic",
            "actual_strategy": "dynamic",
            "params": {
                **common_params,
                **mechanism_params,
            },
        },
        {
            "ablation": "replace_by_greedy",
            "actual_strategy": "greedy",
            "params": {
                **common_params,
                **mechanism_params,
            },
        },
        {
            "ablation": "no_quality_investment",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "quality_investment_enabled": False,
                "lambda_q": 0.0,
                "quality_efficiency": 0.0,
                "invest_eff_u": 0.0,
                "invest_eff_m": 0.0,
            },
        },
        {
            "ablation": "no_supply_penalty",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "shortage_enabled": False,
                "supply_penalty_enabled": False,
                "theta": 0.0,
                "shortage_theta": 0.0,
            },
        },
        {
            "ablation": "no_heterogeneity",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "sigma_theta": 0.0,
                "sigmaU": 0.0,
                "sigmaM": 0.0,
                "preference_sigma": 0.0,
                "user_taste_noise": 0.0,
                "merchant_taste_noise": 0.0,
                "user_type_ratios": {
                "normal": 1.0,
            },
            "merchant_type_ratios": {
                "small_medium": 1.0,
           },
            "user_type_params": {
                "normal": {
                    "w_q": 1.0,
                    "w_m": 1.0,
                    "w_s": 1.0,
                    "w_p": 1.0,
                    "w_c": 1.0,
                    "inertia": 0.4,
                },
            },
            "merchant_type_params": {
                "small_medium": {
                    "v_r": 1.0,
                    "v_u": 1.0,
                    "v_s": 1.0,
                    "v_c": 1.0,
                    "multi_home_prob": 0.4,
                },
            },
        },
    },

        {
            "ablation": "no_network_effect",
            "actual_strategy": "long_term",
            "params": {
                **common_params,
                **mechanism_params,
                "alpha": 0.0,
                "beta": 0.0,
            },
        },
    ]

    # ============================================================
    # 3. 逐个消融版本运行 ABM
    # ============================================================

    all_results = []

    for case in ablation_cases:
        ablation_name = case["ablation"]
        actual_strategy = case["actual_strategy"]
        params_dict = case["params"]

        print(
            f"  Mesa 实验5：正在运行 {STRATEGY_LABELS.get(ablation_name, ablation_name)}"
        )

        base_params = make_mesa_params_safe(**params_dict)

        df = run_repeated_abm_ablation(
            base_params=base_params,
            ablation_name=ablation_name,
            actual_strategy=actual_strategy,
            n_runs=base_params.n_runs,
        )

        all_results.append(df)

    result_df = pd.concat(all_results, ignore_index=True)

    result_df.to_csv(
        os.path.join(output_dir, "mesa_exp5_ablation_timeseries.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 4. 汇总表：对应报告表 17
    # ============================================================

    summary = summarize_ablation_results(result_df)

    order = [
        "full_targeted_long_term",
        "no_targeting",
        "replace_by_dynamic",
        "replace_by_greedy",
        "no_quality_investment",
        "no_supply_penalty",
        "no_heterogeneity",
        "no_network_effect",
    ]

    order_map = {name: i for i, name in enumerate(order)}
    summary["order"] = summary["ablation"].map(order_map)
    summary = summary.sort_values("order").drop(columns=["order"])

    summary.to_csv(
        os.path.join(output_dir, "mesa_exp5_ablation_summary_table17.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # ============================================================
    # 5. 图 1：最终 LB 与最大 CB
    # ============================================================

    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.bar(
        x - width / 2,
        summary["final_LB"],
        width,
        yerr=summary["final_LB_std"],
        capsize=4,
        label=r"最终 $L_B$",
    )

    ax1.axhline(
        0.6,
        linestyle="--",
        linewidth=1.0,
        label=r"突破阈值 $L_B=0.6$",
    )

    ax1.set_ylabel(r"最终 $L_B$")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x + width / 2,
        summary["max_CB"],
        marker="o",
        linewidth=2,
        label=r"最大 $C_B$",
    )
    ax2.set_ylabel(r"最大供给不足 $C_B$")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Mesa 实验5：消融实验中的最终份额与供给压力")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mesa_exp5_ablation_final_LB_and_CB.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # ============================================================
    # 6. 图 2：贴现利润与突破概率
    # ============================================================

    fig, ax1 = plt.subplots(figsize=(13, 6))

    ax1.bar(
        x - width / 2,
        summary["discounted_profit"],
        width,
        yerr=summary["discounted_profit_std"],
        capsize=4,
        label="贴现利润",
    )

    ax1.set_ylabel("贴现利润")
    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        x + width / 2,
        summary["prob_LB_ge_0_6"],
        marker="o",
        linewidth=2,
        label=r"$P(L_B\geq 0.6)$",
    )

    ax2.set_ylabel(r"突破概率 $P(L_B\geq 0.6)$")
    ax2.set_ylim(0, 1.05)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Mesa 实验5：消融实验中的贴现利润与打破锁定概率")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "mesa_exp5_ablation_profit_and_success.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"  Mesa 实验5完成，结果已保存到：{output_dir}")
# ============================================================
# 阶段 6 总入口
# ============================================================

def run_stage6(output_root: str):
    setup_matplotlib()

    output_dir = os.path.join(output_root, "stage6_mesa_abm")
    ensure_dir(output_dir)

    print("开始运行 Mesa 实验1：个体异质性强度扫描.")
    mesa_experiment_1_heterogeneity_strength_scan(output_dir)

    print("开始运行 Mesa 实验2：商户多归属成本扫描.")
    mesa_experiment_2_merchant_multihoming_cost_scan(output_dir)

    print("开始运行 Mesa 实验3：长期策略稳健性...")
    mesa_experiment_3_long_term_robustness(output_dir)

    print("开始运行 Mesa 实验4：ODE 最优策略组合在 ABM 中的表现.")
    mesa_experiment_4_ode_strategy_validation(output_dir)

    
    print("开始运行 Mesa 实验5：消融实验.")
    mesa_experiment_5_ablation(output_dir)


    print(f"Mesa 阶段全部完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    run_stage6("outputs_full_logit")