import os
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.mesa_config import MesaABMParams
from src.mesa_abm_model import TwoSidedPlatformABM
from src.utils import ensure_dir, setup_matplotlib

try:
    from experiments.report_common import make_params as make_ode_params
    from experiments.report_common import call_simulate as call_ode_simulate
except Exception:
    make_ode_params = None
    call_ode_simulate = None


STRATEGY_LABELS = {
    "none": "无策略",
    "user_subsidy": "用户补贴",
    "merchant_subsidy": "商户补贴",
    "bilateral_subsidy": "双边补贴",
    "greedy": "贪心策略",
    "long_term": "长期策略",
    "dynamic": "动态策略",
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
    "uniform_subsidy": "统一补贴",
    "random_subsidy": "随机补贴",
    "swing_user": "摇摆用户",
    "key_merchant": "关键商户",
    "bilateral_targeted": "双边精准补贴",
}


# 图 14、15、16、18 等机制展示用：小规模、短期、接近临界初始优势
REPORT_BASIC_ABM_COMMON = {
    "n_users": 400,
    "n_merchants": 20,
    "N_U": 400,
    "N_M": 20,
    "max_steps": 70,
    "n_runs": 30,
    "user_multi_home_prob": 0.0,
    
    "use_report_profit": True,
    "profit_mu": 10.0,
    "discount": 0.98,
}

REPORT_BASIC_INIT = {
    "x0": 0.55,
    "y0": 0.55,
    "active_strategy_platform": "B",
}

# 表 16、表 17 等策略验证用：与报告 ODE 策略验证口径一致
REPORT_POLICY_ABM_COMMON = {
    "n_users": 1000,
    "n_merchants": 50,
    "N_U": 1000,
    "N_M": 50,
    "max_steps": 300,
    "n_runs": 30,
    "user_multi_home_prob": 0.0,
    
}

REPORT_POLICY_INIT = {
    "x0": 0.8,
    "y0": 0.8,
    "active_strategy_platform": "B",
}

REPORT_POLICY_MECHANISM = {
    "rho": 10.0,
    "shortage_rho": 10.0,
    "theta": 1.0,
    "shortage_theta": 1.0,
    "shortage_enabled": True,
    "supply_penalty_enabled": True,
    "lambda_q": 0.05,
    "quality_efficiency": 0.05,
    "invest_eff_u": 0.05,
    "invest_eff_m": 0.05,
    "invest_eff_user": 0.05,
    "invest_eff_merchant": 0.05,
    "quality_decay": 0.01,
    "qmax": 3.0,
    "q_max": 3.0,
    "quality_investment_enabled": True,
    "budget": 0.8,
    "total_budget": 0.8,
    "budget_total": 0.8,
    "dynamic_budget": 0.8,
    "target_share": 0.60,
    "discount": 0.98,
    "profit_mu": 10.0,
    "use_report_profit": True,
    "targeted": True,
    "targeting_enabled": True,
}

ODE_PROFIT_TABLE16 = {
    ("中等", "targeted_long_term"): 178.52,
    ("中等", "targeted_dynamic"): 193.53,
    ("中等", "pure_quality"): 191.66,
    ("强", "targeted_long_term"): 124.48,
    ("强", "targeted_dynamic"): 151.57,
    ("强", "pure_quality"): 152.16,
}


def make_mesa_params(**kwargs) -> MesaABMParams:
    valid_fields = {f.name for f in fields(MesaABMParams)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return MesaABMParams(**filtered)


def make_mesa_params_safe(**kwargs) -> MesaABMParams:
    return make_mesa_params(**kwargs)


def replace_mesa_params_safe(base_params: MesaABMParams, **kwargs) -> MesaABMParams:
    valid_fields = {f.name for f in fields(MesaABMParams)}
    filtered = {k: v for k, v in kwargs.items() if k in valid_fields}
    return replace(base_params, **filtered)


def run_one_abm(params: MesaABMParams) -> pd.DataFrame:
    model = TwoSidedPlatformABM(params)
    model.run_model()
    return pd.DataFrame(model.history)


def run_repeated_abm(base_params: MesaABMParams, strategy: str, n_runs: int) -> pd.DataFrame:
    all_runs = []
    for run_id in range(n_runs):
        params = replace_mesa_params_safe(
            base_params,
            strategy=strategy,
            seed=base_params.seed + run_id,
        )
        df = run_one_abm(params)
        df["run_id"] = run_id
        df["strategy"] = strategy
        all_runs.append(df)
    return pd.concat(all_runs, ignore_index=True)


def _safe_last_value(df: pd.DataFrame, candidates: list[str], default=np.nan) -> float:
    for col in candidates:
        if col in df.columns:
            return float(df[col].iloc[-1])
    return float(default)


def _safe_max_value(df: pd.DataFrame, candidates: list[str], default=np.nan) -> float:
    for col in candidates:
        if col in df.columns:
            return float(df[col].max())
    return float(default)


def _platform_cols(platform: str) -> dict[str, str]:
    platform = platform.upper()
    return {
        "user": f"user_share_{platform}",
        "merchant": f"merchant_share_{platform}",
        "L": f"L_{platform}",
        "profit": f"cum_profit_{platform}",
        "total_cost": f"cum_total_cost_{platform}",
        "shortage": f"shortage_{platform}",
    }


def aggregate_time_series(df: pd.DataFrame, extra_groups: list[str] | None = None) -> pd.DataFrame:
    group_cols = ["strategy", "t"]
    if extra_groups:
        group_cols = extra_groups + group_cols
    numeric_cols = [
        col for col in df.columns
        if col not in group_cols + ["run_id"] and pd.api.types.is_numeric_dtype(df[col])
    ]
    return df.groupby(group_cols)[numeric_cols].mean().reset_index()


def classify_market_state(final_user: float, final_merchant: float) -> str:
    c = abs(final_user - 0.5) + abs(final_merchant - 0.5)
    if c < 0.20:
        return "共存"
    if c < 0.80:
        return "倾斜"
    return "锁定"


def summarize_final(df: pd.DataFrame, platform: str = "B") -> pd.DataFrame:
    c = _platform_cols(platform)
    final_df = df.sort_values("t").groupby(["strategy", "run_id"]).tail(1)
    summary = final_df.groupby("strategy").agg(
        final_user_share_mean=(c["user"], "mean"),
        final_user_share_std=(c["user"], "std"),
        final_merchant_share_mean=(c["merchant"], "mean"),
        final_merchant_share_std=(c["merchant"], "std"),
        final_L_mean=(c["L"], "mean"),
        final_L_std=(c["L"], "std"),
        user_satisfaction_mean=("avg_user_utility", "mean"),
        merchant_satisfaction_mean=("avg_merchant_utility", "mean"),
        user_multi_home_mean=("user_multi_home_rate", "mean"),
        merchant_multi_home_mean=("merchant_multi_home_rate", "mean"),
        lock_in_index_mean=("lock_in_index", "mean"),
        lock_in_index_std=("lock_in_index", "std"),
        cum_profit_mean=(c["profit"], "mean"),
        cum_profit_std=(c["profit"], "std"),
        cum_total_cost_mean=(c["total_cost"], "mean"),
    ).reset_index()
    summary["strategy_label"] = summary["strategy"].map(lambda s: STRATEGY_LABELS.get(s, s))
    return summary


def summarize_report_validation(df: pd.DataFrame, network_name: str) -> pd.DataFrame:
    final_df = df.sort_values("t").groupby(["strategy", "run_id"]).tail(1).copy()
    if "L_B" not in final_df.columns:
        final_df["L_B"] = 0.5 * (final_df["user_share_B"] + final_df["merchant_share_B"])
    rows = []
    for strategy, sub in final_df.groupby("strategy"):
        final_LB = float(sub["L_B"].mean())
        prob_success = float((sub["L_B"] >= 0.6).mean())
        abm_profit = float(sub["cum_profit_B"].mean())
        ode_profit = ODE_PROFIT_TABLE16.get((network_name, strategy), np.nan)
        rows.append({
            "network": network_name,
            "strategy": strategy,
            "strategy_label": STRATEGY_LABELS.get(strategy, strategy),
            "final_LB": final_LB,
            "prob_LB_ge_0_6": prob_success,
            "abm_discounted_profit": abm_profit,
            "ode_discounted_profit": ode_profit,
            "profit_gap": abm_profit - ode_profit if not np.isnan(ode_profit) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_basic_timeseries(df: pd.DataFrame, output_dir: str, filename: str, title: str, platform: str = "B") -> None:
    grouped = aggregate_time_series(df)
    c = _platform_cols(platform)
    strategies = grouped["strategy"].unique()
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()
    plot_items = [
        (c["user"], f"平台 {platform} 用户份额"),
        (c["merchant"], f"平台 {platform} 商户份额"),
        (c["L"], f"平台 {platform} 平均份额"),
        (c["shortage"], f"平台 {platform} 供给不足"),
        (c["profit"], f"平台 {platform} 累计贴现利润"),
        ("lock_in_index", "市场锁定指数"),
    ]
    for ax, (col, ylabel) in zip(axes, plot_items):
        if col not in grouped.columns:
            ax.axis("off")
            continue
        for strategy in strategies:
            sub = grouped[grouped["strategy"] == strategy]
            ax.plot(sub["t"], sub[col], label=STRATEGY_LABELS.get(strategy, strategy), linewidth=2)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("时间")
    axes[-2].set_xlabel("时间")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_final_summary(summary: pd.DataFrame, output_dir: str, filename: str, title: str) -> None:
    labels = summary.get("strategy_label", summary["strategy"]).tolist()
    x = np.arange(len(labels))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(12, 6))
    final_l_col = "final_L_mean" if "final_L_mean" in summary.columns else None
    if final_l_col is None:
        final_l = 0.5 * (summary["final_user_share_mean"] + summary["final_merchant_share_mean"])
    else:
        final_l = summary[final_l_col]
    ax1.bar(x - width, summary["final_user_share_mean"], width, label="用户份额")
    ax1.bar(x, summary["final_merchant_share_mean"], width, label="商户份额")
    ax1.bar(x + width, final_l, width, label="平均份额")
    ax1.axhline(0.6, linestyle="--", linewidth=1.0, label="突破阈值 0.6")
    ax1.set_ylabel("份额")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x, summary["cum_profit_mean"], marker="o", linewidth=2, label="贴现利润")
    ax2.set_ylabel("贴现利润")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def mesa_experiment_0_ode_abm_basic_comparison(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp0_ode_abm_basic_comparison")
    ensure_dir(output_dir)
    cases = [("弱", 0.3, 0.3), ("中等", 0.8, 0.8), ("强", 1.5, 1.5)]
    abm_results = []
    ode_rows = []
    final_rows = []
    for name, alpha, beta in cases:
        print(f"  Mesa 实验0：{name}网络效应 ODE/ABM 对比")
        base_params = make_mesa_params_safe(
            seed=1000,
            **REPORT_BASIC_ABM_COMMON,
            **REPORT_BASIC_INIT,
            alpha=alpha,
            beta=beta,
            shortage_enabled=False,
            supply_penalty_enabled=False,
            strategy="none",
        )
        abm_df = run_repeated_abm(base_params, strategy="none", n_runs=base_params.n_runs)
        abm_df["network"] = name
        abm_results.append(abm_df)
        if make_ode_params is not None and call_ode_simulate is not None:
            ode_params = make_ode_params(alpha=alpha, beta=beta, shortage_enabled=False)
            ode_res = call_ode_simulate(
                REPORT_BASIC_INIT["x0"],
                REPORT_BASIC_INIT["y0"],
                ode_params,
                T=70,
                dt=0.05,
                policy=None,
                method="euler",
            )
            t = np.asarray(ode_res["t"])
            x = np.asarray(ode_res["x"])
            y = np.asarray(ode_res["y"])
            for ti, xi, yi in zip(t, x, y):
                ode_rows.append({
                    "network": name,
                    "t": ti,
                    "user_share_A": xi,
                    "merchant_share_A": yi,
                    "L_A": 0.5 * (xi + yi),
                    "C": abs(xi - 0.5) + abs(yi - 0.5),
                })
            ode_final_L = 0.5 * (x[-1] + y[-1])
        else:
            ode_final_L = np.nan
        final_df = abm_df.sort_values("t").groupby("run_id").tail(1).copy()
        final_rows.append({
            "network": name,
            "abm_final_L_A": final_df["L_A"].mean(),
            "abm_avg_C": (abs(final_df["user_share_A"] - 0.5) + abs(final_df["merchant_share_A"] - 0.5)).mean(),
            "abm_coexist_prob": ((abs(final_df["user_share_A"] - 0.5) + abs(final_df["merchant_share_A"] - 0.5)) < 0.2).mean(),
            "abm_tilt_prob": (((abs(final_df["user_share_A"] - 0.5) + abs(final_df["merchant_share_A"] - 0.5)) >= 0.2) & ((abs(final_df["user_share_A"] - 0.5) + abs(final_df["merchant_share_A"] - 0.5)) < 0.8)).mean(),
            "abm_lock_prob": ((abs(final_df["user_share_A"] - 0.5) + abs(final_df["merchant_share_A"] - 0.5)) >= 0.8).mean(),
            "ode_final_L_A": ode_final_L,
        })
    abm_all = pd.concat(abm_results, ignore_index=True)
    abm_all.to_csv(os.path.join(output_dir, "abm_basic_timeseries.csv"), index=False, encoding="utf-8-sig")
    if ode_rows:
        pd.DataFrame(ode_rows).to_csv(os.path.join(output_dir, "ode_basic_timeseries.csv"), index=False, encoding="utf-8-sig")
    final_summary = pd.DataFrame(final_rows)
    final_summary.to_csv(os.path.join(output_dir, "ode_abm_final_summary.csv"), index=False, encoding="utf-8-sig")
    grouped = abm_all.groupby(["network", "t"])[["L_A", "lock_in_index"]].mean().reset_index()
    plt.figure(figsize=(10, 6))
    for name, _, _ in cases:
        sub = grouped[grouped["network"] == name]
        plt.plot(sub["t"], sub["L_A"], linewidth=2, label=f"ABM-{name}")
    if ode_rows:
        ode_df = pd.DataFrame(ode_rows)
        for name, _, _ in cases:
            sub = ode_df[ode_df["network"] == name]
            plt.plot(sub["t"], sub["L_A"], linestyle="--", linewidth=2, label=f"ODE-{name}")
    plt.xlabel("时间")
    plt.ylabel(r"$L_A$")
    plt.title("Mesa 实验0：ODE 与 ABM 基础动态路径对比")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp0_ode_abm_dynamic_compare.png"), dpi=300, bbox_inches="tight")
    plt.close()
    x = np.arange(len(final_summary))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, final_summary["abm_final_L_A"], width, label="ABM 最终 $L_A$")
    plt.bar(x + width / 2, final_summary["ode_final_L_A"], width, label="ODE 最终 $L_A$")
    plt.xticks(x, final_summary["network"])
    plt.ylim(0, 1.05)
    plt.ylabel(r"最终 $L_A$")
    plt.title("Mesa 实验0：ODE 与 ABM 最终结果对比")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp0_ode_abm_final_compare.png"), dpi=300, bbox_inches="tight")
    plt.close()


def mesa_experiment_1_heterogeneity_strength_scan(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp1_heterogeneity_strength_scan")
    ensure_dir(output_dir)
    sigma_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    all_results = []
    for sigma in sigma_values:
        print(f"  Mesa 实验1：异质性强度 sigma_theta={sigma}")
        base_params = make_mesa_params_safe(
            seed=2026,
            **REPORT_BASIC_ABM_COMMON,
            **REPORT_BASIC_INIT,
            alpha=0.8,
            beta=0.8,
            sigma_theta=sigma,
            sigmaU=sigma,
            sigmaM=sigma,
            preference_sigma=sigma,
            user_preference_sigma=sigma,
            merchant_preference_sigma=sigma,
            shortage_enabled=False,
            supply_penalty_enabled=False,
            strategy="none",
        )
        df = run_repeated_abm(base_params, strategy="none", n_runs=base_params.n_runs)
        df["sigma_theta"] = sigma
        all_results.append(df)
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp1_heterogeneity_timeseries.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for (sigma, run_id), sub in result_df.groupby(["sigma_theta", "run_id"]):
        sub = sub.sort_values("t")
        fu = _safe_last_value(sub, ["user_share_A"])
        fm = _safe_last_value(sub, ["merchant_share_A"])
        C = abs(fu - 0.5) + abs(fm - 0.5)
        rows.append({
            "sigma_theta": sigma,
            "run_id": run_id,
            "final_user_share_A": fu,
            "final_merchant_share_A": fm,
            "final_LA": 0.5 * (fu + fm),
            "concentration_C": C,
            "coexist": int(C < 0.2),
            "tilt": int(0.2 <= C < 0.8),
            "strict_lock": int(C >= 0.8),
        })
    run_summary = pd.DataFrame(rows)
    summary = run_summary.groupby("sigma_theta").agg(
        final_LA_mean=("final_LA", "mean"),
        final_LA_std=("final_LA", "std"),
        concentration_mean=("concentration_C", "mean"),
        concentration_std=("concentration_C", "std"),
        coexist_prob=("coexist", "mean"),
        tilt_prob=("tilt", "mean"),
        strict_lock_prob=("strict_lock", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "mesa_exp1_heterogeneity_summary.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    plt.errorbar(summary["sigma_theta"], summary["final_LA_mean"], yerr=summary["final_LA_std"], marker="o", capsize=4, label=r"最终 $L_A$")
    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.xlabel(r"异质性强度 $\sigma_\theta$")
    plt.ylabel(r"最终 $L_A$")
    plt.title("Mesa 实验1：异质性强度与平均最终份额")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp1_heterogeneity_final_share_lock_prob.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(summary["sigma_theta"], summary["concentration_mean"], marker="o", linewidth=2)
    plt.xlabel(r"异质性强度 $\sigma_\theta$")
    plt.ylabel("平均集中度 C")
    plt.title("Mesa 实验1：异质性强度与市场集中度")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp1_heterogeneity_concentration.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(summary["sigma_theta"], summary["coexist_prob"], marker="o", label="共存概率")
    plt.plot(summary["sigma_theta"], summary["tilt_prob"], marker="s", label="倾斜概率")
    plt.plot(summary["sigma_theta"], summary["strict_lock_prob"], marker="^", label="锁定概率")
    plt.xlabel(r"异质性强度 $\sigma_\theta$")
    plt.ylabel("概率")
    plt.ylim(0, 1.05)
    plt.title("Mesa 实验1：异质性强度与市场状态概率")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp1_heterogeneity_state_prob.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plot_basic_timeseries(result_df, output_dir, "mesa_exp1_heterogeneity_timeseries.png", "Mesa 实验1：异质性强度扫描", platform="A")


def mesa_experiment_2_merchant_multihoming_cost_scan(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp2_multihoming_cost_scan")
    ensure_dir(output_dir)
    k_values = [0.0, 0.2, 0.5, 0.8, 1.2, 1.6, 2.0]
    all_results = []
    for k in k_values:
        print(f"  Mesa 实验2：商户多归属成本 k_M={k}")
        base_params = make_mesa_params_safe(
            seed=3030,
            **REPORT_BASIC_ABM_COMMON,
            **REPORT_BASIC_INIT,
            alpha=0.8,
            beta=0.8,
            k_M=k,
            merchant_multi_home_cost=k,
            multi_home_cost=k,
            multi_home_gap=1.2,
            shortage_enabled=False,
            supply_penalty_enabled=False,
            strategy="none",
        )
        df = run_repeated_abm(base_params, strategy="none", n_runs=base_params.n_runs)
        df["k_M"] = k
        all_results.append(df)
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp2_multihoming_cost_timeseries.csv"), index=False, encoding="utf-8-sig")
    rows = []
    for (k, run_id), sub in result_df.groupby(["k_M", "run_id"]):
        sub = sub.sort_values("t")
        fu = _safe_last_value(sub, ["user_share_A"])
        fm = _safe_last_value(sub, ["merchant_share_A"])
        C = abs(fu - 0.5) + abs(fm - 0.5)
        rows.append({
            "k_M": k,
            "run_id": run_id,
            "final_LA": 0.5 * (fu + fm),
            "merchant_multi_home_rate": _safe_last_value(sub, ["merchant_multi_home_rate"]),
            "concentration_C": C,
            "coexist": int(C < 0.2),
            "tilt": int(0.2 <= C < 0.8),
            "strict_lock": int(C >= 0.8),
        })
    run_summary = pd.DataFrame(rows)
    summary = run_summary.groupby("k_M").agg(
        merchant_multi_home_rate=("merchant_multi_home_rate", "mean"),
        final_LA_mean=("final_LA", "mean"),
        concentration_mean=("concentration_C", "mean"),
        coexist_prob=("coexist", "mean"),
        tilt_prob=("tilt", "mean"),
        strict_lock_prob=("strict_lock", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(output_dir, "mesa_exp2_multihoming_cost_summary.csv"), index=False, encoding="utf-8-sig")
    plt.figure(figsize=(9, 5))
    plt.plot(summary["k_M"], summary["merchant_multi_home_rate"], marker="o", linewidth=2)
    plt.xlabel(r"商户多归属成本 $k_M$")
    plt.ylabel("商户多归属比例")
    plt.title("Mesa 实验2：多归属成本与商户多归属比例")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp2_multihoming_cost_lock_prob.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(summary["k_M"], summary["concentration_mean"], marker="o", linewidth=2)
    plt.xlabel(r"商户多归属成本 $k_M$")
    plt.ylabel("平均集中度 C")
    plt.title("Mesa 实验2：多归属成本与市场集中度")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp2_multihoming_cost_concentration.png"), dpi=300, bbox_inches="tight")
    plt.close()
    plt.figure(figsize=(9, 5))
    plt.plot(summary["k_M"], summary["coexist_prob"], marker="o", label="共存概率")
    plt.plot(summary["k_M"], summary["tilt_prob"], marker="s", label="倾斜概率")
    plt.plot(summary["k_M"], summary["strict_lock_prob"], marker="^", label="锁定概率")
    plt.xlabel(r"商户多归属成本 $k_M$")
    plt.ylabel("概率")
    plt.ylim(0, 1.05)
    plt.title("Mesa 实验2：多归属成本与市场状态概率")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp2_multihoming_market_state_prob.png"), dpi=300, bbox_inches="tight")
    plt.close()


def mesa_experiment_3_long_term_robustness(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp3_long_term_robustness")
    ensure_dir(output_dir)
    base_params = make_mesa_params_safe(
        seed=4040,
        **REPORT_POLICY_ABM_COMMON,
        **REPORT_POLICY_INIT,
        **REPORT_POLICY_MECHANISM,
        alpha=0.8,
        beta=0.8,
    )
    strategies = ["greedy", "long_term", "dynamic", "pure_quality"]
    all_results = []
    for strategy in strategies:
        print(f"  Mesa 实验3：策略 {strategy}")
        df = run_repeated_abm(base_params, strategy=strategy, n_runs=base_params.n_runs)
        all_results.append(df)
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp3_timeseries.csv"), index=False, encoding="utf-8-sig")
    summary = summarize_final(result_df, platform="B")
    summary.to_csv(os.path.join(output_dir, "mesa_exp3_summary.csv"), index=False, encoding="utf-8-sig")
    plot_basic_timeseries(result_df, output_dir, "mesa_exp3_basic_timeseries.png", "Mesa 实验3：长期策略稳健性比较", platform="B")
    plot_final_summary(summary, output_dir, "mesa_exp3_final_summary.png", "Mesa 实验3：贪心、长期、动态与纯质量策略比较")


def mesa_experiment_equal_budget_subsidy_screening(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp_equal_budget_subsidy_screening")
    ensure_dir(output_dir)
    screen_budget = 20.0
    effective_target_count = 30.0
    b_eff = screen_budget / effective_target_count
    cases = [
        ("none", "无补贴", "none", {}),
        ("uniform_subsidy", "统一补贴", "bilateral_subsidy", {
            "bilateral_user_subsidy0": b_eff,
            "bilateral_merchant_subsidy0": b_eff,
        }),
        ("random_subsidy", "随机补贴", "bilateral_subsidy", {
            "bilateral_user_subsidy0": b_eff * 0.9,
            "bilateral_merchant_subsidy0": b_eff * 0.9,
            "user_taste_noise": 0.25,
            "merchant_taste_noise": 0.20,
        }),
        ("swing_user", "摇摆用户", "user_subsidy", {"user_subsidy0": b_eff}),
        ("key_merchant", "关键商户", "merchant_subsidy", {"merchant_subsidy0": b_eff}),
        ("bilateral_targeted", "双边精准补贴", "targeted_dynamic", {
            "budget": 0.8,
            "dynamic_budget": 0.8,
            "targeted": True,
            "targeting_enabled": True,
        }),
    ]
    all_results = []
    rows = []
    for case_key, label, actual_strategy, extra in cases:
        print(f"  Mesa 等预算筛选：{label}")
        base_params = make_mesa_params_safe(
            seed=6060,
            **REPORT_BASIC_ABM_COMMON,
            **REPORT_POLICY_INIT,
            alpha=0.8,
            beta=0.8,
            
            shortage_enabled=True,
            supply_penalty_enabled=True,
            rho=10.0,
            shortage_rho=10.0,
            theta=1.0,
            shortage_theta=1.0,
           
            **extra,
        )
        df = run_repeated_abm(base_params, actual_strategy, n_runs=base_params.n_runs)
        df["screen_strategy"] = case_key
        df["screen_label"] = label
        all_results.append(df)
        final_df = df.sort_values("t").groupby("run_id").tail(1).copy()
        if "L_B" not in final_df.columns:
            final_df["L_B"] = 0.5 * (final_df["user_share_B"] + final_df["merchant_share_B"])
        total_spend = final_df["cum_total_cost_B"].mean()
        profit = final_df["cum_profit_B"].mean()
        rows.append({
            "screen_strategy": case_key,
            "label": label,
            "actual_strategy": actual_strategy,
            "budget_total": screen_budget,
            "effective_subsidy": b_eff,
            "final_LB": final_df["L_B"].mean(),
            "prob_LB_ge_0_6": (final_df["L_B"] >= 0.6).mean(),
            "strict_lock_prob": ((final_df["user_share_B"] >= 0.8) & (final_df["merchant_share_B"] >= 0.8)).mean(),
            "discounted_profit": profit,
            "total_spend": total_spend,
            "ROI": profit / total_spend if total_spend > 0 else np.nan,
        })
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp_equal_budget_subsidy_screening_timeseries.csv"), index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "mesa_exp_equal_budget_subsidy_screening.csv"), index=False, encoding="utf-8-sig")
    x = np.arange(len(summary))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.bar(x - width / 2, summary["final_LB"], width, label=r"最终 $L_B$")
    ax1.axhline(0.6, linestyle="--", linewidth=1.0, label=r"$L_B=0.6$")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel(r"最终 $L_B$")
    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, summary["prob_LB_ge_0_6"], marker="o", linewidth=2, label=r"$P(L_B\geq0.6)$")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("突破概率")
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary["label"], rotation=20, ha="right")
    ax1.set_title("Mesa 等预算补贴策略筛选")
    ax1.grid(axis="y", alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mesa_exp_equal_budget_subsidy_screening.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def mesa_experiment_4_ode_strategy_validation(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp4_ode_strategy_validation")
    ensure_dir(output_dir)
    network_cases = [("中等", 0.8, 0.8), ("强", 1.5, 1.5)]
    strategies = ["targeted_long_term", "targeted_dynamic", "pure_quality"]
    all_summary = []
    all_timeseries = []
    for network_name, alpha, beta in network_cases:
        network_runs = []
        for strategy in strategies:
            print(f"  Mesa 实验4：{network_name}网络效应，策略 {strategy}")
            base_params = make_mesa_params_safe(
                seed=5050,
                **REPORT_POLICY_ABM_COMMON,
                **REPORT_POLICY_INIT,
                **REPORT_POLICY_MECHANISM,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
            )
            df = run_repeated_abm(base_params, strategy=strategy, n_runs=base_params.n_runs)
            df["network"] = network_name
            network_runs.append(df)
            all_timeseries.append(df)
        network_df = pd.concat(network_runs, ignore_index=True)
        all_summary.append(summarize_report_validation(network_df, network_name))
    result_df = pd.concat(all_timeseries, ignore_index=True)
    summary_df = pd.concat(all_summary, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp4_timeseries.csv"), index=False, encoding="utf-8-sig")
    summary_df.to_csv(os.path.join(output_dir, "mesa_exp4_summary_table16.csv"), index=False, encoding="utf-8-sig")
    summary_df["label"] = summary_df.apply(lambda r: f"{r['network']}-{r['strategy_label']}", axis=1)
    x = np.arange(len(summary_df))
    width = 0.35
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, summary_df["abm_discounted_profit"], width, label="ABM 贴现利润")
    plt.bar(x + width / 2, summary_df["ode_discounted_profit"], width, label="ODE 贴现利润")
    plt.xticks(x, summary_df["label"], rotation=20, ha="right")
    plt.ylabel("贴现利润")
    plt.title("Mesa 实验4：ODE 最优策略组合在 ABM 中的表现")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mesa_exp4_ode_abm_profit_compare.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_repeated_abm_ablation(base_params: MesaABMParams, ablation_name: str, actual_strategy: str, n_runs: int) -> pd.DataFrame:
    all_runs = []
    for run_id in range(n_runs):
        params = replace_mesa_params_safe(base_params, strategy=actual_strategy, seed=base_params.seed + run_id)
        df = run_one_abm(params)
        df["run_id"] = run_id
        df["strategy"] = actual_strategy
        df["ablation"] = ablation_name
        all_runs.append(df)
    return pd.concat(all_runs, ignore_index=True)


def summarize_ablation_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ablation, run_id), sub in df.groupby(["ablation", "run_id"]):
        sub = sub.sort_values("t")
        final_user = _safe_last_value(sub, ["user_share_B"])
        final_merchant = _safe_last_value(sub, ["merchant_share_B"])
        final_LB = 0.5 * (final_user + final_merchant)
        profit = _safe_last_value(sub, ["cum_profit_B"])
        total_spend = _safe_last_value(sub, ["cum_total_cost_B"])
        max_cb = _safe_max_value(sub, ["shortage_B", "C_B"])
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
    return run_df.groupby(["ablation", "label"]).agg(
        final_LB=("final_LB", "mean"),
        final_LB_std=("final_LB", "std"),
        prob_LB_ge_0_6=("success_LB_ge_0_6", "mean"),
        strict_lock_prob=("strict_lock", "mean"),
        discounted_profit=("discounted_profit", "mean"),
        discounted_profit_std=("discounted_profit", "std"),
        total_spend=("total_spend", "mean"),
        max_CB=("max_CB", "max"),
    ).reset_index()


def mesa_experiment_5_ablation(output_root: str) -> None:
    output_dir = os.path.join(output_root, "mesa_exp5_ablation")
    ensure_dir(output_dir)
    common_params = dict(
        seed=7070,
        **REPORT_POLICY_ABM_COMMON,
        **REPORT_POLICY_INIT,
        alpha=0.8,
        beta=0.8,
    )
    mechanism_params = dict(**REPORT_POLICY_MECHANISM)
    ablation_cases = [
        ("full_targeted_long_term", "long_term", {"targeted": True, "targeting_enabled": True}),
        ("no_targeting", "long_term", {"targeted": False, "targeting_enabled": False}),
        ("replace_by_dynamic", "dynamic", {}),
        ("replace_by_greedy", "greedy", {}),
        ("no_quality_investment", "long_term", {
            "quality_investment_enabled": False,
            "lambda_q": 0.0,
            "quality_efficiency": 0.0,
            "invest_eff_u": 0.0,
            "invest_eff_m": 0.0,
            "invest_eff_user": 0.0,
            "invest_eff_merchant": 0.0,
        }),
        ("no_supply_penalty", "long_term", {
            "shortage_enabled": False,
            "supply_penalty_enabled": False,
            "theta": 0.0,
            "shortage_theta": 0.0,
        }),
        ("no_heterogeneity", "long_term", {
            "sigma_theta": 0.0,
            "sigmaU": 0.0,
            "sigmaM": 0.0,
            "preference_sigma": 0.0,
            "user_taste_noise": 0.0,
            "merchant_taste_noise": 0.0,
            "user_type_ratios": {"normal": 1.0},
            "merchant_type_ratios": {"small_medium": 1.0},
            "user_type_params": {
                "normal": {"w_q": 1.0, "w_m": 1.0, "w_s": 1.0, "w_p": 1.0, "w_c": 1.0, "inertia": 0.4},
            },
            "merchant_type_params": {
                "small_medium": {"v_r": 1.0, "v_u": 1.0, "v_s": 1.0, "v_c": 1.0, "multi_home_prob": 0.4},
            },
        }),
        ("no_network_effect", "long_term", {"alpha": 0.0, "beta": 0.0}),
    ]
    all_results = []
    for ablation_name, actual_strategy, extra in ablation_cases:
        print(f"  Mesa 实验5：{STRATEGY_LABELS.get(ablation_name, ablation_name)}")
        params_dict = {**common_params, **mechanism_params, **extra}
        base_params = make_mesa_params_safe(**params_dict)
        df = run_repeated_abm_ablation(base_params, ablation_name, actual_strategy, n_runs=base_params.n_runs)
        all_results.append(df)
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_csv(os.path.join(output_dir, "mesa_exp5_ablation_timeseries.csv"), index=False, encoding="utf-8-sig")
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
    summary.to_csv(os.path.join(output_dir, "mesa_exp5_ablation_summary_table17.csv"), index=False, encoding="utf-8-sig")
    labels = summary["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(x - width / 2, summary["final_LB"], width, yerr=summary["final_LB_std"], capsize=4, label=r"最终 $L_B$")
    ax1.axhline(0.6, linestyle="--", linewidth=1.0, label=r"突破阈值 $L_B=0.6$")
    ax1.set_ylabel(r"最终 $L_B$")
    ax1.set_ylim(0, 1.05)
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, summary["max_CB"], marker="o", linewidth=2, label=r"最大 $C_B$")
    ax2.set_ylabel(r"最大供给不足 $C_B$")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Mesa 实验5：消融实验中的最终份额与供给压力")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mesa_exp5_ablation_final_LB_and_CB.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax1 = plt.subplots(figsize=(13, 6))
    ax1.bar(x - width / 2, summary["discounted_profit"], width, yerr=summary["discounted_profit_std"], capsize=4, label="贴现利润")
    ax1.set_ylabel("贴现利润")
    ax1.grid(axis="y", alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(x + width / 2, summary["prob_LB_ge_0_6"], marker="o", linewidth=2, label=r"$P(L_B\geq0.6)$")
    ax2.set_ylabel(r"突破概率 $P(L_B\geq0.6)$")
    ax2.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_title("Mesa 实验5：消融实验中的贴现利润与打破锁定概率")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mesa_exp5_ablation_profit_and_success.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_stage6(output_root: str) -> None:
    setup_matplotlib()
    output_dir = os.path.join(output_root, "stage6_mesa_abm")
    ensure_dir(output_dir)
    print("开始运行 Mesa 实验0：ODE 与 ABM 基础动态对比.")
    mesa_experiment_0_ode_abm_basic_comparison(output_dir)
    print("开始运行 Mesa 实验1：个体异质性强度扫描.")
    mesa_experiment_1_heterogeneity_strength_scan(output_dir)
    print("开始运行 Mesa 实验2：商户多归属成本扫描.")
    mesa_experiment_2_merchant_multihoming_cost_scan(output_dir)
    print("开始运行 Mesa 实验3：长期策略稳健性.")
    mesa_experiment_3_long_term_robustness(output_dir)
    print("开始运行 Mesa 等预算补贴策略筛选实验.")
    mesa_experiment_equal_budget_subsidy_screening(output_dir)
    print("开始运行 Mesa 实验4：ODE 最优策略组合在 ABM 中的表现.")
    mesa_experiment_4_ode_strategy_validation(output_dir)
    print("开始运行 Mesa 实验5：消融实验.")
    mesa_experiment_5_ablation(output_dir)
    print(f"Mesa 阶段全部完成，结果已保存到：{output_dir}")


if __name__ == "__main__":
    run_stage6("outputs_full_logit")