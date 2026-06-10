"""Experiment runners for stages 2-4: subsidies, congestion, and quality investment."""

from copy import deepcopy
from pathlib import Path

from .config import (
    BASE_CONFIG,
    CONGESTION_THETA_LEVELS,
    STAGE234_NETWORK_LEVELS,
    SUBSIDY_BUDGETS,
)
from .experiments import first_time_b_overtakes, summarize_run, write_csv, write_json
from .metrics import discounted_total
from .metrics import hhi, lock_index, platform_profit
from .model_base import euler_step, run_simulation
from .strategies import (
    balanced_subsidy_strategy,
    dynamic_strategy,
    greedy_strategy,
    long_term_strategy,
    make_action,
    merchant_leaning_subsidy_strategy,
    merchant_subsidy_strategy,
    no_strategy,
    quality_only_strategy,
    user_leaning_subsidy_strategy,
    user_subsidy_strategy,
)


SUBSIDY_STRATEGIES = [
    ("none", no_strategy),
    ("user_only", user_subsidy_strategy),
    ("merchant_only", merchant_subsidy_strategy),
    ("balanced", balanced_subsidy_strategy),
    ("user_leaning", user_leaning_subsidy_strategy),
    ("merchant_leaning", merchant_leaning_subsidy_strategy),
]

CONGESTION_STRATEGIES = [
    ("user_only", user_subsidy_strategy),
    ("merchant_only", merchant_subsidy_strategy),
    ("balanced", balanced_subsidy_strategy),
    ("merchant_leaning", merchant_leaning_subsidy_strategy),
]

QUALITY_INVESTMENT_STRATEGIES = [
    ("greedy", greedy_strategy),
    ("long_term", long_term_strategy),
    ("dynamic", dynamic_strategy),
    ("quality_only", quality_only_strategy),
]

QUALITY_DELAY_BUDGETS = [0.1, 0.2, 0.8]
QUALITY_BUDGET_SCAN = [round(idx * 0.1, 1) for idx in range(0, 16)]


def _sum_records(records, key):
    return sum(row.get(key, 0.0) for row in records[1:])


def _mean_records(records, key):
    values = [row.get(key, 0.0) for row in records[1:]]
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_policy_run(records):
    summary = summarize_run(records)
    initial = records[0]
    final = records[-1]
    total_user_subsidy = _sum_records(records, "s_B_user")
    total_merchant_subsidy = _sum_records(records, "s_B_merchant")
    total_quality_investment = _sum_records(records, "I_B")
    total_spend = total_user_subsidy + total_merchant_subsidy + total_quality_investment
    share_gain = (
        summary["B_user_T"]
        + summary["B_merchant_T"]
        - initial["u_B_user"]
        - initial["m_B_merchant"]
    )
    profit_values = [row.get("profit_B", 0.0) for row in records[1:]]

    summary.update(
        {
            "B_user_gain": summary["B_user_T"] - initial["u_B_user"],
            "B_merchant_gain": summary["B_merchant_T"] - initial["m_B_merchant"],
            "combined_share_gain": share_gain,
            "total_user_subsidy": total_user_subsidy,
            "total_merchant_subsidy": total_merchant_subsidy,
            "total_quality_investment": total_quality_investment,
            "total_spend": total_spend,
            "share_gain_per_spend": share_gain / total_spend if total_spend > 0 else 0.0,
            "discounted_profit_B": discounted_total(profit_values, discount=0.98),
            "min_profit_B": min(profit_values) if profit_values else 0.0,
            "avg_C_B": _mean_records(records, "C_B"),
            "max_C_B": max((row.get("C_B", 0.0) for row in records[1:]), default=0.0),
            "avg_Uu_B": _mean_records(records, "Uu_B"),
            "q_B_T": final.get("q_B", 0.0),
            "r_B_T": final.get("r_B", 0.0),
            "B_user_overtake_time": first_time_b_overtakes(records, both_sides=False),
            "B_both_overtake_time": first_time_b_overtakes(records, both_sides=True),
        }
    )
    return summary


def _base_challenger_config(base_config=None, network=None, T=None):
    params = deepcopy(base_config or BASE_CONFIG)
    if network:
        params["alpha"] = network["alpha"]
        params["beta"] = network["beta"]
    if T is not None:
        params["T"] = T
    params["q_A"] = 1.0
    params["q_B"] = 1.0
    params["r_A"] = 1.0
    params["r_B"] = 1.0
    params["p_A"] = 0.0
    params["p_B"] = 0.0
    params["c_A"] = 0.0
    params["c_B"] = 0.0
    return params


def run_exit_subsidy_simulation(
    params,
    total_budget,
    exit_step,
    x0=0.80,
    y0=0.80,
):
    """Run a balanced subsidy that exits after ``exit_step`` steps."""
    params = deepcopy(params)
    x = x0
    y = y0
    records = [
        {
            "step": 0,
            "time": 0.0,
            "x_A_user": x,
            "y_A_merchant": y,
            "u_B_user": 1.0 - x,
            "m_B_merchant": 1.0 - y,
            "lock_index": lock_index(x, y),
            "hhi": hhi(x, y),
            "q_A": params["q_A"],
            "q_B": params["q_B"],
            "r_A": params["r_A"],
            "r_B": params["r_B"],
        }
    ]

    for step in range(1, params["T"] + 1):
        action_A = no_strategy()
        if step <= exit_step:
            action_B = make_action(
                user_subsidy=0.5 * total_budget,
                merchant_subsidy=0.5 * total_budget,
            )
        else:
            action_B = no_strategy()

        x, y, info = euler_step(x, y, params, action_A, action_B)
        profit_A = platform_profit(x, y, action_A, params["mu"])
        profit_B = platform_profit(1.0 - x, 1.0 - y, action_B, params["mu"])
        records.append(
            {
                "step": step,
                "time": step * params["dt"],
                "x_A_user": x,
                "y_A_merchant": y,
                "u_B_user": 1.0 - x,
                "m_B_merchant": 1.0 - y,
                "lock_index": lock_index(x, y),
                "hhi": hhi(x, y),
                "profit_A": profit_A,
                "profit_B": profit_B,
                "q_A": params["q_A"],
                "q_B": params["q_B"],
                "r_A": params["r_A"],
                "r_B": params["r_B"],
                "s_A_user": action_A["user_subsidy"],
                "s_A_merchant": action_A["merchant_subsidy"],
                "s_B_user": action_B["user_subsidy"],
                "s_B_merchant": action_B["merchant_subsidy"],
                "I_A": action_A["quality_investment"],
                "I_B": action_B["quality_investment"],
                **info,
            }
        )
    return records


def run_stage2_subsidy_experiments(base_config=None):
    """Stage 2: compare subsidy directions and budget intensity."""
    summaries = []
    trajectories = {}
    for network in STAGE234_NETWORK_LEVELS:
        for budget in SUBSIDY_BUDGETS:
            for strategy_label, strategy in SUBSIDY_STRATEGIES:
                params = _base_challenger_config(base_config, network)
                records = run_simulation(
                    params,
                    x0=0.80,
                    y0=0.80,
                    strategy_A=no_strategy,
                    strategy_B=strategy,
                    budget_B=budget,
                )
                run_id = f"stage2_{network['label']}_{strategy_label}_B{budget:g}"
                trajectories[run_id] = records
                summaries.append(
                    {
                        "stage": 2,
                        "experiment": "subsidy_strategy",
                        "run_id": run_id,
                        "network_label": network["label"],
                        "alpha": network["alpha"],
                        "beta": network["beta"],
                        "strategy": strategy_label,
                        "budget": budget,
                        "use_congestion": False,
                        "use_quality_investment": False,
                        **summarize_policy_run(records),
                    }
                )
    return summaries, trajectories


def run_stage2_low_budget_supplement(base_config=None):
    """Stage 2 supplement: inspect whether one-sided subsidies help at low budgets."""
    summaries = []
    trajectories = {}
    low_budget_strategies = [
        ("user_only", user_subsidy_strategy),
        ("balanced", balanced_subsidy_strategy),
        ("user_leaning", user_leaning_subsidy_strategy),
    ]
    budgets = [round(idx * 0.01, 2) for idx in range(0, 21)]
    for budget in budgets:
        for strategy_label, strategy in low_budget_strategies:
            params = _base_challenger_config(base_config, STAGE234_NETWORK_LEVELS[0])
            records = run_simulation(
                params,
                x0=0.80,
                y0=0.80,
                strategy_A=no_strategy,
                strategy_B=strategy,
                budget_B=budget,
            )
            run_id = f"stage2_low_budget_{strategy_label}_B{budget:g}"
            if budget in {0.05, 0.1, 0.15, 0.2}:
                trajectories[run_id] = records
            summaries.append(
                {
                    "stage": 2,
                    "experiment": "low_budget_single_side",
                    "run_id": run_id,
                    "network_label": "medium",
                    "alpha": STAGE234_NETWORK_LEVELS[0]["alpha"],
                    "beta": STAGE234_NETWORK_LEVELS[0]["beta"],
                    "strategy": strategy_label,
                    "budget": budget,
                    "use_congestion": False,
                    "use_quality_investment": False,
                    **summarize_policy_run(records),
                }
            )
    return summaries, trajectories


def run_stage2_exit_subsidy_experiments(base_config=None):
    """Stage 2 supplement: balanced subsidy exits before the horizon ends."""
    params = _base_challenger_config(base_config, STAGE234_NETWORK_LEVELS[0], T=200)
    budgets = [round(idx * 0.05, 2) for idx in range(0, 25)]
    exit_steps = range(0, params["T"] + 1)
    summaries = []
    trajectories = {}

    for budget in budgets:
        for exit_step in exit_steps:
            records = run_exit_subsidy_simulation(
                params,
                total_budget=budget,
                exit_step=exit_step,
            )
            run_id = f"stage2_exit_B{budget:g}_step{exit_step}"
            if budget == 1.2 and exit_step in {0, 30, 60, 120, 179, 200}:
                trajectories[run_id] = records
            summaries.append(
                {
                    "stage": 2,
                    "experiment": "stage_subsidy_exit",
                    "run_id": run_id,
                    "network_label": "medium",
                    "alpha": STAGE234_NETWORK_LEVELS[0]["alpha"],
                    "beta": STAGE234_NETWORK_LEVELS[0]["beta"],
                    "strategy": "balanced_exit",
                    "budget": budget,
                    "exit_step": exit_step,
                    "exit_time": exit_step * params["dt"],
                    "use_congestion": False,
                    "use_quality_investment": False,
                    **summarize_policy_run(records),
                }
            )
    return summaries, trajectories


def run_stage3_congestion_experiments(base_config=None):
    """Stage 3: add supply-shortage penalty and compare subsidy allocation."""
    summaries = []
    trajectories = {}
    for theta in CONGESTION_THETA_LEVELS:
        for strategy_label, strategy in CONGESTION_STRATEGIES:
            params = _base_challenger_config(base_config, STAGE234_NETWORK_LEVELS[0])
            params["theta"] = theta["theta"]
            records = run_simulation(
                params,
                x0=0.80,
                y0=0.80,
                strategy_A=no_strategy,
                strategy_B=strategy,
                budget_B=0.8,
                use_congestion=True,
            )
            run_id = f"stage3_{theta['label']}_{strategy_label}"
            trajectories[run_id] = records
            summaries.append(
                {
                    "stage": 3,
                    "experiment": "congestion_penalty",
                    "run_id": run_id,
                    "network_label": "medium",
                    "alpha": STAGE234_NETWORK_LEVELS[0]["alpha"],
                    "beta": STAGE234_NETWORK_LEVELS[0]["beta"],
                    "strategy": strategy_label,
                    "budget": 0.8,
                    "theta_label": theta["label"],
                    "theta": theta["theta"],
                    "rho": params["rho"],
                    "use_congestion": True,
                    "use_quality_investment": False,
                    **summarize_policy_run(records),
                }
            )
    return summaries, trajectories


def run_stage4_quality_investment_experiments(base_config=None):
    """Stage 4: compare strategies that allocate budget to quality investment."""
    summaries = []
    trajectories = {}
    for network in STAGE234_NETWORK_LEVELS:
        for strategy_label, strategy in QUALITY_INVESTMENT_STRATEGIES:
            params = _base_challenger_config(base_config, network, T=300)
            params["theta"] = 1.0
            records = run_simulation(
                params,
                x0=0.80,
                y0=0.80,
                strategy_A=no_strategy,
                strategy_B=strategy,
                budget_B=0.8,
                use_congestion=True,
                use_quality_investment=True,
            )
            run_id = f"stage4_{network['label']}_{strategy_label}"
            trajectories[run_id] = records
            summaries.append(
                {
                    "stage": 4,
                    "experiment": "quality_investment",
                    "run_id": run_id,
                    "network_label": network["label"],
                    "alpha": network["alpha"],
                    "beta": network["beta"],
                    "strategy": strategy_label,
                    "budget": 0.8,
                    "theta": 1.0,
                    "rho": params["rho"],
                    "use_congestion": True,
                    "use_quality_investment": True,
                    **summarize_policy_run(records),
                }
            )
    return summaries, trajectories


def run_stage4_quality_investment_supplements(base_config=None):
    """Stage 4 supplements: delayed reversal and diminishing returns of quality."""
    summaries = []
    trajectories = {}

    # Strong network effects make the early lock-in pressure visible. Pure
    # quality investment removes immediate subsidy effects, so the path shows
    # whether accumulated quality alone can push B across the critical region.
    strong_network = STAGE234_NETWORK_LEVELS[1]
    for budget in QUALITY_DELAY_BUDGETS:
        params = _base_challenger_config(base_config, strong_network, T=300)
        params["theta"] = 1.0
        records = run_simulation(
            params,
            x0=0.80,
            y0=0.80,
            strategy_A=no_strategy,
            strategy_B=quality_only_strategy,
            budget_B=budget,
            use_congestion=True,
            use_quality_investment=True,
        )
        run_id = f"stage4_quality_delay_strong_B{budget:g}"
        trajectories[run_id] = records
        summaries.append(
            {
                "stage": 4,
                "experiment": "quality_delay_path",
                "run_id": run_id,
                "network_label": "strong",
                "alpha": strong_network["alpha"],
                "beta": strong_network["beta"],
                "strategy": "quality_only",
                "budget": budget,
                "theta": 1.0,
                "rho": params["rho"],
                "use_congestion": True,
                "use_quality_investment": True,
                **summarize_policy_run(records),
            }
        )

    for network in STAGE234_NETWORK_LEVELS:
        for budget in QUALITY_BUDGET_SCAN:
            params = _base_challenger_config(base_config, network, T=300)
            params["theta"] = 1.0
            records = run_simulation(
                params,
                x0=0.80,
                y0=0.80,
                strategy_A=no_strategy,
                strategy_B=quality_only_strategy,
                budget_B=budget,
                use_congestion=True,
                use_quality_investment=True,
            )
            run_id = f"stage4_quality_budget_scan_{network['label']}_B{budget:g}"
            if budget in {0.0, 0.2, 0.6, 1.0, 1.4}:
                trajectories[run_id] = records
            summaries.append(
                {
                    "stage": 4,
                    "experiment": "quality_budget_scan",
                    "run_id": run_id,
                    "network_label": network["label"],
                    "alpha": network["alpha"],
                    "beta": network["beta"],
                    "strategy": "quality_only",
                    "budget": budget,
                    "theta": 1.0,
                    "rho": params["rho"],
                    "use_congestion": True,
                    "use_quality_investment": True,
                    **summarize_policy_run(records),
                }
            )
    return summaries, trajectories


def run_stage234_experiments(base_config=None):
    stage2_summary, stage2_trajectories = run_stage2_subsidy_experiments(base_config)
    stage2_low_summary, stage2_low_trajectories = run_stage2_low_budget_supplement(base_config)
    stage2_exit_summary, stage2_exit_trajectories = run_stage2_exit_subsidy_experiments(base_config)
    stage3_summary, stage3_trajectories = run_stage3_congestion_experiments(base_config)
    stage4_summary, stage4_trajectories = run_stage4_quality_investment_experiments(
        base_config
    )
    (
        stage4_supplement_summary,
        stage4_supplement_trajectories,
    ) = run_stage4_quality_investment_supplements(base_config)
    trajectories = {
        **stage2_trajectories,
        **stage2_low_trajectories,
        **stage2_exit_trajectories,
        **stage3_trajectories,
        **stage4_trajectories,
        **stage4_supplement_trajectories,
    }
    return (
        stage2_summary
        + stage2_low_summary
        + stage2_exit_summary
        + stage3_summary
        + stage4_summary
        + stage4_supplement_summary,
        trajectories,
    )


def save_stage234_outputs(output_dir="results"):
    output_dir = Path(output_dir)
    summaries, trajectories = run_stage234_experiments()
    write_csv(output_dir / "tables" / "stage234_summary.csv", summaries)
    write_json(output_dir / "tables" / "stage234_summary.json", summaries)

    trajectory_rows = []
    for run_id, records in trajectories.items():
        for row in records:
            trajectory_rows.append({"run_id": run_id, **row})
    write_csv(output_dir / "tables" / "stage234_trajectories.csv", trajectory_rows)
    return summaries, trajectories
