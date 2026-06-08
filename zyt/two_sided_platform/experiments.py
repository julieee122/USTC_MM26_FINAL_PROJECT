"""Experiment runners for stage 1 of the implementation plan."""

import csv
import json
from copy import deepcopy
from pathlib import Path

from .config import (
    BASE_CONFIG,
    INITIAL_SHARE_SCENARIOS,
    NETWORK_EFFECT_LEVELS,
    QUALITY_ADVANTAGE_SCENARIOS,
)
from .metrics import hhi, is_lock_in, lock_index
from .model_base import run_simulation
from .strategies import no_strategy


def _final_record(records):
    return records[-1]


def convergence_time(records, tolerance=1e-5, window=10):
    """First time from which all later window changes remain smaller than tolerance."""
    if len(records) <= window:
        return records[-1]["time"]

    diffs = []
    for prev, curr in zip(records, records[1:]):
        diffs.append(
            max(
                abs(curr["x_A_user"] - prev["x_A_user"]),
                abs(curr["y_A_merchant"] - prev["y_A_merchant"]),
            )
        )

    for idx in range(0, len(diffs) - window + 1):
        if max(diffs[idx : idx + window]) <= tolerance:
            return records[idx]["time"]
    return records[-1]["time"]


def first_time_b_overtakes(records, both_sides=True):
    for row in records:
        user_overtake = row["u_B_user"] > 0.5
        merchant_overtake = row["m_B_merchant"] > 0.5
        if (user_overtake and merchant_overtake) if both_sides else user_overtake:
            return row["time"]
    return None


def summarize_run(records):
    final = _final_record(records)
    x_T = final["x_A_user"]
    y_T = final["y_A_merchant"]
    return {
        "x_T": x_T,
        "y_T": y_T,
        "B_user_T": 1.0 - x_T,
        "B_merchant_T": 1.0 - y_T,
        "lock_index": lock_index(x_T, y_T),
        "hhi": hhi(x_T, y_T),
        "is_lock_in": is_lock_in(x_T, y_T),
        "convergence_time": convergence_time(records),
        "B_overtake_time": first_time_b_overtakes(records),
    }


def run_network_initial_scale_experiment(
    base_config=None,
    network_levels=None,
    initial_scenarios=None,
):
    """Stage 1 experiment 1: network effects and initial scale coupling."""
    base_config = deepcopy(base_config or BASE_CONFIG)
    network_levels = network_levels or NETWORK_EFFECT_LEVELS
    initial_scenarios = initial_scenarios or INITIAL_SHARE_SCENARIOS

    summaries = []
    trajectories = {}
    for network in network_levels:
        for scenario in initial_scenarios:
            params = deepcopy(base_config)
            params["alpha"] = network["alpha"]
            params["beta"] = network["beta"]
            records = run_simulation(
                params,
                x0=scenario["x0"],
                y0=scenario["y0"],
                strategy_A=no_strategy,
                strategy_B=no_strategy,
            )
            run_id = f"exp1_{network['label']}_{scenario['label']}"
            trajectories[run_id] = records
            summaries.append(
                {
                    "experiment": "network_initial_scale",
                    "run_id": run_id,
                    "network_label": network["label"],
                    "alpha": network["alpha"],
                    "beta": network["beta"],
                    "initial_label": scenario["label"],
                    "x0": scenario["x0"],
                    "y0": scenario["y0"],
                    **summarize_run(records),
                }
            )
    return summaries, trajectories


def run_quality_break_lock_experiment(
    base_config=None,
    network_levels=None,
    quality_scenarios=None,
    x0=0.80,
    y0=0.80,
):
    """Stage 1 experiment 2: whether B quality advantage breaks A lock-in."""
    base_config = deepcopy(base_config or BASE_CONFIG)
    network_levels = network_levels or NETWORK_EFFECT_LEVELS
    quality_scenarios = quality_scenarios or QUALITY_ADVANTAGE_SCENARIOS

    summaries = []
    trajectories = {}
    for network in network_levels:
        critical_break_delta = None
        critical_full_delta = None
        for quality in quality_scenarios:
            params = deepcopy(base_config)
            params["alpha"] = network["alpha"]
            params["beta"] = network["beta"]
            params["q_A"] = 1.0
            params["r_A"] = 1.0
            params["q_B"] = 1.0 + quality["delta_q"]
            params["r_B"] = 1.0 + quality["delta_r"]
            records = run_simulation(
                params,
                x0=x0,
                y0=y0,
                strategy_A=no_strategy,
                strategy_B=no_strategy,
            )
            summary = summarize_run(records)
            b_breaks_lock = (
                summary["B_user_T"] >= 0.5 and summary["B_merchant_T"] >= 0.5
            )
            b_fully_overtakes = (
                summary["B_user_T"] >= 0.99 and summary["B_merchant_T"] >= 0.99
            )
            if critical_break_delta is None and b_breaks_lock:
                critical_break_delta = quality["delta_q"]
            if critical_full_delta is None and b_fully_overtakes:
                critical_full_delta = quality["delta_q"]

            run_id = f"exp2_{network['label']}_{quality['label']}"
            trajectories[run_id] = records
            summaries.append(
                {
                    "experiment": "quality_break_lock",
                    "run_id": run_id,
                    "network_label": network["label"],
                    "alpha": network["alpha"],
                    "beta": network["beta"],
                    "quality_label": quality["label"],
                    "delta_q": quality["delta_q"],
                    "delta_r": quality["delta_r"],
                    "x0": x0,
                    "y0": y0,
                    "B_breaks_lock_final": b_breaks_lock,
                    "B_fully_overtakes_final": b_fully_overtakes,
                    "B_overtakes_final": b_breaks_lock,
                    **summary,
                }
            )

        for row in summaries:
            if (
                row["experiment"] == "quality_break_lock"
                and row["network_label"] == network["label"]
            ):
                row["critical_break_delta"] = critical_break_delta
                row["critical_full_delta"] = critical_full_delta
                row["critical_quality_delta"] = critical_break_delta
    return summaries, trajectories


def run_stage1_experiments(base_config=None):
    exp1_summary, exp1_trajectories = run_network_initial_scale_experiment(base_config)
    exp2_summary, exp2_trajectories = run_quality_break_lock_experiment(base_config)
    trajectories = {**exp1_trajectories, **exp2_trajectories}
    return exp1_summary + exp2_summary, trajectories


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_stage1_outputs(output_dir="results"):
    output_dir = Path(output_dir)
    summaries, trajectories = run_stage1_experiments()
    write_csv(output_dir / "tables" / "stage1_summary.csv", summaries)
    write_json(output_dir / "tables" / "stage1_summary.json", summaries)

    trajectory_rows = []
    for run_id, records in trajectories.items():
        for row in records:
            trajectory_rows.append({"run_id": run_id, **row})
    write_csv(output_dir / "tables" / "stage1_trajectories.csv", trajectory_rows)
    return summaries, trajectories
