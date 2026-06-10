# experiments/report_common.py

from __future__ import annotations

import csv
import inspect
import os
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt

from src.config import Params
from src.model import simulate_path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "SimHei",
        "Microsoft YaHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["savefig.dpi"] = 220


def save_rows_csv(rows: list[dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))

    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def concentration(x: float, y: float) -> float:
    return abs(x - 0.5) + abs(y - 0.5)


def combined_share(x: float, y: float) -> float:
    return 0.5 * (x + y)


def lock_index(x: float, y: float) -> float:
    return max(x, 1.0 - x) * max(y, 1.0 - y)


def market_state_by_c(c: float) -> str:
    if c < 0.2:
        return "双平台共存"
    if c < 0.8:
        return "市场倾斜"
    return "市场锁定"


def platform_state(x: float, y: float) -> str:
    if x >= 0.8 and y >= 0.8:
        return "平台锁定"
    if x <= 0.2 and y <= 0.2:
        return "平台失败"
    if concentration(x, y) < 0.2:
        return "双平台共存"
    return "市场倾斜"


def _params_accepts() -> set[str]:
    try:
        sig = inspect.signature(Params)
        return set(sig.parameters.keys())
    except Exception:
        return set()


def make_params(**kwargs: Any) -> Params:
    """
    兼容不同版本 Params 的字段名。
    你的 config.py 如果字段名不同，这里会自动过滤或映射。
    """
    accepts = _params_accepts()

    aliases = {
    "alpha": ["alpha"],
    "beta": ["beta"],

    "eta_u": ["eta_u", "s_u", "sU", "switch_u"],
    "eta_m": ["eta_m", "s_m", "sM", "switch_m"],

    "dq_u_base": ["dq_u_base", "q_u_base", "qA_u", "quality_u"],
    "dq_m_base": ["dq_m_base", "q_m_base", "qA_m", "quality_m"],

    "N_U": ["N_U", "n_users", "num_users", "total_users"],
    "N_M": ["N_M", "n_merchants", "num_merchants", "total_merchants"],

    "rho": ["rho", "shortage_rho", "capacity_ratio"],
    "theta": ["theta", "shortage_theta", "shortage_penalty"],
    "epsilon": ["epsilon", "eps", "shortage_buffer"],

    "shortage_enabled": ["shortage_enabled"],

    "lambda_q": ["lambda_q", "quality_efficiency", "mu_q", "mu", "invest_eff_u", "invest_eff_m"],
    "quality_decay": ["quality_decay", "d", "delta_q", "delta"],
    "qmax": ["qmax", "q_max", "quality_max"],

    "discount": ["discount", "discount_factor"],
   }

    if not accepts:
        try:
            return Params(**kwargs)
        except TypeError:
            return Params()

    translated = {}

    for key, value in kwargs.items():
        candidate_names = aliases.get(key, [key])
        for name in candidate_names:
            if name in accepts:
                translated[name] = value
                break
    
    # 如果模型使用 invest_eff_u / invest_eff_m，而不是 lambda_q，则同步写入两侧投资效率
    if "lambda_q" in kwargs:
        if "invest_eff_u" in accepts:
            translated["invest_eff_u"] = kwargs["lambda_q"]
        if "invest_eff_m" in accepts:
            translated["invest_eff_m"] = kwargs["lambda_q"]

# 如果模型使用 shortage_buffer，而不是 epsilon，则同步写入
    if "epsilon" in kwargs and "shortage_buffer" in accepts:
        translated["shortage_buffer"] = kwargs["epsilon"]

# 如果模型需要显式开启供给不足惩罚
    if kwargs.get("shortage_enabled", False) and "shortage_enabled" in accepts:
       translated["shortage_enabled"] = True

# 如果模型只有 shortage_rho，而没有 rho，也同步写入
    if "rho" in kwargs and "shortage_rho" in accepts:
        translated["shortage_rho"] = kwargs["rho"]

    return Params(**translated)


def call_simulate(
    x0: float,
    y0: float,
    params: Params,
    T: float,
    dt: float,
    policy: Callable | None = None,
):
    """
    兼容不同 simulate_path 接口。
    """
    try:
        return simulate_path(x0, y0, params, T=T, dt=dt, policy=policy)
    except TypeError:
        try:
            return simulate_path(x0, y0, params, T=T, dt=dt, strategy=policy)
        except TypeError:
            try:
                return simulate_path(x0, y0, params, T, dt, policy)
            except TypeError:
                return simulate_path(x0, y0, params)


def get_series(result: dict[str, Any], keys: list[str], default: float = np.nan) -> np.ndarray:
    for key in keys:
        if key in result:
            return np.asarray(result[key], dtype=float)
    return np.asarray([default], dtype=float)


def last_value(result: dict[str, Any], keys: list[str], default: float = np.nan) -> float:
    arr = get_series(result, keys, default)
    if len(arr) == 0:
        return float(default)
    return float(arr[-1])


def get_x(result: dict[str, Any]) -> float:
    return last_value(result, ["x", "u", "user_share"], default=np.nan)


def get_y(result: dict[str, Any]) -> float:
    return last_value(result, ["y", "m", "merchant_share"], default=np.nan)


def get_profit(result: dict[str, Any]) -> float:
    return last_value(
        result,
        [
            "cum_discounted_profit",
            "discounted_profit",
            "cum_profit",
            "profit",
            "total_profit",
        ],
        default=np.nan,
    )


def get_shortage(result: dict[str, Any]) -> tuple[float, float]:
    shortage = get_series(
        result,
        [
            "shortage_A",
            "shortage_B",
            "shortage",
            "C_A",
            "C_B",
            "congestion",
            "penalty",
        ],
        default=np.nan,
    )

    if np.all(np.isnan(shortage)):
        return np.nan, np.nan

    return float(np.nanmax(shortage)), float(np.nanmean(shortage))


def get_quality(result: dict[str, Any]) -> float:
    return last_value(
        result,
        ["q_u", "q_user", "quality_u", "q_B", "quality"],
        default=np.nan,
    )


def static_policy(sub_u: float, sub_m: float, inv: float = 0.0):
    def policy(t: float, state: dict[str, Any], p: Params):
        return {
            "ds_u": float(sub_u),
            "ds_m": float(sub_m),
            "inv_u": float(inv),
            "inv_m": float(inv),
        }

    return policy


def step_subsidy_policy(budget: float, duration: float, split: float = 0.5):
    def policy(t: float, state: dict[str, Any], p: Params):
        if t <= duration:
            return {
                "ds_u": float(split * budget),
                "ds_m": float((1.0 - split) * budget),
                "inv_u": 0.0,
                "inv_m": 0.0,
            }

        return {
            "ds_u": 0.0,
            "ds_m": 0.0,
            "inv_u": 0.0,
            "inv_m": 0.0,
        }

    return policy