from __future__ import annotations

import csv
import inspect
import os
from typing import Any, Callable

import numpy as np
import matplotlib.pyplot as plt

from src.config import Params
from src.model import simulate_path, simulate_path_rk45


def ensure_dir(path: str) -> None:
    if path:
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
    未在 Params 中定义的字段会自动过滤。
    """
    accepts = _params_accepts()

    aliases = {
        "alpha": ["alpha"],
        "beta": ["beta"],

        "eta_u": ["eta_u", "s_u", "sU", "switch_u"],
        "eta_m": ["eta_m", "s_m", "sM", "switch_m"],

        "dq_u_base": ["dq_u_base", "q_u_base", "qA_u", "quality_u"],
        "dq_m_base": ["dq_m_base", "q_m_base", "qA_m", "quality_m"],

        "lambda_u": ["lambda_u"],
        "lambda_m": ["lambda_m"],

        "N_U": ["N_U", "n_users", "num_users", "total_users"],
        "N_M": ["N_M", "n_merchants", "num_merchants", "total_merchants"],

        "rho": ["shortage_rho", "rho", "capacity_ratio"],
        "theta": ["shortage_theta", "theta", "shortage_penalty"],
        "epsilon": ["shortage_buffer", "epsilon", "eps"],

        "shortage_enabled": ["shortage_enabled"],
        "shortage_mode": ["shortage_mode"],

        "lambda_q": ["invest_eff_u", "invest_eff_m", "lambda_q", "quality_efficiency", "mu_q", "mu"],
        "invest_eff_u": ["invest_eff_u"],
        "invest_eff_m": ["invest_eff_m"],
        "quality_decay": ["quality_decay", "d", "delta_q", "delta"],
        "qmax": ["qmax", "q_max", "quality_max"],

        "quality_base_effect_scale": ["quality_base_effect_scale"],
        "quality_stock_effect_scale": ["quality_stock_effect_scale"],

        "discount": ["discount", "discount_factor"],

        "profit_mu": ["profit_mu"],
        "use_report_profit": ["use_report_profit"],
        "subsidy_effect_scale": ["subsidy_effect_scale"],
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

    # lambda_q 同步写入两侧投资效率
    if "lambda_q" in kwargs:
        if "invest_eff_u" in accepts:
            translated["invest_eff_u"] = kwargs["lambda_q"]
        if "invest_eff_m" in accepts:
            translated["invest_eff_m"] = kwargs["lambda_q"]

    # rho / theta / epsilon 兼容
    if "rho" in kwargs and "shortage_rho" in accepts:
        translated["shortage_rho"] = kwargs["rho"]
    if "theta" in kwargs and "shortage_theta" in accepts:
        translated["shortage_theta"] = kwargs["theta"]
    if "epsilon" in kwargs and "shortage_buffer" in accepts:
        translated["shortage_buffer"] = kwargs["epsilon"]

    # 显式开启供给不足
    if kwargs.get("shortage_enabled", False) and "shortage_enabled" in accepts:
        translated["shortage_enabled"] = True

    return Params(**translated)


def call_simulate(
    x0: float,
    y0: float,
    params: Params,
    T: float,
    dt: float,
    policy: Callable | None = None,
    method: str = "euler",
):
    """
    统一仿真入口。
    method="euler"：显式欧拉；
    method="rk45"：scipy RK45。
    """
    method = method.lower()

    if method == "rk45":
        return simulate_path_rk45(
            x0,
            y0,
            params,
            T=T,
            dt=dt,
            policy=policy,
        )

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


def _last_value(result: Any, candidates: list[str], default=np.nan) -> float:
    for key in candidates:
        if isinstance(result, dict) and key in result:
            arr = np.asarray(result[key], dtype=float)
            if arr.size > 0:
                return float(arr[-1])

        if hasattr(result, key):
            arr = np.asarray(getattr(result, key), dtype=float)
            if arr.size > 0:
                return float(arr[-1])

    return float(default)


def get_x(result: Any) -> float:
    return _last_value(result, ["x", "u_A", "user_share_A"])


def get_y(result: Any) -> float:
    return _last_value(result, ["y", "m_A", "merchant_share_A"])


def get_profit(result: Any) -> float:
    return _last_value(
        result,
        [
            "cum_discounted_profit",
            "cum_profit",
            "discounted_profit",
            "profit",
            "cum_discounted_profit_B",
            "cum_profit_B",
        ],
    )


def get_quality(result: Any) -> float:
    return _last_value(result, ["q_u", "quality", "q"])


def get_shortage(result: Any) -> tuple[float, float]:
    """
    兼容旧接口，默认读取平台 B 供给不足。
    """
    return get_shortage_B(result)


def get_shortage_A(result: Any) -> tuple[float, float]:
    if isinstance(result, dict) and "shortage_A" in result:
        arr = np.asarray(result["shortage_A"], dtype=float)
        return float(np.max(arr)), float(np.mean(arr))
    return 0.0, 0.0


def get_shortage_B(result: Any) -> tuple[float, float]:
    if isinstance(result, dict) and "shortage_B" in result:
        arr = np.asarray(result["shortage_B"], dtype=float)
        return float(np.max(arr)), float(np.mean(arr))
    return 0.0, 0.0


def static_policy(sub_u: float, sub_m: float, inv: float = 0.0):
    """
    平台 A 的静态策略。保留用于兼容旧实验。
    """
    def policy(t: float, state: dict[str, Any], p: Params):
        return {
            "ds_u_A": float(sub_u),
            "ds_m_A": float(sub_m),
            "inv_u_A": float(inv),
            "inv_m_A": float(inv),
        }

    return policy


def static_policy_B(sub_u: float, sub_m: float, inv: float = 0.0):
    """
    平台 B 的静态策略。
    用于阶段 2--5：平台 A 初始占优，平台 B 作为挑战者采取补贴/质量投资。
    """
    def policy(t: float, state: dict[str, Any], p: Params):
        return {
            "ds_u_B": float(sub_u),
            "ds_m_B": float(sub_m),
            "inv_u_B": float(inv),
            "inv_m_B": float(inv),
        }

    return policy


def step_subsidy_policy(budget: float, duration: float, split: float = 0.5):
    """
    平台 A 的阶段性补贴策略，保留用于兼容旧代码。
    """
    def policy(t: float, state: dict[str, Any], p: Params):
        if t <= duration:
            return {
                "ds_u_A": float(split * budget),
                "ds_m_A": float((1.0 - split) * budget),
                "inv_u_A": 0.0,
                "inv_m_A": 0.0,
            }

        return {
            "ds_u_A": 0.0,
            "ds_m_A": 0.0,
            "inv_u_A": 0.0,
            "inv_m_A": 0.0,
        }

    return policy


def step_subsidy_policy_B(budget: float, duration: float, split: float = 0.5):
    """
    平台 B 的阶段性补贴策略。
    """
    def policy(t: float, state: dict[str, Any], p: Params):
        if t <= duration:
            return {
                "ds_u_B": float(split * budget),
                "ds_m_B": float((1.0 - split) * budget),
                "inv_u_B": 0.0,
                "inv_m_B": 0.0,
            }

        return {
            "ds_u_B": 0.0,
            "ds_m_B": 0.0,
            "inv_u_B": 0.0,
            "inv_m_B": 0.0,
        }

    return policy


def final_B_from_result(result: Any) -> tuple[float, float, float, float, float]:
    """
    由模型输出的 A 份额反推出平台 B 最终份额。
    """
    xA = get_x(result)
    yA = get_y(result)
    xB = 1.0 - xA
    yB = 1.0 - yA
    LB = combined_share(xB, yB)
    return xA, yA, xB, yB, LB