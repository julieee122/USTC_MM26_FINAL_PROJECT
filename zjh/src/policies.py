import numpy as np
from src.model import compute_shortage


def zero_policy(t, state, p):
    """
    无补贴、无服务质量投资。
    """
    return {
        "ds_u": 0.0,
        "ds_m": 0.0,
        "inv_u": 0.0,
        "inv_m": 0.0,
    }


def make_decay_subsidy_policy(
    su0,
    sm0,
    decay_u=0.06,
    decay_m=0.06,
    duration=None
):
    """
    指数衰减补贴策略。

    su0: 初始用户补贴强度
    sm0: 初始商户补贴强度
    decay_u: 用户补贴衰减速度
    decay_m: 商户补贴衰减速度
    duration: 若不为 None，则超过 duration 后补贴停止
    """
    def policy(t, state, p):
        if duration is not None and t > duration:
            ds_u = 0.0
            ds_m = 0.0
        else:
            ds_u = su0 * np.exp(-decay_u * t)
            ds_m = sm0 * np.exp(-decay_m * t)

        return {
            "ds_u": float(ds_u),
            "ds_m": float(ds_m),
            "inv_u": 0.0,
            "inv_m": 0.0,
        }

    return policy


def greedy_quality_policy(t, state, p):
    """
    贪心服务质量投资策略：
    前期集中投资用户侧服务质量，快速拉动用户。
    """
    if t <= 10:
        inv_u = 0.75
        inv_m = 0.10
    else:
        inv_u = 0.0
        inv_m = 0.0

    return {
        "ds_u": 0.0,
        "ds_m": 0.0,
        "inv_u": inv_u,
        "inv_m": inv_m,
    }


def long_term_quality_policy(t, state, p):
    """
    长期服务质量投资策略：
    持续、均衡地投资用户侧和商户侧。
    """
    inv_u = 0.22
    inv_m = 0.22

    return {
        "ds_u": 0.0,
        "ds_m": 0.0,
        "inv_u": inv_u,
        "inv_m": inv_m,
    }


def dynamic_quality_policy(t, state, p):
    """
    动态服务质量投资策略：
    根据当前用户份额、商户份额和供给不足程度动态分配投资。
    """
    x, y, q_u, q_m = state
    shortage_A, _ = compute_shortage(x, y, p)

    if x > 0.82 and y > 0.82:
        total_budget = 0.25
    else:
        total_budget = 0.65

    need_u = max(0.0, 0.55 - x) + 0.20 * max(0.0, y - x)
    need_m = max(0.0, 0.55 - y) + 1.50 * shortage_A + 0.20 * max(0.0, x - y)

    need_u += 0.05
    need_m += 0.05

    total_need = need_u + need_m

    inv_u = total_budget * need_u / total_need
    inv_m = total_budget * need_m / total_need

    return {
        "ds_u": 0.0,
        "ds_m": 0.0,
        "inv_u": float(inv_u),
        "inv_m": float(inv_m),
    }