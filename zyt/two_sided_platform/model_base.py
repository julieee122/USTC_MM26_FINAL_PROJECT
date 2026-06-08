"""Logit choice dynamics and Euler numerical solver."""

from copy import deepcopy
from math import exp

from .metrics import hhi, lock_index, platform_profit
from .strategies import no_strategy


def clip_share(value):
    return min(1.0, max(0.0, value))


def compute_congestion(
    user_share,
    merchant_share,
    rho=10.0,
    eps=1e-6,
    user_scale=1000.0,
    merchant_scale=50.0,
):
    actual_users = user_share * user_scale
    actual_merchants = merchant_share * merchant_scale
    return max(0.0, actual_users / (actual_merchants + eps) - rho)


def compute_utilities(x, y, params, action_A=None, action_B=None, use_congestion=False):
    """Return utilities, utility gaps, and Logit choice probabilities."""
    action_A = action_A or no_strategy()
    action_B = action_B or no_strategy()

    u_A, u_B = x, 1.0 - x
    m_A, m_B = y, 1.0 - y

    if use_congestion:
        C_A = compute_congestion(
            u_A,
            m_A,
            params["rho"],
            params["eps"],
            params.get("user_scale", 1000.0),
            params.get("merchant_scale", 50.0),
        )
        C_B = compute_congestion(
            u_B,
            m_B,
            params["rho"],
            params["eps"],
            params.get("user_scale", 1000.0),
            params.get("merchant_scale", 50.0),
        )
    else:
        C_A = 0.0
        C_B = 0.0

    Uu_A = (
        params["q_A"]
        + params["alpha"] * m_A
        + action_A["user_subsidy"]
        - params["p_A"]
        - params["theta"] * C_A
        + params.get("s_U", 0.0) * u_A
    )
    Uu_B = (
        params["q_B"]
        + params["alpha"] * m_B
        + action_B["user_subsidy"]
        - params["p_B"]
        - params["theta"] * C_B
        + params.get("s_U", 0.0) * u_B
    )
    Um_A = (
        params["r_A"]
        + params["beta"] * u_A
        + action_A["merchant_subsidy"]
        - params["c_A"]
        + params.get("s_M", 0.0) * m_A
    )
    Um_B = (
        params["r_B"]
        + params["beta"] * u_B
        + action_B["merchant_subsidy"]
        - params["c_B"]
        + params.get("s_M", 0.0) * m_B
    )

    delta_Uu = Uu_A - Uu_B
    delta_Um = Um_A - Um_B
    P_A_user = logit_probability(delta_Uu, params.get("gamma_U", 1.0))
    P_A_merchant = logit_probability(delta_Um, params.get("gamma_M", 1.0))

    return {
        "Uu_A": Uu_A,
        "Uu_B": Uu_B,
        "Um_A": Um_A,
        "Um_B": Um_B,
        "C_A": C_A,
        "C_B": C_B,
        "delta_Uu": delta_Uu,
        "delta_Um": delta_Um,
        "P_A_user": P_A_user,
        "P_A_merchant": P_A_merchant,
    }


def logit_probability(delta_utility, gamma):
    z = max(-700.0, min(700.0, gamma * delta_utility))
    return 1.0 / (1.0 + exp(-z))


def derivatives(x, y, params, action_A=None, action_B=None, use_congestion=False):
    """Continuous Logit adjustment dynamic: dx/dt and dy/dt."""
    info = compute_utilities(x, y, params, action_A, action_B, use_congestion)
    dx = params.get("lambda_U", 1.0) * (info["P_A_user"] - x)
    dy = params.get("lambda_M", 1.0) * (info["P_A_merchant"] - y)
    return dx, dy, info


def euler_step(x, y, params, action_A=None, action_B=None, use_congestion=False):
    """Advance the coupled Logit adjustment ODE by one Euler step."""
    dx, dy, info = derivatives(x, y, params, action_A, action_B, use_congestion)
    x_new = clip_share(x + params["dt"] * dx)
    y_new = clip_share(y + params["dt"] * dy)
    return x_new, y_new, {"dx": dx, "dy": dy, **info}


def update_quality(q, investment, lambda_q=0.05, decay=0.01, q_max=3.0):
    q_new = q + lambda_q * investment - decay * q
    return min(q_max, max(0.0, q_new))


def _call_strategy(strategy, budget, params, x, y):
    if strategy.__name__ == "dynamic_strategy":
        return strategy(
            budget,
            user_share=1.0 - x,
            merchant_share=1.0 - y,
            rho=params["rho"],
            eps=params["eps"],
            user_scale=params.get("user_scale", 1000.0),
            merchant_scale=params.get("merchant_scale", 50.0),
        )
    return strategy(budget)


def run_simulation(
    params,
    x0,
    y0,
    strategy_A=no_strategy,
    strategy_B=no_strategy,
    budget_A=0.0,
    budget_B=0.0,
    use_congestion=False,
    use_quality_investment=False,
):
    """Run a full Euler simulation and return per-step records."""
    params = deepcopy(params)
    x = clip_share(x0)
    y = clip_share(y0)
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
        action_A = _call_strategy(strategy_A, budget_A, params, x, y)
        action_B = _call_strategy(strategy_B, budget_B, params, x, y)
        x, y, info = euler_step(x, y, params, action_A, action_B, use_congestion)

        if use_quality_investment:
            quality_args = {
                "lambda_q": params.get("lambda_q", 0.05),
                "decay": params.get("quality_decay", 0.01),
                "q_max": params.get("q_max", 3.0),
            }
            params["q_A"] = update_quality(
                params["q_A"], action_A["quality_investment"], **quality_args
            )
            params["r_A"] = update_quality(
                params["r_A"], action_A["quality_investment"], **quality_args
            )
            params["q_B"] = update_quality(
                params["q_B"], action_B["quality_investment"], **quality_args
            )
            params["r_B"] = update_quality(
                params["r_B"], action_B["quality_investment"], **quality_args
            )

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
