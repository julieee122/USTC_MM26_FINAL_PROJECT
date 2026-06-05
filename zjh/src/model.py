import numpy as np


def sigmoid(z):
    """
    稳定版本 sigmoid。
    """
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def compute_shortage(x, y, p):
    """
    供给不足刻画。

    shortage_A = max(0, 用户份额 - 商户份额 - 缓冲项)

    如果平台 A 用户增长明显快于商户增长，
    则说明需求扩张过快而供给不足，会降低用户体验。
    """
    shortage_A = np.maximum(
        0.0,
        x - p.supply_ratio * y - p.shortage_buffer
    )

    shortage_B = np.maximum(
        0.0,
        (1.0 - x) - p.supply_ratio * (1.0 - y) - p.shortage_buffer
    )

    return shortage_A, shortage_B


def model_terms(t, state, p, policy):
    """
    双边 Logit 动态模型。

    state = [x, y, q_u, q_m]

    x: 平台 A 用户份额
    y: 平台 A 商户份额
    q_u: 用户侧服务质量投资形成的质量优势
    q_m: 商户侧服务质量投资形成的质量优势
    """
    x, y, q_u_stock, q_m_stock = state

    action = policy(t, state, p)

    ds_u = action.get("ds_u", 0.0)
    ds_m = action.get("ds_m", 0.0)
    inv_u = action.get("inv_u", 0.0)
    inv_m = action.get("inv_m", 0.0)

    shortage_A, shortage_B = compute_shortage(x, y, p)

    if p.shortage_enabled:
        shortage_term = p.shortage_rho * (shortage_A - shortage_B)
    else:
        shortage_term = 0.0

    dq_u = p.dq_u_base + q_u_stock
    dq_m = p.dq_m_base + q_m_stock

    # 用户侧吸引力差异：A 相对 B
    D_u = (
        dq_u
        + p.alpha * (2.0 * y - 1.0)
        + p.eta_u * (2.0 * x - 1.0)
        + ds_u
        - p.dp_u
        - shortage_term
    )

    # 商户侧吸引力差异：A 相对 B
    D_m = (
        dq_m
        + p.beta * (2.0 * x - 1.0)
        + p.eta_m * (2.0 * y - 1.0)
        + ds_m
        - p.dc_m
    )

    P_u = sigmoid(p.lambda_u * D_u)
    P_m = sigmoid(p.lambda_m * D_m)

    dxdt = p.r_u * (P_u - x)
    dydt = p.r_m * (P_m - y)

    dq_u_dt = p.invest_eff_u * inv_u - p.quality_decay * q_u_stock
    dq_m_dt = p.invest_eff_m * inv_m - p.quality_decay * q_m_stock

    revenue_rate = p.revenue_user * x + p.revenue_merchant * y

    subsidy_cost_rate = (
        p.cost_user_subsidy * max(ds_u, 0.0) * x
        + p.cost_merchant_subsidy * max(ds_m, 0.0) * y
    )

    invest_cost_rate = p.cost_invest_u * inv_u + p.cost_invest_m * inv_m
    total_cost_rate = subsidy_cost_rate + invest_cost_rate
    profit_rate = revenue_rate - total_cost_rate

    return {
        "derivative": np.array([dxdt, dydt, dq_u_dt, dq_m_dt], dtype=float),
        "D_u": D_u,
        "D_m": D_m,
        "P_u": P_u,
        "P_m": P_m,
        "ds_u": ds_u,
        "ds_m": ds_m,
        "inv_u": inv_u,
        "inv_m": inv_m,
        "shortage_A": shortage_A,
        "shortage_B": shortage_B,
        "revenue_rate": revenue_rate,
        "subsidy_cost_rate": subsidy_cost_rate,
        "invest_cost_rate": invest_cost_rate,
        "total_cost_rate": total_cost_rate,
        "profit_rate": profit_rate,
    }


def simulate_path(
    x0,
    y0,
    p,
    T=80.0,
    dt=0.03,
    policy=None,
):
    """
    模拟单条市场演化轨迹。
    """
    from src.policies import zero_policy

    if policy is None:
        policy = zero_policy

    n_steps = int(T / dt)
    t_values = np.linspace(0, T, n_steps + 1)

    keys = [
        "t", "x", "y", "q_u", "q_m",
        "D_u", "D_m", "P_u", "P_m",
        "ds_u", "ds_m", "inv_u", "inv_m",
        "shortage_A", "shortage_B",
        "revenue_rate", "subsidy_cost_rate",
        "invest_cost_rate", "total_cost_rate", "profit_rate",
        "cum_revenue", "cum_subsidy_cost", "cum_invest_cost",
        "cum_total_cost", "cum_profit"
    ]

    result = {k: np.zeros(n_steps + 1) for k in keys}

    state = np.array([x0, y0, 0.0, 0.0], dtype=float)

    cum_revenue = 0.0
    cum_subsidy_cost = 0.0
    cum_invest_cost = 0.0
    cum_total_cost = 0.0
    cum_profit = 0.0

    for i, t in enumerate(t_values):
        terms = model_terms(t, state, p, policy)

        result["t"][i] = t
        result["x"][i] = state[0]
        result["y"][i] = state[1]
        result["q_u"][i] = state[2]
        result["q_m"][i] = state[3]

        for k in [
            "D_u", "D_m", "P_u", "P_m",
            "ds_u", "ds_m", "inv_u", "inv_m",
            "shortage_A", "shortage_B",
            "revenue_rate", "subsidy_cost_rate",
            "invest_cost_rate", "total_cost_rate", "profit_rate"
        ]:
            result[k][i] = terms[k]

        result["cum_revenue"][i] = cum_revenue
        result["cum_subsidy_cost"][i] = cum_subsidy_cost
        result["cum_invest_cost"][i] = cum_invest_cost
        result["cum_total_cost"][i] = cum_total_cost
        result["cum_profit"][i] = cum_profit

        if i < n_steps:
            state = state + dt * terms["derivative"]

            state[0] = np.clip(state[0], 0.0, 1.0)
            state[1] = np.clip(state[1], 0.0, 1.0)
            state[2] = max(state[2], 0.0)
            state[3] = max(state[3], 0.0)

            cum_revenue += terms["revenue_rate"] * dt
            cum_subsidy_cost += terms["subsidy_cost_rate"] * dt
            cum_invest_cost += terms["invest_cost_rate"] * dt
            cum_total_cost += terms["total_cost_rate"] * dt
            cum_profit += terms["profit_rate"] * dt

    return result


def simulate_grid_initial(
    p,
    alpha,
    beta,
    grid_n=51,
    T=70.0,
    dt=0.05,
):
    """
    对不同初始份额 x(0), y(0) 做网格模拟。
    用于分析初始规模和网络效应耦合。
    """
    grid = np.linspace(0.05, 0.95, grid_n)
    X, Y = np.meshgrid(grid, grid)

    steps = int(T / dt)

    for _ in range(steps):
        D_u = (
            p.dq_u_base
            + alpha * (2 * Y - 1)
            + p.eta_u * (2 * X - 1)
        )

        D_m = (
            p.dq_m_base
            + beta * (2 * X - 1)
            + p.eta_m * (2 * Y - 1)
        )

        P_u = sigmoid(p.lambda_u * D_u)
        P_m = sigmoid(p.lambda_m * D_m)

        X = X + dt * p.r_u * (P_u - X)
        Y = Y + dt * p.r_m * (P_m - Y)

        X = np.clip(X, 0.0, 1.0)
        Y = np.clip(Y, 0.0, 1.0)

    return grid, X, Y


def simulate_alpha_beta_scan(
    p,
    x0,
    y0,
    alpha_values,
    beta_values,
    T=70.0,
    dt=0.05,
):
    """
    扫描 alpha 和 beta 不相等的情况。
    横轴 alpha，纵轴 beta，颜色表示最终 x*。
    """
    A, B = np.meshgrid(alpha_values, beta_values)

    X = np.ones_like(A) * x0
    Y = np.ones_like(A) * y0

    steps = int(T / dt)

    for _ in range(steps):
        D_u = (
            p.dq_u_base
            + A * (2 * Y - 1)
            + p.eta_u * (2 * X - 1)
        )

        D_m = (
            p.dq_m_base
            + B * (2 * X - 1)
            + p.eta_m * (2 * Y - 1)
        )

        P_u = sigmoid(p.lambda_u * D_u)
        P_m = sigmoid(p.lambda_m * D_m)

        X = X + dt * p.r_u * (P_u - X)
        Y = Y + dt * p.r_m * (P_m - Y)

        X = np.clip(X, 0.0, 1.0)
        Y = np.clip(Y, 0.0, 1.0)

    return A, B, X, Y