import numpy as np


def sigmoid(z):
    """
    稳定版本 sigmoid。
    """
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def compute_shortage(x, y, p):
    
    eps = getattr(p, "shortage_buffer", 1e-6)
    rho = getattr(p, "shortage_rho", 10.0)
    N_U = getattr(p, "N_U", 1000)
    N_M = getattr(p, "N_M", 50)

    shortage_A = np.maximum(
        0.0,
        N_U * x / (N_M * y + eps) - rho,
    )

    shortage_B = np.maximum(
        0.0,
        N_U * (1.0 - x) / (N_M * (1.0 - y) + eps) - rho,
    )

    return shortage_A, shortage_B


def _read_action(action, key, default=0.0):
    try:
        return float(action.get(key, default))
    except Exception:
        return float(default)


def model_terms(t, state, p, policy):
    """
    双边 Logit 动态模型。

    统一口径：
    - x, y 始终表示平台 A 的用户份额、商户份额；
    - 平台 B 份额为 1-x, 1-y；
    - q_u, q_m 表示平台 A 相对平台 B 的服务质量优势；
      因此平台 B 做质量投资时，q_u/q_m 会下降为负数。

    策略字段：
    - ds_u_A, ds_m_A, inv_u_A, inv_m_A：平台 A 策略；
    - ds_u_B, ds_m_B, inv_u_B, inv_m_B：平台 B 策略；
    - 为兼容旧代码，ds_u/ds_m/inv_u/inv_m 默认解释为平台 A 策略。
    """
    x, y, q_u_stock, q_m_stock = state

    action = policy(t, state, p)
    if action is None:
        action = {}

    # ------------------------------------------------------------
    # 读取 A/B 双方策略
    # ------------------------------------------------------------

    ds_u_A = _read_action(action, "ds_u_A", _read_action(action, "ds_u", 0.0))
    ds_m_A = _read_action(action, "ds_m_A", _read_action(action, "ds_m", 0.0))
    inv_u_A = _read_action(action, "inv_u_A", _read_action(action, "inv_u", 0.0))
    inv_m_A = _read_action(action, "inv_m_A", _read_action(action, "inv_m", 0.0))

    ds_u_B = _read_action(action, "ds_u_B", 0.0)
    ds_m_B = _read_action(action, "ds_m_B", 0.0)
    inv_u_B = _read_action(action, "inv_u_B", 0.0)
    inv_m_B = _read_action(action, "inv_m_B", 0.0)

    subsidy_effect_scale = getattr(p, "subsidy_effect_scale", 1.0)

    # A 相对 B 的净策略优势。
    net_ds_u = subsidy_effect_scale * (ds_u_A - ds_u_B)
    net_ds_m = subsidy_effect_scale * (ds_m_A - ds_m_B)

    net_inv_u = inv_u_A - inv_u_B
    net_inv_m = inv_m_A - inv_m_B

    # ------------------------------------------------------------
    # 供给不足惩罚
    # ------------------------------------------------------------

    shortage_A, shortage_B = compute_shortage(x, y, p)

    if getattr(p, "shortage_enabled", False):
        theta = getattr(p, "shortage_theta", 1.0)
        mode = getattr(p, "shortage_mode", "absolute_B")

        if mode == "absolute_B":
            # D_u 是 A 相对 B 的用户侧吸引力。
            # B 的供给不足越大，B 吸引力越低，A 相对 B 吸引力越高。
            shortage_shift_for_A = theta * shortage_B
        else:
            # 旧模式：只考虑 A/B 供给不足的相对差异。
            shortage_shift_for_A = theta * (shortage_B - shortage_A)
    else:
        shortage_shift_for_A = 0.0

    # ------------------------------------------------------------
    # 服务质量
    # ------------------------------------------------------------

    quality_base_scale = getattr(p, "quality_base_effect_scale", 1.0)
    quality_stock_scale = getattr(p, "quality_stock_effect_scale", 1.0)

    dq_u = quality_base_scale * getattr(p, "dq_u_base", 0.0) + quality_stock_scale * q_u_stock
    dq_m = quality_base_scale * getattr(p, "dq_m_base", 0.0) + quality_stock_scale * q_m_stock

    # ------------------------------------------------------------
    # Logit 选择概率
    # ------------------------------------------------------------

    # 用户侧吸引力差异：A 相对 B
    D_u = (
        dq_u
        + p.alpha * (2.0 * y - 1.0)
        + p.eta_u * (2.0 * x - 1.0)
        + net_ds_u
        - p.dp_u
        + shortage_shift_for_A
    )

    # 商户侧吸引力差异：A 相对 B
    D_m = (
        dq_m
        + p.beta * (2.0 * x - 1.0)
        + p.eta_m * (2.0 * y - 1.0)
        + net_ds_m
        - p.dc_m
    )

    P_u = sigmoid(p.lambda_u * D_u)
    P_m = sigmoid(p.lambda_m * D_m)

    dxdt = p.r_u * (P_u - x)
    dydt = p.r_m * (P_m - y)

    # 服务质量状态：A 相对 B 的质量优势
    dq_u_dt = p.invest_eff_u * net_inv_u - p.quality_decay * q_u_stock
    dq_m_dt = p.invest_eff_m * net_inv_m - p.quality_decay * q_m_stock

    # ------------------------------------------------------------
    # 收益与成本
    # ------------------------------------------------------------

    u_A = x
    m_A = y
    u_B = 1.0 - x
    m_B = 1.0 - y

    if getattr(p, "use_report_profit", True):
        # 报告口径：
        # pi_B = mu * u_B * m_B - s_u_B - s_m_B - I_B
        mu = getattr(p, "profit_mu", 5.0)

        revenue_rate_A = mu * u_A * m_A
        revenue_rate_B = mu * u_B * m_B

        subsidy_cost_rate_A = max(ds_u_A, 0.0) + max(ds_m_A, 0.0)
        subsidy_cost_rate_B = max(ds_u_B, 0.0) + max(ds_m_B, 0.0)

        invest_cost_rate_A = max(inv_u_A, 0.0) + max(inv_m_A, 0.0)
        invest_cost_rate_B = max(inv_u_B, 0.0) + max(inv_m_B, 0.0)

    else:
        # 旧线性收入口径，保留兼容。
        revenue_rate_A = p.revenue_user * u_A + p.revenue_merchant * m_A
        revenue_rate_B = p.revenue_user * u_B + p.revenue_merchant * m_B

        subsidy_cost_rate_A = (
            p.cost_user_subsidy * max(ds_u_A, 0.0) * u_A
            + p.cost_merchant_subsidy * max(ds_m_A, 0.0) * m_A
        )
        subsidy_cost_rate_B = (
            p.cost_user_subsidy * max(ds_u_B, 0.0) * u_B
            + p.cost_merchant_subsidy * max(ds_m_B, 0.0) * m_B
        )

        invest_cost_rate_A = (
            p.cost_invest_u * max(inv_u_A, 0.0)
            + p.cost_invest_m * max(inv_m_A, 0.0)
        )
        invest_cost_rate_B = (
            p.cost_invest_u * max(inv_u_B, 0.0)
            + p.cost_invest_m * max(inv_m_B, 0.0)
        )

    total_cost_rate_A = subsidy_cost_rate_A + invest_cost_rate_A
    total_cost_rate_B = subsidy_cost_rate_B + invest_cost_rate_B

    profit_rate_A = revenue_rate_A - total_cost_rate_A
    profit_rate_B = revenue_rate_B - total_cost_rate_B

    return {
        "derivative": np.array([dxdt, dydt, dq_u_dt, dq_m_dt], dtype=float),
        "D_u": D_u,
        "D_m": D_m,
        "P_u": P_u,
        "P_m": P_m,

        # 兼容旧输出：ds_u/ds_m/inv_u/inv_m 默认记录平台 B 策略强度。
        "ds_u": ds_u_B,
        "ds_m": ds_m_B,
        "inv_u": inv_u_B,
        "inv_m": inv_m_B,

        "ds_u_A": ds_u_A,
        "ds_m_A": ds_m_A,
        "inv_u_A": inv_u_A,
        "inv_m_A": inv_m_A,

        "ds_u_B": ds_u_B,
        "ds_m_B": ds_m_B,
        "inv_u_B": inv_u_B,
        "inv_m_B": inv_m_B,

        "shortage_A": shortage_A,
        "shortage_B": shortage_B,

        "revenue_rate_A": revenue_rate_A,
        "revenue_rate_B": revenue_rate_B,
        "subsidy_cost_rate_A": subsidy_cost_rate_A,
        "subsidy_cost_rate_B": subsidy_cost_rate_B,
        "invest_cost_rate_A": invest_cost_rate_A,
        "invest_cost_rate_B": invest_cost_rate_B,
        "total_cost_rate_A": total_cost_rate_A,
        "total_cost_rate_B": total_cost_rate_B,
        "profit_rate_A": profit_rate_A,
        "profit_rate_B": profit_rate_B,

        # 兼容旧接口：默认把这些指标解释为挑战者 B 的指标。
        "revenue_rate": revenue_rate_B,
        "subsidy_cost_rate": subsidy_cost_rate_B,
        "invest_cost_rate": invest_cost_rate_B,
        "total_cost_rate": total_cost_rate_B,
        "profit_rate": profit_rate_B,
    }


def _result_keys():
    return [
        "t", "x", "y", "q_u", "q_m",
        "D_u", "D_m", "P_u", "P_m",
        "ds_u", "ds_m", "inv_u", "inv_m",
        "ds_u_A", "ds_m_A", "inv_u_A", "inv_m_A",
        "ds_u_B", "ds_m_B", "inv_u_B", "inv_m_B",
        "shortage_A", "shortage_B",

        "revenue_rate", "subsidy_cost_rate", "invest_cost_rate", "total_cost_rate", "profit_rate",
        "revenue_rate_A", "subsidy_cost_rate_A", "invest_cost_rate_A", "total_cost_rate_A", "profit_rate_A",
        "revenue_rate_B", "subsidy_cost_rate_B", "invest_cost_rate_B", "total_cost_rate_B", "profit_rate_B",

        "cum_revenue", "cum_subsidy_cost", "cum_invest_cost", "cum_total_cost", "cum_profit", "cum_discounted_profit",
        "cum_revenue_A", "cum_subsidy_cost_A", "cum_invest_cost_A", "cum_total_cost_A", "cum_profit_A", "cum_discounted_profit_A",
        "cum_revenue_B", "cum_subsidy_cost_B", "cum_invest_cost_B", "cum_total_cost_B", "cum_profit_B", "cum_discounted_profit_B",
    ]


def _clip_state(state, p):
    state = np.asarray(state, dtype=float).copy()
    state[0] = np.clip(state[0], 0.0, 1.0)
    state[1] = np.clip(state[1], 0.0, 1.0)

    qmax = getattr(p, "qmax", np.inf)
    state[2] = np.clip(state[2], -qmax, qmax)
    state[3] = np.clip(state[3], -qmax, qmax)

    return state


def _fill_common_result(result, i, t, state, terms):
    result["t"][i] = t
    result["x"][i] = state[0]
    result["y"][i] = state[1]
    result["q_u"][i] = state[2]
    result["q_m"][i] = state[3]

    for k in [
        "D_u", "D_m", "P_u", "P_m",
        "ds_u", "ds_m", "inv_u", "inv_m",
        "ds_u_A", "ds_m_A", "inv_u_A", "inv_m_A",
        "ds_u_B", "ds_m_B", "inv_u_B", "inv_m_B",
        "shortage_A", "shortage_B",
        "revenue_rate", "subsidy_cost_rate", "invest_cost_rate", "total_cost_rate", "profit_rate",
        "revenue_rate_A", "subsidy_cost_rate_A", "invest_cost_rate_A", "total_cost_rate_A", "profit_rate_A",
        "revenue_rate_B", "subsidy_cost_rate_B", "invest_cost_rate_B", "total_cost_rate_B", "profit_rate_B",
    ]:
        result[k][i] = terms[k]


def simulate_path(x0, y0, p, T=80.0, dt=0.03, policy=None):
    """
    显式欧拉法模拟单条市场演化轨迹。
    """
    from src.policies import zero_policy

    if policy is None:
        policy = zero_policy

    n_steps = int(T / dt)
    t_values = np.linspace(0, T, n_steps + 1)

    result = {k: np.zeros(n_steps + 1) for k in _result_keys()}

    state = np.array([x0, y0, 0.0, 0.0], dtype=float)

    cum = {
        "A": {"revenue": 0.0, "subsidy": 0.0, "invest": 0.0, "total": 0.0, "profit": 0.0, "discounted_profit": 0.0},
        "B": {"revenue": 0.0, "subsidy": 0.0, "invest": 0.0, "total": 0.0, "profit": 0.0, "discounted_profit": 0.0},
    }

    for i, t in enumerate(t_values):
        state = _clip_state(state, p)
        terms = model_terms(t, state, p, policy)

        _fill_common_result(result, i, t, state, terms)

        # 兼容旧接口：cum_* 默认记录平台 B。
        result["cum_revenue"][i] = cum["B"]["revenue"]
        result["cum_subsidy_cost"][i] = cum["B"]["subsidy"]
        result["cum_invest_cost"][i] = cum["B"]["invest"]
        result["cum_total_cost"][i] = cum["B"]["total"]
        result["cum_profit"][i] = cum["B"]["profit"]
        result["cum_discounted_profit"][i] = cum["B"]["discounted_profit"]

        for platform in ["A", "B"]:
            result[f"cum_revenue_{platform}"][i] = cum[platform]["revenue"]
            result[f"cum_subsidy_cost_{platform}"][i] = cum[platform]["subsidy"]
            result[f"cum_invest_cost_{platform}"][i] = cum[platform]["invest"]
            result[f"cum_total_cost_{platform}"][i] = cum[platform]["total"]
            result[f"cum_profit_{platform}"][i] = cum[platform]["profit"]
            result[f"cum_discounted_profit_{platform}"][i] = cum[platform]["discounted_profit"]

        if i < n_steps:
            state = state + dt * terms["derivative"]
            state = _clip_state(state, p)

            discount = getattr(p, "discount", 1.0)
            discount_weight = discount ** t

            for platform in ["A", "B"]:
                cum[platform]["revenue"] += terms[f"revenue_rate_{platform}"] * dt
                cum[platform]["subsidy"] += terms[f"subsidy_cost_rate_{platform}"] * dt
                cum[platform]["invest"] += terms[f"invest_cost_rate_{platform}"] * dt
                cum[platform]["total"] += terms[f"total_cost_rate_{platform}"] * dt
                cum[platform]["profit"] += terms[f"profit_rate_{platform}"] * dt
                cum[platform]["discounted_profit"] += discount_weight * terms[f"profit_rate_{platform}"] * dt

    return result


def simulate_path_rk45(
    x0,
    y0,
    p,
    T=80.0,
    dt=0.03,
    policy=None,
    rtol=1e-8,
    atol=1e-10,
    max_step=None,
):
    """
    使用 scipy.integrate.solve_ivp 的 RK45 方法模拟单条市场演化轨迹。
    返回格式与 simulate_path 保持一致。
    """
    from scipy.integrate import solve_ivp
    from src.policies import zero_policy

    if policy is None:
        policy = zero_policy

    n_steps = int(T / dt)
    t_values = np.linspace(0, T, n_steps + 1)

    result = {k: np.zeros(n_steps + 1) for k in _result_keys()}

    def rhs(t, z):
        z = _clip_state(z, p)
        terms = model_terms(t, z, p, policy)
        return terms["derivative"]

    if max_step is None:
        max_step = dt

    sol = solve_ivp(
        rhs,
        t_span=(0.0, T),
        y0=np.array([x0, y0, 0.0, 0.0], dtype=float),
        method="RK45",
        t_eval=t_values,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"RK45 求解失败：{sol.message}")

    states = np.vstack([_clip_state(z, p) for z in sol.y.T])

    for i, t in enumerate(t_values):
        state = states[i]
        terms = model_terms(t, state, p, policy)
        _fill_common_result(result, i, t, state, terms)

    dt_array = np.diff(t_values)

    def cumulative_trapezoid(rate):
        out = np.zeros_like(rate)
        out[1:] = np.cumsum(0.5 * (rate[:-1] + rate[1:]) * dt_array)
        return out

    for platform in ["A", "B"]:
        result[f"cum_revenue_{platform}"] = cumulative_trapezoid(result[f"revenue_rate_{platform}"])
        result[f"cum_subsidy_cost_{platform}"] = cumulative_trapezoid(result[f"subsidy_cost_rate_{platform}"])
        result[f"cum_invest_cost_{platform}"] = cumulative_trapezoid(result[f"invest_cost_rate_{platform}"])
        result[f"cum_total_cost_{platform}"] = cumulative_trapezoid(result[f"total_cost_rate_{platform}"])
        result[f"cum_profit_{platform}"] = cumulative_trapezoid(result[f"profit_rate_{platform}"])

        discount = getattr(p, "discount", 1.0)
        discounted_profit_rate = (discount ** t_values) * result[f"profit_rate_{platform}"]
        result[f"cum_discounted_profit_{platform}"] = cumulative_trapezoid(discounted_profit_rate)

    # 兼容旧接口：cum_* 默认记录平台 B。
    result["cum_revenue"] = result["cum_revenue_B"].copy()
    result["cum_subsidy_cost"] = result["cum_subsidy_cost_B"].copy()
    result["cum_invest_cost"] = result["cum_invest_cost_B"].copy()
    result["cum_total_cost"] = result["cum_total_cost_B"].copy()
    result["cum_profit"] = result["cum_profit_B"].copy()
    result["cum_discounted_profit"] = result["cum_discounted_profit_B"].copy()

    return result


def simulate_grid_initial(p, alpha, beta, grid_n=51, T=70.0, dt=0.05):
    """
    对不同初始份额 x(0), y(0) 做网格模拟。
    用于分析初始规模和网络效应耦合。
    """
    grid = np.linspace(0.05, 0.95, grid_n)
    X, Y = np.meshgrid(grid, grid)

    steps = int(T / dt)

    for _ in range(steps):
        quality_base_scale = getattr(p, "quality_base_effect_scale", 1.0)

        D_u = (
            quality_base_scale * p.dq_u_base
            + alpha * (2 * Y - 1)
            + p.eta_u * (2 * X - 1)
        )

        D_m = (
            quality_base_scale * p.dq_m_base
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


def simulate_alpha_beta_scan(p, x0, y0, alpha_values, beta_values, T=70.0, dt=0.05):
    """
    扫描 alpha 和 beta 不相等的情况。
    横轴 alpha，纵轴 beta，颜色表示最终 x*。
    """
    A, B = np.meshgrid(alpha_values, beta_values)

    X = np.ones_like(A) * x0
    Y = np.ones_like(A) * y0

    steps = int(T / dt)

    for _ in range(steps):
        quality_base_scale = getattr(p, "quality_base_effect_scale", 1.0)

        D_u = (
            quality_base_scale * p.dq_u_base
            + A * (2 * Y - 1)
            + p.eta_u * (2 * X - 1)
        )

        D_m = (
            quality_base_scale * p.dq_m_base
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