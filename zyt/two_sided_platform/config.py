"""Default parameters for the two-sided platform dynamic model."""

BASE_CONFIG = {
    "T": 200,
    "dt": 0.05,
    "alpha": 0.8,
    "beta": 0.8,
    "s_U": 0.2,
    "s_M": 0.2,
    "gamma_U": 2.0,
    "gamma_M": 2.0,
    "lambda_U": 1.0,
    "lambda_M": 1.0,
    "q_A": 1.0,
    "q_B": 1.0,
    "r_A": 1.0,
    "r_B": 1.0,
    "p_A": 0.0,
    "p_B": 0.0,
    "c_A": 0.0,
    "c_B": 0.0,
    "eps": 1e-6,
    "rho": 10.0,
    "user_scale": 1000.0,
    "merchant_scale": 50.0,
    "theta": 1.0,
    "mu": 10.0,
    "discount": 0.98,
    "lambda_q": 0.05,
    "quality_decay": 0.01,
    "q_max": 3.0,
}

NETWORK_EFFECT_LEVELS = [
    {"label": "weak", "alpha": 0.3, "beta": 0.3},
    {"label": "medium", "alpha": 0.8, "beta": 0.8},
    {"label": "strong", "alpha": 1.5, "beta": 1.5},
]

INITIAL_SHARE_SCENARIOS = [
    {"label": "fair", "x0": 0.50, "y0": 0.50},
    {"label": "A_small_lead", "x0": 0.55, "y0": 0.55},
    {"label": "A_medium_lead", "x0": 0.65, "y0": 0.65},
    {"label": "A_large_lead", "x0": 0.75, "y0": 0.75},
    {"label": "A_near_lock_in", "x0": 0.85, "y0": 0.85},
]

QUALITY_ADVANTAGE_SCENARIOS = [
    {
        "label": f"delta_{idx / 100:.2f}".replace(".", "p"),
        "delta_q": idx / 100,
        "delta_r": idx / 100,
    }
    for idx in range(0, 151)
]

STAGE234_NETWORK_LEVELS = [
    {"label": "medium", "alpha": 0.8, "beta": 0.8},
    {"label": "strong", "alpha": 1.5, "beta": 1.5},
]

SUBSIDY_BUDGETS = [0.0, 0.1, 0.2, 0.4, 1.2]

CONGESTION_THETA_LEVELS = [
    {"label": "mild", "theta": 0.5},
    {"label": "standard", "theta": 1.0},
    {"label": "severe", "theta": 2.0},
]
