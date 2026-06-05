from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class PlatformParams:
    alpha: float = 1.0
    beta: float = 1.0
    sU: float = 0.2
    sM: float = 0.2
    gammaU: float = 2.0
    gammaM: float = 2.0
    lambdaU: float = 1.0
    lambdaM: float = 1.0
    qAU: float = 0.0
    qBU: float = 0.0
    qAM: float = 0.0
    qBM: float = 0.0
    bAU: float = 0.0
    bBU: float = 0.0
    bAM: float = 0.0
    bBM: float = 0.0
    pAU: float = 0.0
    pBU: float = 0.0
    rA: float = 0.0
    rB: float = 0.0

    def with_updates(self, **kwargs: float) -> "PlatformParams":
        return replace(self, **kwargs)


def _logit_probability(v_a: float, v_b: float, gamma: float) -> float:
    z = np.clip(gamma * (v_a - v_b), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-z))


def platform_dynamics(t: float, y: np.ndarray, params: PlatformParams) -> list[float]:
    u, m = np.clip(y, 0.0, 1.0)

    v_au = params.qAU + params.alpha * m + params.bAU - params.pAU + params.sU * u
    v_bu = params.qBU + params.alpha * (1.0 - m) + params.bBU - params.pBU + params.sU * (1.0 - u)
    v_am = params.qAM + params.beta * u + params.bAM - params.rA + params.sM * m
    v_bm = params.qBM + params.beta * (1.0 - u) + params.bBM - params.rB + params.sM * (1.0 - m)

    p_au = _logit_probability(v_au, v_bu, params.gammaU)
    p_am = _logit_probability(v_am, v_bm, params.gammaM)

    du = params.lambdaU * (p_au - u)
    dm = params.lambdaM * (p_am - m)
    return [du, dm]


def run_simulation(
    params: PlatformParams,
    u0: float = 0.5,
    m0: float = 0.5,
    T: float = 50.0,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_eval = np.linspace(0.0, T, n_points)
    sol = solve_ivp(
        fun=lambda t, y: platform_dynamics(t, y, params),
        t_span=(0.0, T),
        y0=[u0, m0],
        t_eval=t_eval,
        rtol=1e-8,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    u = np.clip(sol.y[0], 0.0, 1.0)
    m = np.clip(sol.y[1], 0.0, 1.0)
    return sol.t, u, m


def run_final_state(
    params: PlatformParams,
    u0: float,
    m0: float,
    T: float = 50.0,
) -> tuple[float, float]:
    sol = solve_ivp(
        fun=lambda t, y: platform_dynamics(t, y, params),
        t_span=(0.0, T),
        y0=[u0, m0],
        t_eval=[T],
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")
    u_inf, m_inf = sol.y[:, -1]
    return float(np.clip(u_inf, 0.0, 1.0)), float(np.clip(m_inf, 0.0, 1.0))
