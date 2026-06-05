from __future__ import annotations

import math

import numpy as np
from mesa import Agent


def logit_probability(v_a: float, v_b: float, gamma: float) -> float:
    z = float(np.clip(gamma * (v_a - v_b), -700.0, 700.0))
    return 1.0 / (1.0 + math.exp(-z))


class BasePlatformAgent(Agent):
    """Common helper for user and merchant agents."""

    side: str

    def __init__(self, model, initial_platform: str) -> None:
        super().__init__(model)
        self.platform = initial_platform

    def switch_to(self, platform: str) -> None:
        if platform not in {"A", "B"}:
            raise ValueError("platform must be 'A' or 'B'")
        self.platform = platform


class UserAgent(BasePlatformAgent):
    side = "user"

    def step(self) -> None:
        params = self.model.params
        u_a = self.model.user_share_a
        m_a = self.model.merchant_share_a

        eps_a = self.model.random.gauss(0.0, params.sigmaU)
        eps_b = self.model.random.gauss(0.0, params.sigmaU)

        v_a = params.qAU + params.alpha * m_a + params.bAU - params.pAU + params.sU * (1.0 if self.platform == "A" else 0.0) + eps_a
        v_b = params.qBU + params.alpha * (1.0 - m_a) + params.bBU - params.pBU + params.sU * (1.0 if self.platform == "B" else 0.0) + eps_b

        p_choose_a = logit_probability(v_a, v_b, params.gammaU)
        self.switch_to("A" if self.model.random.random() < p_choose_a else "B")


class MerchantAgent(BasePlatformAgent):
    side = "merchant"

    def step(self) -> None:
        params = self.model.params
        u_a = self.model.user_share_a
        m_a = self.model.merchant_share_a

        eps_a = self.model.random.gauss(0.0, params.sigmaM)
        eps_b = self.model.random.gauss(0.0, params.sigmaM)

        v_a = params.qAM + params.beta * u_a + params.bAM - params.rA + params.sM * (1.0 if self.platform == "A" else 0.0) + eps_a
        v_b = params.qBM + params.beta * (1.0 - u_a) + params.bBM - params.rB + params.sM * (1.0 if self.platform == "B" else 0.0) + eps_b

        p_choose_a = logit_probability(v_a, v_b, params.gammaM)
        self.switch_to("A" if self.model.random.random() < p_choose_a else "B")
