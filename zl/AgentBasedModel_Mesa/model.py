from __future__ import annotations

from mesa import Model
from mesa.datacollection import DataCollector

from agents import MerchantAgent, UserAgent
from params import ABMParams


def user_share_a(model: "TwoSidedPlatformModel") -> float:
    return model.user_share_a


def merchant_share_a(model: "TwoSidedPlatformModel") -> float:
    return model.merchant_share_a


def combined_share_a(model: "TwoSidedPlatformModel") -> float:
    return 0.5 * (model.user_share_a + model.merchant_share_a)


class TwoSidedPlatformModel(Model):
    """Mesa agent-based model for two-sided platform competition."""

    def __init__(
        self,
        n_users: int = 1000,
        n_merchants: int = 500,
        u0: float = 0.5,
        m0: float = 0.5,
        params: ABMParams | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.n_users = n_users
        self.n_merchants = n_merchants
        self.params = params or ABMParams()

        self.platform_agents: list[UserAgent | MerchantAgent] = []
        self.user_agents: list[UserAgent] = []
        self.merchant_agents: list[MerchantAgent] = []
        self.running = True

        self._create_agents(u0=u0, m0=m0)
        self.datacollector = DataCollector(
            model_reporters={
                "u_A": user_share_a,
                "m_A": merchant_share_a,
                "L_A": combined_share_a,
            }
        )
        self.datacollector.collect(self)

    @property
    def user_share_a(self) -> float:
        if not self.user_agents:
            return 0.0
        return sum(agent.platform == "A" for agent in self.user_agents) / len(self.user_agents)

    @property
    def merchant_share_a(self) -> float:
        if not self.merchant_agents:
            return 0.0
        return sum(agent.platform == "A" for agent in self.merchant_agents) / len(self.merchant_agents)

    def step(self) -> None:
        agents = self.platform_agents[:]
        self.random.shuffle(agents)
        for agent in agents:
            agent.step()
        self.datacollector.collect(self)

    def run_model(self, steps: int = 50):
        for _ in range(steps):
            self.step()
        return self.datacollector.get_model_vars_dataframe()

    def _create_agents(self, u0: float, m0: float) -> None:
        n_users_a = round(self.n_users * u0)
        n_merchants_a = round(self.n_merchants * m0)

        for i in range(self.n_users):
            platform = "A" if i < n_users_a else "B"
            agent = UserAgent(self, initial_platform=platform)
            self.user_agents.append(agent)
            self.platform_agents.append(agent)

        for i in range(self.n_merchants):
            platform = "A" if i < n_merchants_a else "B"
            agent = MerchantAgent(self, initial_platform=platform)
            self.merchant_agents.append(agent)
            self.platform_agents.append(agent)
