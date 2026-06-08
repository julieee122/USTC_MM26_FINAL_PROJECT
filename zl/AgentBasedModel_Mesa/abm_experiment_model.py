from __future__ import annotations

import math
import random
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


def logit_probability(v_a: float, v_b: float, gamma: float) -> float:
    z = float(np.clip(gamma * (v_a - v_b), -700.0, 700.0))
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class ExperimentParams:
    alpha: float = 2.5
    beta: float = 2.5
    gamma_user: float = 1.8
    gamma_merchant: float = 1.8
    q_a_user: float = 0.0
    q_b_user: float = 0.0
    q_a_merchant: float = 0.0
    q_b_merchant: float = 0.0
    p_a_user: float = 0.0
    p_b_user: float = 0.0
    r_a: float = 0.0
    r_b: float = 0.0
    sigma_user: float = 0.2
    sigma_merchant: float = 0.2
    heterogeneity_sigma: float = 0.5
    subsidy_intensity: float = 0.7
    subsidy_budget: float = 180.0
    max_subsidy_per_agent: float = 2.0
    multi_home_cost: float = 0.5
    multi_home_visibility: float = 0.85
    social_eta: float = 0.0
    user_inertia_mean: float = 0.25
    merchant_inertia_mean: float = 0.25
    initial_a_inertia_boost: float = 0.0


@dataclass
class UserState:
    uid: int
    platform: str
    segment: str
    price_sensitivity: float
    quality_sensitivity: float
    subsidy_sensitivity: float
    inertia: float
    influence: float = 1.0


@dataclass
class MerchantState:
    mid: int
    status: str
    segment: str
    commission_sensitivity: float
    traffic_sensitivity: float
    subsidy_sensitivity: float
    inertia: float
    value_weight: float = 1.0
    can_multi_home: bool = False


class PlatformCompetitionABM:
    """ABM engine for high-value two-sided platform experiments."""

    def __init__(
        self,
        *,
        n_users: int = 500,
        n_merchants: int = 250,
        u0: float = 0.5,
        m0: float = 0.5,
        params: ExperimentParams | None = None,
        seed: int | None = None,
        subsidy_policy: str = "none",
        allow_multi_home: bool = False,
        multi_home_share: float = 1.0,
        network_type: str = "none",
        cold_start_strategy: str = "none",
        seed_users: int = 0,
        seed_merchants: int = 0,
    ) -> None:
        self.params = params or ExperimentParams()
        self.rng = random.Random(seed)
        self.n_users = n_users
        self.n_merchants = n_merchants
        self.subsidy_policy = subsidy_policy
        self.allow_multi_home = allow_multi_home
        self.multi_home_share = multi_home_share
        self.network_type = network_type
        self.cold_start_strategy = cold_start_strategy
        self.seed_users = seed_users
        self.seed_merchants = seed_merchants
        self.total_subsidy_spend = 0.0
        self.seeded_user_ids: set[int] = set()
        self.seeded_merchant_ids: set[int] = set()

        self.users: list[UserState] = []
        self.merchants: list[MerchantState] = []
        self.user_graph = self._build_user_graph(network_type)
        self._create_users(u0)
        self._create_merchants(m0)
        self._apply_cold_start_strategy()

    def _positive_normal(self, mean: float, sigma: float, lower: float = 0.05, upper: float = 3.0) -> float:
        return float(np.clip(self.rng.gauss(mean, sigma), lower, upper))

    def _choose_segment(self, segments: list[str]) -> str:
        return segments[self.rng.randrange(len(segments))]

    def _create_users(self, u0: float) -> None:
        n_a = round(self.n_users * u0)
        sigma = self.params.heterogeneity_sigma
        segments = ["price_sensitive", "quality_sensitive", "subsidy_sensitive", "inertial"]
        for uid in range(self.n_users):
            segment = self._choose_segment(segments)
            price = self._positive_normal(1.0, sigma)
            quality = self._positive_normal(1.0, sigma)
            subsidy = self._positive_normal(1.0, sigma)
            inertia = self._positive_normal(self.params.user_inertia_mean, 0.25 * sigma, lower=0.0, upper=2.0)
            if segment == "price_sensitive":
                price *= 1.5
            elif segment == "quality_sensitive":
                quality *= 1.5
            elif segment == "subsidy_sensitive":
                subsidy *= 1.6
            elif segment == "inertial":
                inertia *= 2.0
            influence = 1.0 + self.user_graph.degree(uid) / max(1, self.n_users - 1) if self.user_graph else 1.0
            platform = "A" if uid < n_a else "B"
            if platform == "A":
                inertia += self.params.initial_a_inertia_boost
            self.users.append(
                UserState(
                    uid=uid,
                    platform=platform,
                    segment=segment,
                    price_sensitivity=price,
                    quality_sensitivity=quality,
                    subsidy_sensitivity=subsidy,
                    inertia=inertia,
                    influence=influence,
                )
            )

    def _create_merchants(self, m0: float) -> None:
        n_a = round(self.n_merchants * m0)
        sigma = self.params.heterogeneity_sigma
        segments = ["traffic_sensitive", "cost_sensitive", "subsidy_sensitive", "inertial"]
        multi_capable = set(self.rng.sample(range(self.n_merchants), round(self.n_merchants * self.multi_home_share)))
        for mid in range(self.n_merchants):
            segment = self._choose_segment(segments)
            commission = self._positive_normal(1.0, sigma)
            traffic = self._positive_normal(1.0, sigma)
            subsidy = self._positive_normal(1.0, sigma)
            inertia = self._positive_normal(self.params.merchant_inertia_mean, 0.25 * sigma, lower=0.0, upper=2.0)
            value_weight = self._positive_normal(1.0, sigma, lower=0.2, upper=4.0)
            if segment == "traffic_sensitive":
                traffic *= 1.5
                value_weight *= 1.25
            elif segment == "cost_sensitive":
                commission *= 1.5
            elif segment == "subsidy_sensitive":
                subsidy *= 1.6
            elif segment == "inertial":
                inertia *= 2.0
            status = "A_only" if mid < n_a else "B_only"
            if status == "A_only":
                inertia += self.params.initial_a_inertia_boost
            self.merchants.append(
                MerchantState(
                    mid=mid,
                    status=status,
                    segment=segment,
                    commission_sensitivity=commission,
                    traffic_sensitivity=traffic,
                    subsidy_sensitivity=subsidy,
                    inertia=inertia,
                    value_weight=value_weight,
                    can_multi_home=mid in multi_capable,
                )
            )

    def _build_user_graph(self, network_type: str) -> nx.Graph | None:
        if network_type == "none":
            return None
        if network_type == "random":
            return nx.erdos_renyi_graph(self.n_users, 0.025, seed=self.rng.randrange(10**9))
        if network_type == "small_world":
            return nx.watts_strogatz_graph(self.n_users, k=8, p=0.12, seed=self.rng.randrange(10**9))
        if network_type == "scale_free":
            return nx.barabasi_albert_graph(self.n_users, m=3, seed=self.rng.randrange(10**9))
        if network_type == "community":
            sizes = [self.n_users // 4] * 4
            sizes[-1] += self.n_users - sum(sizes)
            return nx.random_partition_graph(sizes, 0.08, 0.006, seed=self.rng.randrange(10**9))
        raise ValueError(f"unknown network_type: {network_type}")

    def _apply_cold_start_strategy(self) -> None:
        if self.cold_start_strategy == "none":
            return

        if self.seed_users > 0:
            if self.cold_start_strategy == "high_influence_user" and self.user_graph:
                chosen_users = sorted(self.users, key=lambda u: self.user_graph.degree(u.uid), reverse=True)[: self.seed_users]
            else:
                chosen_users = self.rng.sample(self.users, min(self.seed_users, len(self.users)))
            for user in chosen_users:
                user.platform = "A"
                user.inertia += 2.0
                self.seeded_user_ids.add(user.uid)

        if self.seed_merchants > 0:
            if self.cold_start_strategy == "key_merchant_seed":
                chosen_merchants = sorted(self.merchants, key=lambda m: m.value_weight, reverse=True)[: self.seed_merchants]
            else:
                chosen_merchants = self.rng.sample(self.merchants, min(self.seed_merchants, len(self.merchants)))
            for merchant in chosen_merchants:
                merchant.status = "A_only"
                merchant.inertia += 2.0
                merchant.value_weight *= 1.3
                self.seeded_merchant_ids.add(merchant.mid)

    @property
    def user_share_a(self) -> float:
        return sum(user.platform == "A" for user in self.users) / self.n_users

    @property
    def merchant_presence_a(self) -> float:
        visible = sum(
            1.0 if m.status == "A_only" else self.params.multi_home_visibility if m.status == "multi_home" else 0.0
            for m in self.merchants
        )
        return visible / self.n_merchants

    @property
    def merchant_presence_b(self) -> float:
        visible = sum(
            1.0 if m.status == "B_only" else self.params.multi_home_visibility if m.status == "multi_home" else 0.0
            for m in self.merchants
        )
        return visible / self.n_merchants

    @property
    def merchant_a_only_share(self) -> float:
        return sum(m.status == "A_only" for m in self.merchants) / self.n_merchants

    @property
    def merchant_b_only_share(self) -> float:
        return sum(m.status == "B_only" for m in self.merchants) / self.n_merchants

    @property
    def merchant_multi_share(self) -> float:
        return sum(m.status == "multi_home" for m in self.merchants) / self.n_merchants

    @property
    def combined_share_a(self) -> float:
        return 0.5 * (self.user_share_a + self.merchant_presence_a)

    def _neighbor_share_a(self, user: UserState, platform_by_uid: dict[int, str] | None = None) -> float:
        if not self.user_graph:
            return 0.0
        neighbors = list(self.user_graph.neighbors(user.uid))
        if not neighbors:
            return self.user_share_a
        if platform_by_uid is None:
            platform_by_uid = {u.uid: u.platform for u in self.users}
        return sum(platform_by_uid[n] == "A" for n in neighbors) / len(neighbors)

    def _targeted_users(self) -> set[int]:
        if self.subsidy_policy == "uniform":
            return {u.uid for u in self.users}
        if self.subsidy_policy == "random":
            candidates = [u.uid for u in self.users]
            target_count = max(1, len(candidates) // 3)
            return set(self.rng.sample(candidates, target_count))
        if self.subsidy_policy in {"swing_user", "user_targeted", "two_sided_targeted", "dynamic_targeted"}:
            users = [u for u in self.users if u.platform == "B"]
            users.sort(key=lambda u: (u.subsidy_sensitivity - 0.35 * u.inertia, u.price_sensitivity), reverse=True)
            return {u.uid for u in users[: max(1, len(users) // 3)]}
        return set()

    def _targeted_merchants(self) -> set[int]:
        if self.subsidy_policy == "uniform":
            return {m.mid for m in self.merchants}
        if self.subsidy_policy == "random":
            candidates = [m.mid for m in self.merchants]
            target_count = max(1, len(candidates) // 3)
            return set(self.rng.sample(candidates, target_count))
        if self.subsidy_policy in {"key_merchant", "merchant_targeted", "two_sided_targeted", "dynamic_targeted"}:
            merchants = [m for m in self.merchants if m.status != "A_only"]
            merchants.sort(key=lambda m: (m.traffic_sensitivity * m.value_weight + m.subsidy_sensitivity - 0.35 * m.inertia), reverse=True)
            return {m.mid for m in merchants[: max(1, len(merchants) // 3)]}
        return set()

    def _subsidy_sets(self) -> tuple[set[int], set[int], float, float]:
        if self.subsidy_policy == "none":
            return set(), set(), 0.0, 0.0
        user_targets = self._targeted_users()
        merchant_targets = self._targeted_merchants()
        if self.subsidy_policy == "dynamic_targeted":
            if self.combined_share_a < 0.4:
                user_targets = set()
            elif self.combined_share_a > 0.6:
                user_targets = {u.uid for u in self.users if u.platform == "A" and u.subsidy_sensitivity > 1.0}
                merchant_targets = {m.mid for m in self.merchants if m.status == "A_only" and m.subsidy_sensitivity > 1.0}
        target_count = len(user_targets) + len(merchant_targets)
        if target_count == 0:
            return user_targets, merchant_targets, 0.0, 0.0
        subsidy_amount = min(self.params.max_subsidy_per_agent, self.params.subsidy_budget / target_count)
        spend = subsidy_amount * target_count
        return user_targets, merchant_targets, subsidy_amount, spend

    def step(self) -> dict[str, float]:
        user_targets, merchant_targets, subsidy_amount, spend = self._subsidy_sets()
        self.total_subsidy_spend += spend

        users = self.users[:]
        merchants = self.merchants[:]
        self.rng.shuffle(users)
        self.rng.shuffle(merchants)

        m_a = self.merchant_presence_a
        m_b = self.merchant_presence_b
        u_a = self.user_share_a
        platform_by_uid = {u.uid: u.platform for u in self.users}

        for user in users:
            subsidy_a = subsidy_amount if user.uid in user_targets else 0.0
            neighbor_share_a = self._neighbor_share_a(user, platform_by_uid)
            social_a = self.params.social_eta * neighbor_share_a
            social_b = self.params.social_eta * (1.0 - neighbor_share_a)
            v_a = (
                user.quality_sensitivity * self.params.q_a_user
                + self.params.alpha * m_a
                + user.subsidy_sensitivity * subsidy_a
                - user.price_sensitivity * self.params.p_a_user
                + user.inertia * (1.0 if user.platform == "A" else 0.0)
                + social_a
                + self.rng.gauss(0.0, self.params.sigma_user)
            )
            v_b = (
                user.quality_sensitivity * self.params.q_b_user
                + self.params.alpha * m_b
                - user.price_sensitivity * self.params.p_b_user
                + user.inertia * (1.0 if user.platform == "B" else 0.0)
                + social_b
                + self.rng.gauss(0.0, self.params.sigma_user)
            )
            user.platform = "A" if self.rng.random() < logit_probability(v_a, v_b, self.params.gamma_user) else "B"

        u_a = self.user_share_a
        for merchant in merchants:
            subsidy_a = subsidy_amount if merchant.mid in merchant_targets else 0.0
            v_a = (
                merchant.traffic_sensitivity * self.params.beta * u_a
                + self.params.q_a_merchant
                + merchant.subsidy_sensitivity * subsidy_a
                - merchant.commission_sensitivity * self.params.r_a
                + merchant.inertia * (1.0 if merchant.status == "A_only" else 0.0)
                + self.rng.gauss(0.0, self.params.sigma_merchant)
            )
            v_b = (
                merchant.traffic_sensitivity * self.params.beta * (1.0 - u_a)
                + self.params.q_b_merchant
                - merchant.commission_sensitivity * self.params.r_b
                + merchant.inertia * (1.0 if merchant.status == "B_only" else 0.0)
                + self.rng.gauss(0.0, self.params.sigma_merchant)
            )
            if self.allow_multi_home and merchant.can_multi_home:
                v_multi = (
                    0.85 * max(v_a, v_b)
                    + 0.45 * min(v_a, v_b)
                    + merchant.value_weight * 0.15
                    - self.params.multi_home_cost
                    + merchant.inertia * (1.0 if merchant.status == "multi_home" else 0.0)
                )
                values = [("A_only", v_a), ("B_only", v_b), ("multi_home", v_multi)]
                max_v = max(v for _, v in values)
                weights = [math.exp(np.clip(self.params.gamma_merchant * (v - max_v), -50, 50)) for _, v in values]
                pick = self.rng.random() * sum(weights)
                running = 0.0
                for (status, _), weight in zip(values, weights):
                    running += weight
                    if pick <= running:
                        merchant.status = status
                        break
            else:
                merchant.status = "A_only" if self.rng.random() < logit_probability(v_a, v_b, self.params.gamma_merchant) else "B_only"

        return self.collect_metrics()

    def collect_metrics(self) -> dict[str, float]:
        l_a = self.combined_share_a
        return {
            "u_A": self.user_share_a,
            "merchant_A_presence": self.merchant_presence_a,
            "merchant_B_presence": self.merchant_presence_b,
            "merchant_A_only": self.merchant_a_only_share,
            "merchant_B_only": self.merchant_b_only_share,
            "merchant_multi": self.merchant_multi_share,
            "L_A": l_a,
            "concentration": l_a * l_a + (1.0 - l_a) * (1.0 - l_a),
            "total_subsidy_spend": self.total_subsidy_spend,
        }

    def run(self, steps: int = 60) -> pd.DataFrame:
        rows = [self.collect_metrics() | {"step": 0}]
        for step in range(1, steps + 1):
            rows.append(self.step() | {"step": step})
        return pd.DataFrame(rows)

    def user_segment_distribution(self) -> pd.DataFrame:
        rows = []
        for segment in sorted({u.segment for u in self.users}):
            users = [u for u in self.users if u.segment == segment]
            rows.append(
                {
                    "segment": segment,
                    "share_on_A": sum(u.platform == "A" for u in users) / max(1, len(users)),
                    "count": len(users),
                }
            )
        return pd.DataFrame(rows)

    def merchant_segment_distribution(self) -> pd.DataFrame:
        rows = []
        for segment in sorted({m.segment for m in self.merchants}):
            merchants = [m for m in self.merchants if m.segment == segment]
            rows.append(
                {
                    "segment": segment,
                    "A_only_share": sum(m.status == "A_only" for m in merchants) / max(1, len(merchants)),
                    "B_only_share": sum(m.status == "B_only" for m in merchants) / max(1, len(merchants)),
                    "multi_share": sum(m.status == "multi_home" for m in merchants) / max(1, len(merchants)),
                    "count": len(merchants),
                }
            )
        return pd.DataFrame(rows)

