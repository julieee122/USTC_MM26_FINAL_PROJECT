import mesa
import numpy as np


class UserAgent(mesa.Agent):
    """
    用户智能体。
    """

    def __init__(
        self,
        model,
        user_type,
        platforms,
        w_q,
        w_m,
        w_s,
        w_p,
        w_c,
        inertia,
        can_multi_home,
        base_pref_A,
        base_pref_B,
    ):
        super().__init__(model)

        self.user_type = user_type
        self.platforms = set(platforms)

        self.w_q = w_q
        self.w_m = w_m
        self.w_s = w_s
        self.w_p = w_p
        self.w_c = w_c
        self.inertia = inertia

        self.can_multi_home = can_multi_home

        self.base_pref_A = base_pref_A
        self.base_pref_B = base_pref_B

        self.last_utility = 0.0

    def step(self):
        u_A = self.model.user_utility(self, "A")
        u_B = self.model.user_utility(self, "B")

        self.platforms = self.choose_platforms(u_A, u_B)

        if self.platforms == {"A"}:
            self.last_utility = u_A
        elif self.platforms == {"B"}:
            self.last_utility = u_B
        else:
            self.last_utility = 0.5 * (u_A + u_B)

    def choose_platforms(self, u_A, u_B):
        gap = abs(u_A - u_B)

        base_gap = getattr(self.model.params, "user_multi_home_gap", None)
        if base_gap is None:
            base_gap = getattr(self.model.params, "multi_home_gap", 0.8)

        if self.can_multi_home and gap <= base_gap:
            return {"A", "B"}

        if u_A >= u_B:
            return {"A"}
        return {"B"}


class MerchantAgent(mesa.Agent):
    """
    商户智能体。
    """

    def __init__(
        self,
        model,
        merchant_type,
        platforms,
        v_r,
        v_u,
        v_s,
        v_c,
        can_multi_home,
        base_pref_A,
        base_pref_B,
    ):
        super().__init__(model)

        self.merchant_type = merchant_type
        self.platforms = set(platforms)

        self.v_r = v_r
        self.v_u = v_u
        self.v_s = v_s
        self.v_c = v_c

        self.can_multi_home = can_multi_home

        self.base_pref_A = base_pref_A
        self.base_pref_B = base_pref_B

        self.last_utility = 0.0

    def step(self):
        u_A = self.model.merchant_utility(self, "A")
        u_B = self.model.merchant_utility(self, "B")

        self.platforms = self.choose_platforms(u_A, u_B)

        if self.platforms == {"A"}:
            self.last_utility = u_A
        elif self.platforms == {"B"}:
            self.last_utility = u_B
        else:
            self.last_utility = 0.5 * (u_A + u_B)

    def choose_platforms(self, u_A, u_B):
        gap = abs(u_A - u_B)

        k_M = getattr(self.model.params, "k_M", 0.0)
        k_M = max(k_M, getattr(self.model.params, "merchant_multi_home_cost", 0.0))
        k_M = max(k_M, getattr(self.model.params, "multi_home_cost", 0.0))

        base_gap = getattr(self.model.params, "multi_home_gap", 0.8)

        # k_M 越高，有效多归属门槛越低，因此多归属越难发生
        effective_gap = base_gap / (1.0 + k_M)

        if self.can_multi_home and gap <= effective_gap:
            return {"A", "B"}

        if u_A >= u_B:
            return {"A"}
        return {"B"}


class PlatformAgent(mesa.Agent):
    """
    平台智能体。

    重要约定：
    - self.platform_name == "A" 时，该智能体就是报告中的平台 A；
    - self.platform_name == "B" 时，该智能体就是报告中的平台 B。
    - 策略函数中的 x,y,shortage 都读取“自身平台”的状态。
    """

    def __init__(self, model, platform_name, active_strategy=False):
        super().__init__(model)

        self.platform_name = platform_name
        self.active_strategy = active_strategy

        self.user_subsidy = 0.0
        self.merchant_subsidy = 0.0

        self.user_quality = 0.0
        self.merchant_quality = 0.0

        self.user_invest = 0.0
        self.merchant_invest = 0.0

        self.cum_revenue = 0.0
        self.cum_subsidy_cost = 0.0
        self.cum_invest_cost = 0.0
        self.cum_profit = 0.0

    def reset_actions(self):
        self.user_subsidy = 0.0
        self.merchant_subsidy = 0.0
        self.user_invest = 0.0
        self.merchant_invest = 0.0

    def own_market_state(self):
        """
        返回当前平台自己的用户份额、商户份额和供给不足。
        """
        if self.platform_name == "A":
            return (
                self.model.user_share_A,
                self.model.merchant_share_A,
                self.model.shortage_A,
            )

        return (
            1.0 - self.model.user_share_A,
            1.0 - self.model.merchant_share_A,
            self.model.shortage_B,
        )

    def decide_strategy(self):
        """
        只有 active_strategy=True 的平台执行策略。
        报告策略实验中通常设置 active_strategy_platform="B"。
        """
        self.reset_actions()

        if not self.active_strategy:
            return

        p = self.model.params
        t = self.model.current_step

        if p.strategy == "none":
            return

        if p.strategy == "user_subsidy":
            self.user_subsidy = p.user_subsidy0 * self.model.exp_decay(t)

        elif p.strategy == "merchant_subsidy":
            self.merchant_subsidy = p.merchant_subsidy0 * self.model.exp_decay(t)

        elif p.strategy == "bilateral_subsidy":
            self.user_subsidy = p.bilateral_user_subsidy0 * self.model.exp_decay(t)
            self.merchant_subsidy = p.bilateral_merchant_subsidy0 * self.model.exp_decay(t)

        elif p.strategy in ["greedy", "replace_by_greedy"]:
            if t <= p.greedy_duration:
                self.user_subsidy = p.greedy_user_subsidy
                self.merchant_subsidy = p.greedy_merchant_subsidy
                self.user_invest = p.greedy_user_invest
                self.merchant_invest = p.greedy_merchant_invest

        elif p.strategy in [
            "long_term",
            "targeted_long_term",
            "full_targeted_long_term",
            "no_targeting",
        ]:
            self.targeted_long_term_strategy()

        elif p.strategy in ["dynamic", "targeted_dynamic", "replace_by_dynamic"]:
            self.dynamic_strategy()

        elif p.strategy in ["pure_quality"]:
            self.pure_quality_strategy()

        elif p.strategy in ["no_quality_investment"]:
            self.targeted_long_term_strategy()
            self.user_invest = 0.0
            self.merchant_invest = 0.0

    def targeted_long_term_strategy(self):
        p = self.model.params

        B = getattr(p, "budget", getattr(p, "total_budget", 0.8))

        x, y, shortage = self.own_market_state()

        user_gap = max(0.0, p.target_share - x)
        merchant_gap = max(0.0, p.target_share - y)

        if getattr(p, "targeting_enabled", True) and getattr(p, "targeted", True):
            user_need = user_gap + 0.05
            merchant_need = merchant_gap + 1.5 * shortage + 0.05
        else:
            user_need = 1.0
            merchant_need = 1.0

        total_need = max(user_need + merchant_need, 1e-12)

        subsidy_budget = 0.60 * B
        quality_budget = 0.40 * B

        self.user_subsidy = subsidy_budget * user_need / total_need
        self.merchant_subsidy = subsidy_budget * merchant_need / total_need

        if getattr(p, "quality_investment_enabled", True):
            self.user_invest = 0.5 * quality_budget
            self.merchant_invest = 0.5 * quality_budget
        else:
            self.user_invest = 0.0
            self.merchant_invest = 0.0

    def pure_quality_strategy(self):
        p = self.model.params

        B = getattr(p, "budget", getattr(p, "total_budget", 0.8))

        self.user_subsidy = 0.0
        self.merchant_subsidy = 0.0

        if getattr(p, "quality_investment_enabled", True):
            self.user_invest = 0.5 * B
            self.merchant_invest = 0.5 * B
        else:
            self.user_invest = 0.0
            self.merchant_invest = 0.0

    def dynamic_strategy(self):
        """
        动态策略：
        根据自身用户份额、商户份额和供给不足程度分配预算。
        """
        p = self.model.params

        x, y, shortage = self.own_market_state()

        user_gap = max(0.0, p.target_share - x)
        merchant_gap = max(0.0, p.target_share - y)

        user_need = user_gap + 0.05
        merchant_need = merchant_gap + 1.5 * shortage + 0.05
        total_need = max(user_need + merchant_need, 1e-12)

        B = getattr(p, "dynamic_budget", getattr(p, "budget", 0.8))
        user_budget = B * user_need / total_need
        merchant_budget = B * merchant_need / total_need

        if x < 0.65 or y < 0.65:
            self.user_subsidy = user_budget
            self.merchant_subsidy = merchant_budget
        else:
            self.user_subsidy = 0.30 * user_budget
            self.merchant_subsidy = 0.30 * merchant_budget
            self.user_invest = 0.25
            self.merchant_invest = 0.25

    def update_quality(self):
        p = self.model.params

        if not getattr(p, "quality_investment_enabled", True):
            self.user_invest = 0.0
            self.merchant_invest = 0.0

        qmax = getattr(p, "qmax", getattr(p, "q_max", 3.0))

        eff_user = getattr(p, "invest_eff_user", getattr(p, "invest_eff_u", 0.05))
        eff_merchant = getattr(p, "invest_eff_merchant", getattr(p, "invest_eff_m", 0.05))

        self.user_quality = np.clip(
            self.user_quality
            + eff_user * self.user_invest
            - p.quality_decay * self.user_quality,
            0.0,
            qmax,
        )

        self.merchant_quality = np.clip(
            self.merchant_quality
            + eff_merchant * self.merchant_invest
            - p.quality_decay * self.merchant_quality,
            0.0,
            qmax,
        )

    def update_profit(self):
        p = self.model.params

        if self.platform_name == "A":
            user_share = self.model.user_share_A
            merchant_share = self.model.merchant_share_A
        else:
            user_share = 1.0 - self.model.user_share_A
            merchant_share = 1.0 - self.model.merchant_share_A

        if getattr(p, "use_report_profit", True):
            # 报告口径：
            # pi = mu * u * m - subsidy_cost - investment_cost
            # 每期总预算 B 会被策略分配到 user_subsidy / merchant_subsidy / user_invest / merchant_invest，
            # 因而每期总成本约等于这些动作强度之和。
            mu = getattr(p, "profit_mu", 10.0)

            revenue = mu * user_share * merchant_share
            subsidy_cost = max(self.user_subsidy, 0.0) + max(self.merchant_subsidy, 0.0)
            invest_cost = max(self.user_invest, 0.0) + max(self.merchant_invest, 0.0)
        else:
            revenue = (
                p.revenue_user * user_share
                + p.revenue_merchant * merchant_share
            )

            subsidy_cost = (
                p.cost_user_subsidy * self.user_subsidy * user_share
                + p.cost_merchant_subsidy * self.merchant_subsidy * merchant_share
            )

            invest_cost = (
                p.cost_invest_user * self.user_invest
                + p.cost_invest_merchant * self.merchant_invest
            )

        profit = revenue - subsidy_cost - invest_cost

        discount = getattr(p, "discount", 0.98)
        discount_weight = discount ** self.model.current_step

        self.cum_revenue += revenue
        self.cum_subsidy_cost += subsidy_cost
        self.cum_invest_cost += invest_cost
        self.cum_profit += discount_weight * profit