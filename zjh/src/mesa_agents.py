import mesa
import numpy as np

class UserAgent(mesa.Agent):
    """
    用户智能体。

    用户类型包括：
    1. price_sensitive：价格敏感型；
    2. quality_sensitive：质量敏感型；
    3. inertial：惯性用户；
    4. normal：普通用户。
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

    商户类型包括：
    1. large：大商户；
    2. small_medium：中小商户；
    3. new：新商户；
    4. multi_home：多归属商户。
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

    平台智能体负责：
    1. 设置用户补贴；
    2. 设置商户补贴；
    3. 决定服务质量投资；
    4. 根据市场状态调整策略；
    5. 记录平台利润。
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

    def decide_strategy(self):
        """
        当前实现中，平台 A 可以主动采取策略，
        平台 B 默认作为基准平台，不主动反制。
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

        elif p.strategy == "greedy":
            if t <= p.greedy_duration:
                self.user_subsidy = p.greedy_user_subsidy
                self.merchant_subsidy = p.greedy_merchant_subsidy
                self.user_invest = p.greedy_user_invest
                self.merchant_invest = p.greedy_merchant_invest

        elif p.strategy in ["long_term", "targeted_long_term", "full_targeted_long_term"]:
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

        x = self.model.user_share_A
        y = self.model.merchant_share_A
        shortage = self.model.shortage_A

        user_gap = max(0.0, p.target_share - x)
        merchant_gap = max(0.0, p.target_share - y)

        if getattr(p, "targeting_enabled", True):
            user_need = user_gap + 0.05
            merchant_need = merchant_gap + 1.5 * shortage + 0.05
        else:
            user_need = 1.0
            merchant_need = 1.0

        total_need = user_need + merchant_need

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
        根据当前用户份额、商户份额和供给不足程度分配预算。
        """
        p = self.model.params

        x = self.model.user_share_A
        y = self.model.merchant_share_A
        shortage = self.model.shortage_A

        user_gap = max(0.0, p.target_share - x)
        merchant_gap = max(0.0, p.target_share - y)

        # 供给不足时，更偏向商户侧扶持
        user_need = user_gap + 0.05
        merchant_need = merchant_gap + 1.5 * shortage + 0.05

        total_need = user_need + merchant_need

        user_budget = p.dynamic_budget * user_need / total_need
        merchant_budget = p.dynamic_budget * merchant_need / total_need

        # 前期主要补贴，后期转为质量投资
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

        self.user_quality = np.clip(
            self.user_quality
            + p.invest_eff_user * self.user_invest
            - p.quality_decay * self.user_quality,
            0.0,
            qmax,
        )

        self.merchant_quality = np.clip(
            self.merchant_quality
            + p.invest_eff_merchant * self.merchant_invest
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