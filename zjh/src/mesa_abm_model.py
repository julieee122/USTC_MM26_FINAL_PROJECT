import numpy as np
import mesa

from src.mesa_agents import UserAgent, MerchantAgent, PlatformAgent


class TwoSidedPlatformABM(mesa.Model):
    """
    双边平台竞争 Mesa ABM 模型。

    相比前五阶段宏观 Logit 模型，本模型引入：
    1. 用户异质性；
    2. 商户异质性；
    3. 多归属；
    4. 随机偏好；
    5. 平台智能体动态策略。
    """

    def __init__(self, params):
        super().__init__(seed=params.seed)

        self.params = params
        self.rng = np.random.default_rng(params.seed)

        self.current_step = 0

        self.users = []
        self.merchants = []

        self.platform_A = PlatformAgent(
            model=self,
            platform_name="A",
            active_strategy=True
        )

        self.platform_B = PlatformAgent(
            model=self,
            platform_name="B",
            active_strategy=False
        )

        self.user_share_A = params.x0
        self.merchant_share_A = params.y0

        self.user_multi_home_rate = 0.0
        self.merchant_multi_home_rate = 0.0

        self.shortage_A = 0.0
        self.shortage_B = 0.0

        self.avg_user_utility = 0.0
        self.avg_merchant_utility = 0.0

        self.history = []

        self.create_users()
        self.create_merchants()
        self.update_market_state()
        self.record_history()

    # ============================================================
    # 初始化工具函数
    # ============================================================

    def exp_decay(self, t):
        return float(np.exp(-self.params.subsidy_decay * t))

    def sample_type(self, ratio_dict):
        types = list(ratio_dict.keys())
        probs = np.array([ratio_dict[k] for k in types], dtype=float)
        probs = probs / probs.sum()
        return str(self.rng.choice(types, p=probs))

    def initial_platforms(self, share_A, can_multi_home):
        if can_multi_home and self.rng.random() < 0.5:
            return {"A", "B"}

        if self.rng.random() < share_A:
            return {"A"}

        return {"B"}

    # ============================================================
    # 创建智能体
    # ============================================================

    def create_users(self):
        p = self.params

        for _ in range(p.n_users):
            user_type = self.sample_type(p.user_type_ratios)
            type_param = p.user_type_params[user_type]

            can_multi_home = self.rng.random() < p.user_multi_home_prob
            platforms = self.initial_platforms(p.x0, can_multi_home)

            agent = UserAgent(
                model=self,
                user_type=user_type,
                platforms=platforms,
                w_q=type_param["w_q"],
                w_m=type_param["w_m"],
                w_s=type_param["w_s"],
                w_p=type_param["w_p"],
                w_c=type_param["w_c"],
                inertia=type_param["inertia"],
                can_multi_home=can_multi_home,
                base_pref_A=float(self.rng.normal(0.0, 0.05)),
                base_pref_B=float(self.rng.normal(0.0, 0.05)),
            )

            self.users.append(agent)

    def create_merchants(self):
        p = self.params

        for _ in range(p.n_merchants):
            merchant_type = self.sample_type(p.merchant_type_ratios)
            type_param = p.merchant_type_params[merchant_type]

            if p.merchant_multi_home_override is None:
                multi_prob = type_param["multi_home_prob"]
            else:
                multi_prob = p.merchant_multi_home_override

            can_multi_home = self.rng.random() < multi_prob
            platforms = self.initial_platforms(p.y0, can_multi_home)

            agent = MerchantAgent(
                model=self,
                merchant_type=merchant_type,
                platforms=platforms,
                v_r=type_param["v_r"],
                v_u=type_param["v_u"],
                v_s=type_param["v_s"],
                v_c=type_param["v_c"],
                can_multi_home=can_multi_home,
                base_pref_A=float(self.rng.normal(0.0, 0.05)),
                base_pref_B=float(self.rng.normal(0.0, 0.05)),
            )

            self.merchants.append(agent)

    # ============================================================
    # 份额、多归属、满意度统计
    # ============================================================

    @staticmethod
    def platform_weight(agent, platform):
        if platform not in agent.platforms:
            return 0.0

        if len(agent.platforms) == 2:
            return 0.5

        return 1.0

    def weighted_share(self, agents, platform):
        if not agents:
            return 0.0

        return sum(self.platform_weight(a, platform) for a in agents) / len(agents)

    @staticmethod
    def multi_home_rate(agents):
        if not agents:
            return 0.0

        return sum(1 for a in agents if len(a.platforms) == 2) / len(agents)

    @staticmethod
    def average_utility(agents):
        if not agents:
            return 0.0

        return sum(a.last_utility for a in agents) / len(agents)

    def type_weighted_share(self, agents, type_attr, type_name, platform):
        sub_agents = [a for a in agents if getattr(a, type_attr) == type_name]

        if not sub_agents:
            return 0.0

        return self.weighted_share(sub_agents, platform)

    def type_average_utility(self, agents, type_attr, type_name):
        sub_agents = [a for a in agents if getattr(a, type_attr) == type_name]

        if not sub_agents:
            return 0.0

        return self.average_utility(sub_agents)

    def lock_in_index(self):
        """
        市场锁定指数：
        0 表示两个平台均衡；
        1 表示极端赢家通吃。
        """
        user_lock = abs(2.0 * self.user_share_A - 1.0)
        merchant_lock = abs(2.0 * self.merchant_share_A - 1.0)

        return 0.5 * (user_lock + merchant_lock)

    def update_market_state(self):
        self.user_share_A = self.weighted_share(self.users, "A")
        self.merchant_share_A = self.weighted_share(self.merchants, "A")

        self.user_multi_home_rate = self.multi_home_rate(self.users)
        self.merchant_multi_home_rate = self.multi_home_rate(self.merchants)

        self.shortage_A = max(
            0.0,
            self.user_share_A
            - self.params.supply_capacity * self.merchant_share_A
            - self.params.shortage_buffer
        )

        self.shortage_B = max(
            0.0,
            (1.0 - self.user_share_A)
            - self.params.supply_capacity * (1.0 - self.merchant_share_A)
            - self.params.shortage_buffer
        )

        self.avg_user_utility = self.average_utility(self.users)
        self.avg_merchant_utility = self.average_utility(self.merchants)

    # ============================================================
    # 效用函数
    # ============================================================

    def user_utility(self, user, platform):
        p = self.params

        if platform == "A":
            merchant_share = self.merchant_share_A
            quality = self.platform_A.user_quality
            subsidy = self.platform_A.user_subsidy
            price = p.price_A
            shortage = self.shortage_A
            base_pref = user.base_pref_A
        else:
            merchant_share = 1.0 - self.merchant_share_A
            quality = self.platform_B.user_quality
            subsidy = self.platform_B.user_subsidy
            price = p.price_B
            shortage = self.shortage_B
            base_pref = user.base_pref_B

        inertia_bonus = user.inertia if platform in user.platforms else 0.0
        noise = float(self.rng.gumbel(0.0, p.user_taste_noise))

        utility = (
            base_pref
            + user.w_q * quality
            + user.w_m * merchant_share
            + user.w_s * subsidy
            - user.w_p * price
            - user.w_c * p.shortage_rho * shortage
            + inertia_bonus
            + noise
        )

        return utility

    def merchant_utility(self, merchant, platform):
        p = self.params

        if platform == "A":
            user_share = self.user_share_A
            quality = self.platform_A.merchant_quality
            subsidy = self.platform_A.merchant_subsidy
            commission = p.commission_A
            base_pref = merchant.base_pref_A
        else:
            user_share = 1.0 - self.user_share_A
            quality = self.platform_B.merchant_quality
            subsidy = self.platform_B.merchant_subsidy
            commission = p.commission_B
            base_pref = merchant.base_pref_B

        inertia_bonus = 0.25 if platform in merchant.platforms else 0.0
        noise = float(self.rng.gumbel(0.0, p.merchant_taste_noise))

        utility = (
            base_pref
            + merchant.v_r * quality
            + merchant.v_u * user_share
            + merchant.v_s * subsidy
            - merchant.v_c * commission
            + inertia_bonus
            + noise
        )

        return utility

    # ============================================================
    # 仿真循环
    # ============================================================

    def step(self):
        self.platform_A.decide_strategy()
        self.platform_B.decide_strategy()

        self.platform_A.update_quality()
        self.platform_B.update_quality()

        # 商户先决策，再用户决策
        self.rng.shuffle(self.merchants)
        for merchant in self.merchants:
            merchant.step()

        self.update_market_state()

        self.rng.shuffle(self.users)
        for user in self.users:
            user.step()

        self.update_market_state()

        self.platform_A.update_profit()
        self.platform_B.update_profit()

        self.current_step += 1
        self.record_history()

    def run_model(self):
        for _ in range(self.params.max_steps):
            self.step()

    def record_history(self):
        row = {
            "t": self.current_step,
            "strategy": self.params.strategy,

            "user_share_A": self.user_share_A,
            "merchant_share_A": self.merchant_share_A,

            "user_multi_home_rate": self.user_multi_home_rate,
            "merchant_multi_home_rate": self.merchant_multi_home_rate,

            "shortage_A": self.shortage_A,
            "shortage_B": self.shortage_B,

            "avg_user_utility": self.avg_user_utility,
            "avg_merchant_utility": self.avg_merchant_utility,

            "lock_in_index": self.lock_in_index(),

            "user_subsidy_A": self.platform_A.user_subsidy,
            "merchant_subsidy_A": self.platform_A.merchant_subsidy,

            "user_quality_A": self.platform_A.user_quality,
            "merchant_quality_A": self.platform_A.merchant_quality,

            "cum_revenue_A": self.platform_A.cum_revenue,
            "cum_subsidy_cost_A": self.platform_A.cum_subsidy_cost,
            "cum_invest_cost_A": self.platform_A.cum_invest_cost,
            "cum_profit_A": self.platform_A.cum_profit,
        }

        # 用户类型分组指标
        for user_type in self.params.user_type_ratios.keys():
            row[f"user_share_A_{user_type}"] = self.type_weighted_share(
                self.users,
                "user_type",
                user_type,
                "A"
            )
            row[f"user_utility_{user_type}"] = self.type_average_utility(
                self.users,
                "user_type",
                user_type
            )

        # 商户类型分组指标
        for merchant_type in self.params.merchant_type_ratios.keys():
            row[f"merchant_share_A_{merchant_type}"] = self.type_weighted_share(
                self.merchants,
                "merchant_type",
                merchant_type,
                "A"
            )
            row[f"merchant_utility_{merchant_type}"] = self.type_average_utility(
                self.merchants,
                "merchant_type",
                merchant_type
            )

        self.history.append(row)