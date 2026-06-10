from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MesaABMParams:
    """
    Mesa Agent-Based Model 参数。

    本模型用于在前五阶段宏观 Logit 动态模型基础上，
    引入用户异质性、商户异质性、多归属、随机偏好和平台动态策略。
    """

    # 随机种子
    seed: int = 2026

    # 智能体规模
    n_users: int = 1000
    n_merchants: int = 300
    n_platforms: int = 2

    # 仿真期数与重复次数
    max_steps: int = 300
    n_runs: int = 30

    # 初始市场份额：平台 A 初始用户份额与商户份额
    x0: float = 0.35
    y0: float = 0.35

    # 双边网络效应强度
    alpha: float = 0.8
    beta: float = 0.8

    # 个体异质性强度
    sigma_theta: float = 0.8
    sigmaU: float = 0.8
    sigmaM: float = 0.8
    preference_sigma: float = 0.8
    # 用户类型比例
    user_type_ratios: Dict[str, float] = field(default_factory=lambda: {
        "price_sensitive": 0.40,
        "quality_sensitive": 0.30,
        "inertial": 0.20,
        "normal": 0.10,
    })

    # 商户类型比例
    merchant_type_ratios: Dict[str, float] = field(default_factory=lambda: {
        "large": 0.20,
        "small_medium": 0.50,
        "new": 0.20,
        "multi_home": 0.10,
    })

    # 用户类型参数
    user_type_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # w_q: 服务质量权重
        # w_m: 商户规模权重
        # w_s: 用户补贴权重
        # w_p: 价格成本权重
        # w_c: 供给不足惩罚权重
        # inertia: 切换惯性
        "price_sensitive": {
            "w_q": 0.6,
            "w_m": 0.8,
            "w_s": 1.5,
            "w_p": 1.2,
            "w_c": 0.8,
            "inertia": 0.2,
        },
        "quality_sensitive": {
            "w_q": 1.5,
            "w_m": 0.8,
            "w_s": 0.6,
            "w_p": 0.8,
            "w_c": 1.2,
            "inertia": 0.2,
        },
        "inertial": {
            "w_q": 0.8,
            "w_m": 0.8,
            "w_s": 0.6,
            "w_p": 0.8,
            "w_c": 0.8,
            "inertia": 1.0,
        },
        "normal": {
            "w_q": 1.0,
            "w_m": 1.0,
            "w_s": 1.0,
            "w_p": 1.0,
            "w_c": 1.0,
            "inertia": 0.4,
        },
    })

    # 商户类型参数
    merchant_type_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        # v_r: 平台服务/声誉权重
        # v_u: 用户规模权重
        # v_s: 商户补贴/扶持权重
        # v_c: 佣金成本权重
        # multi_home_prob: 多归属概率
        "large": {
            "v_r": 1.0,
            "v_u": 1.5,
            "v_s": 0.5,
            "v_c": 0.8,
            "multi_home_prob": 0.3,
        },
        "small_medium": {
            "v_r": 0.8,
            "v_u": 1.0,
            "v_s": 1.2,
            "v_c": 1.2,
            "multi_home_prob": 0.4,
        },
        "new": {
            "v_r": 0.8,
            "v_u": 0.8,
            "v_s": 1.5,
            "v_c": 1.0,
            "multi_home_prob": 0.5,
        },
        "multi_home": {
            "v_r": 1.0,
            "v_u": 1.0,
            "v_s": 1.0,
            "v_c": 1.0,
            "multi_home_prob": 1.0,
        },
    })

    # 用户多归属设置
    user_multi_home_prob: float = 0.10

    # 商户多归属实验中可覆盖商户多归属概率
    merchant_multi_home_override: Optional[float] = None

    # 效用差距低于该阈值时，多归属智能体可同时选择两个平台
    multi_home_gap: float = 0.80

    # 商户多归属成本，报告实验 2 扫描 k_M
    k_M: float = 0.0
    merchant_multi_home_cost: float = 0.0
    multi_home_cost: float = 0.0

    # 随机偏好扰动
    user_taste_noise: float = 0.15
    merchant_taste_noise: float = 0.12

    # 平台基础价格、佣金
    price_A: float = 0.0
    price_B: float = 0.0
    commission_A: float = 0.0
    commission_B: float = 0.0

    shortage_enabled: bool = True
    supply_penalty_enabled: bool = True

    N_U: int = 1000
    N_M: int = 50

    shortage_rho: float = 10.0
    shortage_theta: float = 1.0
    shortage_buffer: float = 1e-6

   # 兼容旧字段
    supply_capacity: float = 1.0

    # 收益参数
    revenue_user: float = 1.0
    revenue_merchant: float = 1.2

    # 补贴成本
    cost_user_subsidy: float = 0.8
    cost_merchant_subsidy: float = 0.9

    # 服务质量投资参数
    quality_investment_enabled: bool = True

    invest_eff_user: float = 0.05
    invest_eff_merchant: float = 0.05

    # 兼容 stage6 传入的不同字段名
    lambda_q: float = 0.05
    quality_efficiency: float = 0.05

    quality_decay: float = 0.01
    qmax: float = 3.0
    q_max: float = 3.0
    cost_invest_user: float = 1.1
    cost_invest_merchant: float = 1.1

    # 补贴强度
    user_subsidy0: float = 0.85
    merchant_subsidy0: float = 0.85
    bilateral_user_subsidy0: float = 0.55
    bilateral_merchant_subsidy0: float = 0.55
    subsidy_decay: float = 0.03

    # 贪心、长期、动态策略参数
    greedy_duration: int = 35
    greedy_user_subsidy: float = 0.90
    greedy_merchant_subsidy: float = 0.10
    greedy_user_invest: float = 0.55
    greedy_merchant_invest: float = 0.10

    long_term_user_invest: float = 0.22
    long_term_merchant_invest: float = 0.22

    dynamic_budget: float = 0.85
    target_share: float = 0.60

    # 总预算/策略预算
    budget: float = 0.8
    total_budget: float = 0.8
    budget_total: float = 0.8

    # 精准补贴开关
    targeted: bool = True
    targeting_enabled: bool = True

    # 平台 A 策略
    # 可选：
    # none, user_subsidy, merchant_subsidy, bilateral_subsidy,
    # greedy, long_term, dynamic
    strategy: str = "none"

    # 是否允许平台 B 反制，当前默认不反制
    platform_B_response: bool = False