from dataclasses import dataclass


@dataclass
class Params:
    """
    双边平台 Logit 动态模型参数。
    """

    # 双边交叉网络效应
    alpha: float = 1.0        # 商户规模对用户选择的影响
    beta: float = 1.0         # 用户规模对商户选择的影响

    # 同侧锁定效应
    eta_u: float = 0.25
    eta_m: float = 0.25

    # 平台 A 相对平台 B 的基础服务质量差异
    dq_u_base: float = 0.0
    dq_m_base: float = 0.0

    # 用户价格差异、商户成本差异
    dp_u: float = 0.0
    dc_m: float = 0.0

    # Logit 敏感度
    lambda_u: float = 3.5
    lambda_m: float = 3.5

    # 市场份额调整速度
    r_u: float = 1.0
    r_m: float = 1.0

    # 供给不足惩罚
    shortage_enabled: bool = False

    # 报告中的供需规模参数
    N_U: int = 1000
    N_M: int = 50

     # 报告中的供给不足阈值 rho 和惩罚强度 theta
    shortage_rho: float = 10.0
    shortage_theta: float = 1.0

    # 防止分母为 0 的极小量
    shortage_buffer: float = 1e-6

    # 兼容旧模型字段；如果使用报告公式，可不再依赖 supply_ratio
    supply_ratio: float = 1.0
    # 服务质量投资动态参数
    quality_decay: float = 0.01
    invest_eff_u: float = 0.05
    invest_eff_m: float = 0.05
    qmax: float = 3.0

    # 收益与成本参数
    # 贴现因子
    discount: float = 0.98
    revenue_user: float = 1.0
    revenue_merchant: float = 1.2

    cost_user_subsidy: float = 0.8
    cost_merchant_subsidy: float = 0.9

    cost_invest_u: float = 1.1
    cost_invest_m: float = 1.1