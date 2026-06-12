from dataclasses import dataclass


@dataclass
class Params:
    

    # ============================================================
    # 双边网络效应
    # ============================================================

    alpha: float = 1.0
    beta: float = 1.0

    # 同侧锁定 / 切换成本
    eta_u: float = 0.25
    eta_m: float = 0.25

    # 平台 A 相对平台 B 的外生质量差异
    dq_u_base: float = 0.0
    dq_m_base: float = 0.0

    # 价格 / 成本差异
    dp_u: float = 0.0
    dc_m: float = 0.0

    # Logit 敏感度
    # Stage 1 主实验建议保持 2.0；Stage 2--5 可在 make_params 中显式设为 2.6。
    lambda_u: float = 2.0
    lambda_m: float = 2.0

    # 市场份额调整速度
    r_u: float = 1.0
    r_m: float = 1.0

    # ============================================================
    # 供给不足惩罚
    # ============================================================

    shortage_enabled: bool = False

    # 报告中的供需规模参数
    N_U: int = 1000
    N_M: int = 50

    # 供给不足阈值和惩罚强度
    shortage_rho: float = 10.0
    shortage_theta: float = 1.0

    # 防止分母为 0
    shortage_buffer: float = 1e-6

    # 供给不足惩罚模式：
    # absolute_B：平台 B 自身供给不足 C_B 直接降低 B 吸引力；
    # relative：旧口径，使用 C_A 和 C_B 的相对差异。
    shortage_mode: str = "absolute_B"

    # 兼容旧字段
    supply_ratio: float = 1.0

    # ============================================================
    # 服务质量投资
    # ============================================================

    quality_decay: float = 0.01
    invest_eff_u: float = 0.05
    invest_eff_m: float = 0.05
    qmax: float = 3.0

    # 外生质量优势和质量投资存量对效用的作用可以分开控制。
    # Stage 1 的 Δq 阈值偏低时，可调低 quality_base_effect_scale；
    # Stage 4 的质量投资反超过慢时，可调高 quality_stock_effect_scale。
    quality_base_effect_scale: float = 1.0
    quality_stock_effect_scale: float = 1.0

    # ============================================================
    # 利润与成本
    # ============================================================

    discount: float = 0.98

    # 旧线性收入口径，保留兼容
    revenue_user: float = 1.0
    revenue_merchant: float = 1.2

    cost_user_subsidy: float = 0.8
    cost_merchant_subsidy: float = 0.9
    cost_invest_u: float = 1.1
    cost_invest_m: float = 1.1

    # 报告利润口径：
    # pi_B = profit_mu * u_B * m_B - s_u_B - s_m_B - I_B
    use_report_profit: bool = True
    profit_mu: float = 5.0

    # 对补贴进入效用的缩放。默认 1.0。
    # 如果 Stage 2 低预算仍过强，可在实验中设为 0.8 或更小。
    subsidy_effect_scale: float = 1.0