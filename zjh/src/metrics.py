def concentration(x: float, y: float) -> float:
    """
    市场集中度指标。

    x: 平台 A 用户份额
    y: 平台 A 商户份额

    C 越接近 0，说明两个平台越接近共存；
    C 越接近 1，说明市场越接近锁定。
    """
    return abs(x - 0.5) + abs(y - 0.5)


def combined_share(x: float, y: float) -> float:
    """
    平台 A 综合份额。
    """
    return 0.5 * (x + y)


def lock_in_index(x: float, y: float) -> float:
    """
    市场锁定指数，归一化到 [0, 1]。
    """
    return 0.5 * (abs(2.0 * x - 1.0) + abs(2.0 * y - 1.0))


def market_state(c: float) -> str:
    """
    根据市场集中度判断市场状态。
    """
    if c < 0.2:
        return "双平台共存"
    if c < 0.8:
        return "市场倾斜"
    return "市场锁定"


def directional_market_state(x: float, y: float) -> str:
    """
    判断市场锁定方向。
    """
    c = concentration(x, y)

    if c < 0.2:
        return "双平台共存"

    if x > 0.8 and y > 0.8:
        return "平台A锁定"

    if x < 0.2 and y < 0.2:
        return "平台B锁定"

    return "市场倾斜"