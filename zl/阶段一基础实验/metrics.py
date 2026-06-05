def concentration(u: float, m: float) -> float:
    return abs(u - 0.5) + abs(m - 0.5)


def combined_share(u: float, m: float) -> float:
    return 0.5 * (u + m)


def market_state(c: float) -> str:
    if c < 0.2:
        return "双平台共存"
    if c < 0.8:
        return "市场倾斜"
    return "市场锁定"


def directional_market_state(u: float, m: float) -> str:
    c = concentration(u, m)
    if c < 0.2:
        return "双平台共存"
    if u > 0.8 and m > 0.8:
        return "平台A锁定"
    if u < 0.2 and m < 0.2:
        return "平台B锁定"
    return "市场倾斜"
