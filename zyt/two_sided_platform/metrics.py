"""Evaluation metrics for market concentration and platform performance."""


def lock_index(x, y):
    """Market lock-in index: close to 1 means highly concentrated."""
    return max(x, 1.0 - x) * max(y, 1.0 - y)


def hhi_user(x):
    return x**2 + (1.0 - x) ** 2


def hhi_merchant(y):
    return y**2 + (1.0 - y) ** 2


def hhi(x, y):
    return 0.5 * (hhi_user(x) + hhi_merchant(y))


def is_lock_in(x, y, threshold=0.80):
    return lock_index(x, y) > threshold


def platform_profit(user_share, merchant_share, action, mu=10.0):
    revenue = mu * user_share * merchant_share
    cost = (
        action["user_subsidy"]
        + action["merchant_subsidy"]
        + action["quality_investment"]
    )
    return revenue - cost


def discounted_total(values, discount=0.98):
    return sum((discount**idx) * value for idx, value in enumerate(values, start=1))


def subsidy_efficiency(delta_user_share, delta_merchant_share, total_subsidy):
    if total_subsidy <= 0:
        return 0.0
    return (delta_user_share + delta_merchant_share) / total_subsidy
