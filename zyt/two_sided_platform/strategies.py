"""Simple platform action strategies."""


def make_action(user_subsidy=0.0, merchant_subsidy=0.0, quality_investment=0.0):
    return {
        "user_subsidy": float(user_subsidy),
        "merchant_subsidy": float(merchant_subsidy),
        "quality_investment": float(quality_investment),
    }


def no_strategy(budget=0.0):
    return make_action()


def user_subsidy_strategy(budget):
    return make_action(user_subsidy=budget)


def merchant_subsidy_strategy(budget):
    return make_action(merchant_subsidy=budget)


def balanced_subsidy_strategy(budget):
    return make_action(user_subsidy=0.5 * budget, merchant_subsidy=0.5 * budget)


def user_leaning_subsidy_strategy(budget):
    return make_action(user_subsidy=0.75 * budget, merchant_subsidy=0.25 * budget)


def merchant_leaning_subsidy_strategy(budget):
    return make_action(user_subsidy=0.25 * budget, merchant_subsidy=0.75 * budget)


def quality_only_strategy(budget):
    return make_action(quality_investment=budget)


def greedy_strategy(budget):
    return make_action(
        user_subsidy=0.8 * budget,
        merchant_subsidy=0.1 * budget,
        quality_investment=0.1 * budget,
    )


def long_term_strategy(budget):
    return make_action(
        user_subsidy=0.3 * budget,
        merchant_subsidy=0.4 * budget,
        quality_investment=0.3 * budget,
    )


def dynamic_strategy(
    budget,
    user_share,
    merchant_share,
    rho=10.0,
    eps=1e-6,
    user_scale=1000.0,
    merchant_scale=50.0,
):
    ratio = (user_share * user_scale) / (merchant_share * merchant_scale + eps)
    if ratio > rho:
        return make_action(
            user_subsidy=0.2 * budget,
            merchant_subsidy=0.6 * budget,
            quality_investment=0.2 * budget,
        )
    if ratio < 0.8 * rho:
        return make_action(
            user_subsidy=0.6 * budget,
            merchant_subsidy=0.2 * budget,
            quality_investment=0.2 * budget,
        )
    return make_action(
        user_subsidy=0.3 * budget,
        merchant_subsidy=0.3 * budget,
        quality_investment=0.4 * budget,
    )
