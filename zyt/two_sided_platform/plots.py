"""Optional plotting helpers for stage 1 experiments."""

from pathlib import Path


def plot_stage1_figures(summaries, trajectories, output_dir="results/figures"):
    """Create the plan's first three suggested figures when matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, records in trajectories.items():
        if run_id.startswith("exp1_") and "_A_small_lead" in run_id:
            times = [row["time"] for row in records]
            xs = [row["x_A_user"] for row in records]
            ax.plot(times, xs, label=run_id.replace("exp1_", ""))
    ax.set_xlabel("time")
    ax.set_ylabel("platform A user share x(t)")
    ax.set_title("Experiment 1: network effect and x(t)")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig1_exp1_network_trajectories.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    exp1_rows = [row for row in summaries if row["experiment"] == "network_initial_scale"]
    network_labels = []
    initial_labels = []
    for row in exp1_rows:
        if row["network_label"] not in network_labels:
            network_labels.append(row["network_label"])
        if row["initial_label"] not in initial_labels:
            initial_labels.append(row["initial_label"])

    matrix = [
        [
            next(
                row["lock_index"]
                for row in exp1_rows
                if row["network_label"] == network and row["initial_label"] == initial
            )
            for initial in initial_labels
        ]
        for network in network_labels
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(matrix, vmin=0.25, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(initial_labels)), initial_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(network_labels)), network_labels)
    ax.set_title("Experiment 1: final lock-in index")
    fig.colorbar(image, ax=ax, label="Lock_T")
    fig.tight_layout()
    path = output_dir / "fig2_exp1_lock_heatmap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    exp2_rows = [row for row in summaries if row["experiment"] == "quality_break_lock"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for network in network_labels:
        rows = [row for row in exp2_rows if row["network_label"] == network]
        rows.sort(key=lambda row: row["delta_q"])
        ax.plot(
            [row["delta_q"] for row in rows],
            [row["B_user_T"] for row in rows],
            marker="o",
            label=f"{network}",
        )
    ax.axhline(0.5, color="gray", linewidth=1)
    ax.set_xlabel("B quality advantage delta")
    ax.set_ylabel("B final user/merchant share")
    ax.set_title("Experiment 2: quality advantage and final share")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig3_exp2_quality_advantage.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    return written


def plot_stage234_figures(summaries, trajectories, output_dir="results/figures"):
    """Create compact figures for stages 2-4 when matplotlib exists."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    stage2_rows = [
        row
        for row in summaries
        if row["stage"] == 2
        and row["experiment"] == "subsidy_strategy"
        and row["network_label"] == "medium"
        and row["strategy"] != "none"
    ]
    strategy_order = []
    for row in stage2_rows:
        if row["strategy"] not in strategy_order:
            strategy_order.append(row["strategy"])
    combined_strategy_groups = [
        ("merchant_only/user_only", ["merchant_only", "user_only"]),
        ("balanced", ["balanced"]),
        ("merchant_leaning/user_leaning", ["merchant_leaning", "user_leaning"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, strategies in combined_strategy_groups:
        rows = [
            row
            for row in stage2_rows
            if row["strategy"] == strategies[0]
        ]
        rows.sort(key=lambda row: row["budget"])
        ax.plot(
            [row["budget"] for row in rows],
            [0.5 * (row["B_user_T"] + row["B_merchant_T"]) for row in rows],
            marker="o",
            label=label,
        )
    ax.axhline(0.5, color="gray", linewidth=1)
    ax.set_xlabel("budget per period")
    ax.set_ylabel("B final average share")
    ax.set_title("Stage 2: subsidy allocation and final share")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig4_stage2_subsidy_share.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 5))
    selected_budget = 0.1
    rows = []
    for label, strategies in [("none", ["none"]), *combined_strategy_groups]:
        row = next(
            row
            for row in summaries
            if row["stage"] == 2
            and row["network_label"] == "medium"
            and row["budget"] == selected_budget
            and row["strategy"] == strategies[0]
        )
        rows.append({"label": label, **row})
    rows.sort(key=lambda row: row["discounted_profit_B"], reverse=True)
    ax.bar(
        [row["label"] for row in rows],
        [row["discounted_profit_B"] for row in rows],
        color="#4c78a8",
    )
    ax.set_ylabel("discounted profit of B")
    ax.set_title("Stage 2: profit comparison at budget 0.1")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = output_dir / "fig5_stage2_profit.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    for ax, selected_budget in zip(axes, [0.1, 1.2]):
        selected_runs = {
            run_id.replace("stage2_medium_", "").replace(
                f"_B{selected_budget:g}", ""
            ): records
            for run_id, records in trajectories.items()
            if run_id.startswith("stage2_medium_")
            and run_id.endswith(f"_B{selected_budget:g}")
        }

        records = selected_runs["user_only"]
        times = [row["time"] for row in records]
        ax.plot(
            times,
            [row["u_B_user"] for row in records],
            color="#4c78a8",
            label="user_only/merchant_only subsidized side",
        )
        ax.plot(
            times,
            [row["m_B_merchant"] for row in records],
            color="#4c78a8",
            linestyle="--",
            label="user_only/merchant_only other side",
        )

        records = selected_runs["balanced"]
        ax.plot(
            times,
            [row["u_B_user"] for row in records],
            color="#54a24b",
            label="balanced both sides",
        )

        records = selected_runs["user_leaning"]
        ax.plot(
            times,
            [row["u_B_user"] for row in records],
            color="#e45756",
            label="user_leaning/merchant_leaning favored side",
        )
        ax.plot(
            times,
            [row["m_B_merchant"] for row in records],
            color="#e45756",
            linestyle="--",
            label="user_leaning/merchant_leaning less-subsidized side",
        )
        ax.axhline(0.5, color="gray", linewidth=1)
        ax.set_title(f"medium network, budget {selected_budget:g}")
        ax.set_xlabel("time")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("B share")
    axes[1].legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    path = output_dir / "fig8_stage2_share_trajectories.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    selected_budget = 0.1
    rows = [
        row
        for row in summaries
        if row["stage"] == 2
        and row["experiment"] == "subsidy_strategy"
        and row["network_label"] == "medium"
        and row["budget"] == selected_budget
        and row["strategy"] != "none"
    ]
    rows.sort(key=lambda row: strategy_order.index(row["strategy"]))
    positions = range(len(rows))
    width = 0.36
    ax.bar(
        [pos - width / 2 for pos in positions],
        [row["B_user_T"] for row in rows],
        width=width,
        label="B user share",
        color="#4c78a8",
    )
    ax.bar(
        [pos + width / 2 for pos in positions],
        [row["B_merchant_T"] for row in rows],
        width=width,
        label="B merchant share",
        color="#f58518",
    )
    ax.axhline(0.5, color="gray", linewidth=1)
    ax.set_xticks(list(positions), [row["strategy"] for row in rows], rotation=25, ha="right")
    ax.set_ylabel("final share")
    ax.set_title("Stage 2: side imbalance at budget 0.1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig9_stage2_low_budget_side_gap.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    low_budget_rows = [
        row
        for row in summaries
        if row["stage"] == 2
        and row["experiment"] == "low_budget_single_side"
        and row["network_label"] == "medium"
    ]
    if low_budget_rows:
        label_map = {
            "user_only": "single-side subsidy",
            "balanced": "balanced subsidy",
            "user_leaning": "user-leaning subsidy",
        }
        rows_by_strategy = {}
        for strategy in label_map:
            rows = [row for row in low_budget_rows if row["strategy"] == strategy]
            rows.sort(key=lambda row: row["budget"])
            rows_by_strategy[strategy] = rows
        balanced_by_budget = {
            row["budget"]: row for row in rows_by_strategy["balanced"]
        }

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        for strategy, label in label_map.items():
            rows = rows_by_strategy[strategy]
            budgets = [row["budget"] for row in rows]
            avg_shares = [
                0.5 * (row["B_user_T"] + row["B_merchant_T"])
                for row in rows
            ]
            profits = [row["discounted_profit_B"] for row in rows]
            share_gaps = []
            profit_gaps = []
            for row, share, profit in zip(rows, avg_shares, profits):
                base = balanced_by_budget[row["budget"]]
                base_share = 0.5 * (base["B_user_T"] + base["B_merchant_T"])
                share_gaps.append(share - base_share)
                profit_gaps.append(profit - base["discounted_profit_B"])
            axes[0, 0].plot(budgets, avg_shares, marker="o", markersize=3, label=label)
            axes[0, 1].plot(budgets, profits, marker="o", markersize=3, label=label)
            axes[1, 0].plot(budgets, share_gaps, marker="o", markersize=3, label=label)
            axes[1, 1].plot(budgets, profit_gaps, marker="o", markersize=3, label=label)
        axes[0, 0].axhline(0.5, color="gray", linewidth=1)
        axes[1, 0].axhline(0, color="gray", linewidth=1)
        axes[1, 1].axhline(0, color="gray", linewidth=1)
        axes[0, 0].set_title("final average share")
        axes[0, 0].set_ylabel("B final average share")
        axes[0, 1].set_title("discounted profit")
        axes[0, 1].set_ylabel("discounted profit of B")
        axes[1, 0].set_title("share advantage over balanced")
        axes[1, 0].set_ylabel("share difference")
        axes[1, 1].set_title("profit advantage over balanced")
        axes[1, 1].set_ylabel("profit difference")
        for ax in axes.flat:
            ax.set_xlabel("budget per period")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle("Stage 2 supplement: low-budget one-sided subsidy effect")
        fig.tight_layout()
        path = output_dir / "fig_stage2_exp1_low_budget_single_side.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    selected_budgets = [0.1, 1.2]
    selected_network = "strong"
    strategies = [strategy for strategy in strategy_order if strategy != "none"]
    positions = range(len(strategies))
    width = 0.36
    budget_colors = {0.1: "#4c78a8", 1.2: "#e45756"}
    for offset, budget in [(-width / 2, selected_budgets[0]), (width / 2, selected_budgets[1])]:
        rows = [
            next(
                row
                for row in summaries
                if row["stage"] == 2
                and row["experiment"] == "subsidy_strategy"
                and row["budget"] == budget
                and row["network_label"] == selected_network
                and row["strategy"] == strategy
            )
            for strategy in strategies
        ]
        axes[0].bar(
            [pos + offset for pos in positions],
            [0.5 * (row["B_user_T"] + row["B_merchant_T"]) for row in rows],
            width=width,
            color=budget_colors[budget],
            label=f"B={budget:g}",
        )
        axes[1].bar(
            [pos + offset for pos in positions],
            [row["discounted_profit_B"] for row in rows],
            width=width,
            color=budget_colors[budget],
            label=f"B={budget:g}",
        )
    axes[0].axhline(0.5, color="gray", linewidth=1)
    axes[0].set_ylabel("B final average share")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("strong network: final share")
    axes[1].axhline(0, color="gray", linewidth=1)
    axes[1].set_ylabel("discounted profit of B")
    axes[1].set_title("strong network: discounted profit")
    for ax in axes:
        ax.set_xticks(list(positions), strategies, rotation=35, ha="right")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig10_stage2_strong_network_budget12.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    exit_rows = [
        row
        for row in summaries
        if row["stage"] == 2 and row["experiment"] == "stage_subsidy_exit"
    ]
    if exit_rows:
        budgets = sorted({row["budget"] for row in exit_rows})
        exit_times = sorted({row["exit_time"] for row in exit_rows})
        share_matrix = []
        profit_matrix = []
        for exit_time in exit_times:
            share_row = []
            profit_row = []
            for budget in budgets:
                row = next(
                    row
                    for row in exit_rows
                    if row["budget"] == budget and row["exit_time"] == exit_time
                )
                share_row.append(0.5 * (row["B_user_T"] + row["B_merchant_T"]))
                profit_row.append(row["discounted_profit_B"])
            share_matrix.append(share_row)
            profit_matrix.append(profit_row)

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(
            share_matrix,
            origin="lower",
            extent=[min(budgets), max(budgets), min(exit_times), max(exit_times)],
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=1,
        )
        ax.set_xlabel("total budget B during subsidy period")
        ax.set_ylabel("exit time Ts")
        ax.set_title("Stage 2 experiment 2: final B share after subsidy exit")
        fig.colorbar(image, ax=ax, label="final average share L_B")
        fig.tight_layout()
        path = output_dir / "fig_stage2_exp3_subsidy_exit_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

        fig, ax = plt.subplots(figsize=(8, 5))
        image = ax.imshow(
            profit_matrix,
            origin="lower",
            extent=[min(budgets), max(budgets), min(exit_times), max(exit_times)],
            aspect="auto",
            cmap="magma",
        )
        ax.set_xlabel("total budget B during subsidy period")
        ax.set_ylabel("exit time Ts")
        ax.set_title("Stage 2 experiment 2: discounted profit after subsidy exit")
        fig.colorbar(image, ax=ax, label="discounted profit of B")
        fig.tight_layout()
        path = output_dir / "fig_stage2_exp3_subsidy_exit_profit_heatmap.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for run_id, records in trajectories.items():
        if run_id in {
            "stage3_standard_user_only",
            "stage3_standard_merchant_only",
            "stage3_standard_balanced",
            "stage3_standard_merchant_leaning",
        }:
            label = run_id.replace("stage3_standard_", "")
            times = [row["time"] for row in records]
            axes[0].plot(
                times,
                [0.5 * (row["u_B_user"] + row["m_B_merchant"]) for row in records],
                label=label,
            )
            axes[1].plot(times, [row.get("C_B", 0.0) for row in records], label=label)
    axes[0].axhline(0.5, color="gray", linewidth=1)
    axes[0].set_title("B final-basis average share dynamics")
    axes[0].set_ylabel("average share")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("B congestion penalty")
    axes[1].set_ylabel("C_B")
    for ax in axes:
        ax.set_xlabel("time")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig6_stage3_congestion.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for run_id, records in trajectories.items():
        if run_id in {
            "stage4_medium_greedy",
            "stage4_medium_long_term",
            "stage4_medium_dynamic",
            "stage4_medium_quality_only",
        }:
            label = run_id.replace("stage4_medium_", "")
            times = [row["time"] for row in records]
            axes[0].plot(
                times,
                [0.5 * (row["u_B_user"] + row["m_B_merchant"]) for row in records],
                label=label,
            )
            axes[1].plot(times, [row["q_B"] for row in records], label=label)
    axes[0].axhline(0.5, color="gray", linewidth=1)
    axes[0].set_title("B average share")
    axes[0].set_ylabel("average share")
    axes[0].set_ylim(0, 1)
    axes[1].set_title("B user-side quality q_B")
    axes[1].set_ylabel("q_B")
    for ax in axes:
        ax.set_xlabel("time")
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "fig7_stage4_quality_investment.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    return written
