"""Run stages 2-4 policy experiments.

Outputs:
    results/tables/stage234_summary.csv
    results/tables/stage234_summary.json
    results/tables/stage234_trajectories.csv
    results/figures/fig4_stage2_subsidy_share.png
    results/figures/fig5_stage2_profit.png
    results/figures/fig6_stage3_congestion.png
    results/figures/fig7_stage4_quality_investment.png
"""

from two_sided_platform.plots import plot_stage234_figures
from two_sided_platform.policy_experiments import save_stage234_outputs


def main():
    summaries, trajectories = save_stage234_outputs("results")
    figures = plot_stage234_figures(summaries, trajectories, "results/figures")

    stage2_count = sum(1 for row in summaries if row["stage"] == 2)
    stage3_count = sum(1 for row in summaries if row["stage"] == 3)
    stage4_count = sum(1 for row in summaries if row["stage"] == 4)

    print(
        "Stages 2-4 completed: "
        f"{stage2_count} stage-2 runs, "
        f"{stage3_count} stage-3 runs, "
        f"{stage4_count} stage-4 runs."
    )
    print("Summary table: results/tables/stage234_summary.csv")
    print("Trajectories: results/tables/stage234_trajectories.csv")
    if figures:
        print("Figures:")
        for path in figures:
            print(f"  {path}")
    else:
        print("Figures skipped: matplotlib is not installed.")


if __name__ == "__main__":
    main()
