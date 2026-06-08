"""Run stage 1 basic dynamic model experiments.

Outputs:
    results/tables/stage1_summary.csv
    results/tables/stage1_summary.json
    results/tables/stage1_trajectories.csv
    results/figures/*.png when matplotlib is installed
"""

from two_sided_platform.experiments import save_stage1_outputs
from two_sided_platform.plots import plot_stage1_figures


def main():
    summaries, trajectories = save_stage1_outputs("results")
    figures = plot_stage1_figures(summaries, trajectories, "results/figures")

    exp1 = [row for row in summaries if row["experiment"] == "network_initial_scale"]
    exp2 = [row for row in summaries if row["experiment"] == "quality_break_lock"]
    print(f"Stage 1 completed: {len(exp1)} experiment-1 runs, {len(exp2)} experiment-2 runs.")
    print("Summary table: results/tables/stage1_summary.csv")
    print("Trajectories: results/tables/stage1_trajectories.csv")
    if figures:
        print("Figures:")
        for path in figures:
            print(f"  {path}")
    else:
        print("Figures skipped: matplotlib is not installed.")


if __name__ == "__main__":
    main()

