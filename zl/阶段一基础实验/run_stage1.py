from pathlib import Path

from experiments import (
    exp1_basic_dynamics,
    exp2_network_effect,
    exp3_initial_advantage,
    exp4_quality_threshold,
    exp5_subsidy_basic,
)
from model import PlatformParams
from plotting import setup_matplotlib


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()
    params = PlatformParams()

    print("Running stage 1 experiments with Logit + dynamic adjustment model...")
    exp1_basic_dynamics(params, fig_dir, table_dir)
    print("  done: experiment 1")
    exp2_network_effect(params, fig_dir, table_dir)
    print("  done: experiment 2")
    exp3_initial_advantage(params, fig_dir, table_dir)
    print("  done: experiment 3")
    exp4_quality_threshold(params, fig_dir, table_dir)
    print("  done: experiment 4")
    exp5_subsidy_basic(params, fig_dir, table_dir)
    print("  done: experiment 5")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to:  {table_dir}")


if __name__ == "__main__":
    main()
