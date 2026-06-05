from pathlib import Path

from plotting import setup_matplotlib
from stage2_experiments import (
    exp6_lockin_region,
    exp7_reversal_threshold,
    exp8_subsidy_allocation,
    exp9_subsidy_exit,
    exp10_profit_constraint,
)

import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE1_DIR = PROJECT_DIR / "阶段一基础实验"
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

from model import PlatformParams  # noqa: E402


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    table_dir = base_dir / "results" / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    setup_matplotlib()
    params = PlatformParams()

    print("Running stage 2 experiments...")
    exp6_lockin_region(params, fig_dir, table_dir)
    print("  done: experiment 6")
    exp7_reversal_threshold(params, fig_dir, table_dir)
    print("  done: experiment 7")
    exp8_subsidy_allocation(params, fig_dir, table_dir)
    print("  done: experiment 8")
    exp9_subsidy_exit(params, fig_dir, table_dir)
    print("  done: experiment 9")
    exp10_profit_constraint(params, fig_dir, table_dir)
    print("  done: experiment 10")
    print(f"Figures saved to: {fig_dir}")
    print(f"Tables saved to:  {table_dir}")


if __name__ == "__main__":
    main()
