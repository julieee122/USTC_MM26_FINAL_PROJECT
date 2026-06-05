from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from model import TwoSidedPlatformModel
from params import ABMParams


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    result_dir = base_dir / "results"
    fig_dir = result_dir / "figures"
    table_dir = result_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    params = ABMParams(alpha=3.0, beta=3.0, sU=0.2, sM=0.2, sigmaU=0.2, sigmaM=0.2)
    model = TwoSidedPlatformModel(n_users=1000, n_merchants=500, u0=0.55, m0=0.55, params=params, seed=2026)
    df = model.run_model(steps=60)
    df.to_csv(table_dir / "abm_basic_run.csv", encoding="utf-8-sig", index_label="step")

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["u_A"], label="用户份额 u_A")
    plt.plot(df.index, df["m_A"], label="商户份额 m_A")
    plt.plot(df.index, df["L_A"], linestyle="--", label="综合份额 L_A")
    plt.xlabel("离散时间步")
    plt.ylabel("平台 A 份额")
    plt.title("Mesa ABM：双边平台竞争基础仿真")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "abm_basic_dynamics.png", dpi=220)
    plt.close()

    print(df.tail(1).to_string())
    print(f"Saved table to: {table_dir / 'abm_basic_run.csv'}")
    print(f"Saved figure to: {fig_dir / 'abm_basic_dynamics.png'}")


if __name__ == "__main__":
    main()
