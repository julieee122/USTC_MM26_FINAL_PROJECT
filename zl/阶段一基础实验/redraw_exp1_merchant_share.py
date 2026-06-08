from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from model import PlatformParams, run_simulation


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    fig_dir = base_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    params = PlatformParams()
    cases = [
        {
            "label": "对称初始",
            "detail": r"$u_0=m_0=0.50$",
            "u0": 0.50,
            "m0": 0.50,
            "color": "#557C83",
            "lw": 2.2,
            "linestyle": "-",
        },
        {
            "label": "轻微领先",
            "detail": r"$u_0=m_0=0.55$",
            "u0": 0.55,
            "m0": 0.55,
            "color": "#E58F3A",
            "lw": 2.7,
            "linestyle": "-",
        },
        {
            "label": "明显领先",
            "detail": r"$u_0=m_0=0.70$",
            "u0": 0.70,
            "m0": 0.70,
            "color": "#2F9C5B",
            "lw": 2.7,
            "linestyle": "-",
        },
    ]

    setup_style()
    fig, ax = plt.subplots(figsize=(9.6, 5.8), constrained_layout=True)
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#FBFAF6")

    # Soft reference bands help the final equilibrium and neutral zone read quickly.
    ax.axhspan(0.80, 0.86, color="#2F9C5B", alpha=0.055, linewidth=0)
    ax.axhline(0.5, color="#6F7D7D", lw=1.1, ls=(0, (4, 4)), alpha=0.45)

    final_values: list[float] = []
    label_offsets = {
        "对称初始": -0.026,
        "轻微领先": -0.034,
        "明显领先": 0.034,
    }

    for case in cases:
        t, _u, m = run_simulation(params, u0=case["u0"], m0=case["m0"])
        final_values.append(float(m[-1]))

        ax.plot(
            t,
            m,
            color=case["color"],
            lw=case["lw"],
            ls=case["linestyle"],
            solid_capstyle="round",
            label=f"{case['label']}  {case['detail']}",
            zorder=3,
        )
        ax.scatter(
            [t[-1]],
            [m[-1]],
            s=34,
            color=case["color"],
            edgecolor="#FBFAF6",
            linewidth=1.1,
            zorder=4,
        )
        ax.text(
            t[-1] + 1.15,
            m[-1] + label_offsets[case["label"]],
            f"{case['label']}  {m[-1]:.3f}\n{case['detail']}",
            color=case["color"],
            fontsize=10.2,
            va="center",
            linespacing=1.25,
        )

    equilibrium = max(final_values)
    ax.axhline(equilibrium, color="#2F9C5B", lw=1.0, ls=(0, (2, 4)), alpha=0.42, zorder=1)
    ax.text(
        1.0,
        equilibrium + 0.018,
        f"领先情形稳态 ≈ {equilibrium:.3f}",
        color="#2F6F49",
        fontsize=10.5,
        va="bottom",
    )

    ax.set_title("实验 1：平台 A 商户份额动态演化", fontsize=18, pad=18, weight="bold")
    ax.set_xlabel("时间 t", fontsize=12.5, labelpad=10)
    ax.set_ylabel("平台 A 商户份额  m(t)", fontsize=12.5, labelpad=12)

    ax.set_xlim(-0.5, 64.0)
    ax.set_ylim(0.44, 0.88)
    ax.set_xticks(np.arange(0, 51, 10))
    ax.set_yticks(np.arange(0.45, 0.89, 0.05))
    ax.grid(axis="y", color="#CFC8BA", alpha=0.45, linewidth=0.85)
    ax.grid(axis="x", color="#D9D3C8", alpha=0.22, linewidth=0.75)

    ax.tick_params(axis="both", labelsize=10.5, colors="#2F3437")
    ax.spines["left"].set_color("#A9A197")
    ax.spines["bottom"].set_color("#A9A197")

    png_path = fig_dir / "exp1_merchant_share_dynamics_beautiful.png"
    svg_path = fig_dir / "exp1_merchant_share_dynamics_beautiful.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
