from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
            "axes.spines.left": False,
        }
    )


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def short_label(case: str) -> str:
    replacements = {
        "双边同步领先": "双边同步领先",
        "用户领先商户落后：A总份额略占优": "用户先行 / A 略占优",
        "用户领先商户落后：B总份额略占优": "用户先行 / B 略占优",
        "商户领先用户落后：A总份额略占优": "商户先行 / A 略占优",
        "商户领先用户落后：B总份额略占优": "商户先行 / B 略占优",
        "双边同步落后": "双边同步落后",
    }
    return replacements.get(case, case)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    table_path = base_dir / "results" / "tables" / "exp3_asymmetric_initial_cases.csv"
    fig_dir = base_dir / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(table_path)
    shares = np.array([float(row["L_A"]) for row in rows])
    initial_pairs = [f"u0={float(row['u0']):.4g}, m0={float(row['m0']):.4g}" for row in rows]
    labels = [f"{short_label(row['case'])}\n{pair}" for row, pair in zip(rows, initial_pairs)]

    # Center around 0.5 so A-lock-in and B-lock-in become visually symmetric.
    centered = shares - 0.5
    y = np.arange(len(rows))[::-1]
    colors = np.where(shares >= 0.5, "#2F9C5B", "#D95F59")

    setup_style()
    fig, ax = plt.subplots(figsize=(10.2, 6.1), constrained_layout=True)
    fig.patch.set_facecolor("#F7F5EF")
    ax.set_facecolor("#FBFAF6")

    ax.axvspan(-0.5, 0, color="#D95F59", alpha=0.07, linewidth=0)
    ax.axvspan(0, 0.5, color="#2F9C5B", alpha=0.07, linewidth=0)
    ax.axvline(0, color="#687070", lw=1.25, ls=(0, (4, 4)), alpha=0.7)

    bars = ax.barh(y, centered, height=0.56, color=colors, alpha=0.92, edgecolor="none")
    ax.scatter(centered, y, s=62, color=colors, edgecolor="#FBFAF6", linewidth=1.3, zorder=3)

    for bar, share in zip(bars, shares):
        yy = bar.get_y() + bar.get_height() / 2
        state_text = "A 锁定" if share >= 0.5 else "B 锁定"
        x_text = 0.245 if share >= 0.5 else -0.245
        ax.text(
            x_text,
            yy,
            f"{state_text}\n$L_A$={share:.3f}",
            ha="center",
            va="center",
            fontsize=10.8,
            color="white",
            weight="bold",
            linespacing=1.2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.7, linespacing=1.25)
    ax.set_xlim(-0.56, 0.56)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xticklabels(["B 完全锁定", "B 优势", "均势线\nL_A=0.5", "A 优势", "A 完全锁定"], fontsize=10)
    ax.grid(axis="x", color="#D8D1C5", alpha=0.45, linewidth=0.85)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", colors="#394041")
    ax.spines["bottom"].set_color("#A9A197")

    ax.set_title("实验 3 补充：非同步初始优势的锁定方向", fontsize=17.5, pad=18, weight="bold")
    ax.set_xlabel("最终综合份额相对均势线的偏离  $L_A - 0.5$", fontsize=12.2, labelpad=12)

    ax.text(
        -0.5,
        len(rows) - 0.36,
        "平台 B 获胜区域",
        color="#A94844",
        fontsize=10.5,
        va="top",
    )
    ax.text(
        0.5,
        len(rows) - 0.36,
        "平台 A 获胜区域",
        color="#247B46",
        fontsize=10.5,
        va="top",
        ha="right",
    )

    png_path = fig_dir / "exp3_asymmetric_initial_cases_beautiful.png"
    svg_path = fig_dir / "exp3_asymmetric_initial_cases_beautiful.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


if __name__ == "__main__":
    main()
