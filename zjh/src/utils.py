import os
import csv
import numpy as np
import matplotlib.pyplot as plt


def setup_matplotlib():
    """
    设置 matplotlib 支持中文显示。
    """
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_dir(path: str):
    """
    若文件夹不存在，则创建。
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics_csv(metrics, path: str):
    """
    保存策略比较指标。
    """
    if len(metrics) == 0:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)


def save_timeseries_csv(result, path: str):
    """
    保存单条模拟轨迹的完整时间序列数据。
    """
    keys = list(result.keys())
    n = len(result[keys[0]])

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(keys)

        for i in range(n):
            writer.writerow([result[k][i] for k in keys])


def summarize_result(name: str, result):
    """
    汇总一条策略轨迹的最终份额、成本收益、供给不足等指标。
    """
    return {
        "name": name,
        "final_x": float(result["x"][-1]),
        "final_y": float(result["y"][-1]),
        "final_q_u": float(result["q_u"][-1]),
        "final_q_m": float(result["q_m"][-1]),
        "avg_shortage_A": float(np.mean(result["shortage_A"])),
        "max_shortage_A": float(np.max(result["shortage_A"])),
        "cum_revenue": float(result["cum_revenue"][-1]),
        "cum_subsidy_cost": float(result["cum_subsidy_cost"][-1]),
        "cum_invest_cost": float(result["cum_invest_cost"][-1]),
        "cum_total_cost": float(result["cum_total_cost"][-1]),
        "cum_profit": float(result["cum_profit"][-1]),
    }