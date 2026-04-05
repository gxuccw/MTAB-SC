"""
可视化工具模块

实现论文中的关键图表：
  - 不同预算下的 MAPE 对比图（类似论文图 13、14）
  - 训练过程的 MAPE 变化图（类似论文图 15）
  - MGSTNet vs GSTNet 对比图（类似论文图 16）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────────────────────
# 通用样式设置
# ─────────────────────────────────────────────────────────────────────────────

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]


def _set_style():
    plt.rcParams.update({
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 120,
    })


# ─────────────────────────────────────────────────────────────────────────────
# 不同预算下的 MAPE 对比图（图 13 / 图 14）
# ─────────────────────────────────────────────────────────────────────────────

def plot_mape_vs_budget(budgets: list,
                        method_results: dict,
                        title: str = "MAPE vs Budget",
                        ylabel: str = "MAPE (%)",
                        save_path: str = None):
    """
    绘制不同预算（x 轴）下各方法的 MAPE 对比曲线（对应论文图 13、14）。

    Args:
        budgets:        预算值列表，如 [5, 6, 7, 8, 9]。
        method_results: 字典，键为方法名，值为与 budgets 等长的 MAPE 列表。
                        示例：{"AB-CoDC": [30.1, 25.3, ...], "CoDC": [35.0, ...]}
        title:          图标题。
        ylabel:         y 轴标签。
        save_path:      若不为 None，则将图保存到该路径。
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (method, mape_vals) in enumerate(method_results.items()):
        color = _COLORS[i % len(_COLORS)]
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(budgets, mape_vals,
                label=method, color=color, marker=marker,
                linewidth=2, markersize=7)

    ax.set_xlabel("Budget", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(budgets)
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 训练过程 MAPE 变化图（图 15）
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(method_curves: dict,
                        title: str = "Training Convergence",
                        xlabel: str = "Epoch",
                        ylabel: str = "MAPE (%)",
                        smooth_window: int = 10,
                        save_path: str = None):
    """
    绘制训练过程中 MAPE 随 epoch 变化的曲线（对应论文图 15）。

    Args:
        method_curves: 字典，键为方法名，值为每个 epoch 的 MAPE 列表。
        title:         图标题。
        xlabel:        x 轴标签。
        ylabel:        y 轴标签。
        smooth_window: 滑动平均窗口大小（平滑曲线用）。
        save_path:     保存路径。
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (method, curve) in enumerate(method_curves.items()):
        color = _COLORS[i % len(_COLORS)]
        epochs = list(range(1, len(curve) + 1))
        # 滑动平均平滑
        smooth = _smooth(curve, smooth_window)
        ax.plot(epochs, smooth,
                label=method, color=color, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# MGSTNet vs GSTNet 对比图（图 16）
# ─────────────────────────────────────────────────────────────────────────────

def plot_mgstnet_vs_gstnet(budgets: list,
                           mgstnet_mape: list,
                           gstnet_mape: list,
                           title: str = "MGSTNet vs GSTNet",
                           save_path: str = None):
    """
    绘制 MGSTNet 与 GSTNet 在不同预算下的 MAPE 对比柱状图（对应论文图 16）。

    Args:
        budgets:       预算值列表，如 [5, 6, 7, 8, 9]。
        mgstnet_mape:  MGSTNet 对应的 MAPE 列表。
        gstnet_mape:   GSTNet 对应的 MAPE 列表。
        title:         图标题。
        save_path:     保存路径。
    """
    _set_style()
    x = np.arange(len(budgets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, mgstnet_mape, width,
                   label="MGSTNet", color=_COLORS[0], alpha=0.8)
    bars2 = ax.bar(x + width / 2, gstnet_mape, width,
                   label="GSTNet", color=_COLORS[1], alpha=0.8)

    ax.set_xlabel("Budget", fontsize=13)
    ax.set_ylabel("MAPE (%)", fontsize=13)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(values: list, window: int) -> list:
    """滑动平均平滑序列。"""
    if window <= 1:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result
