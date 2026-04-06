"""
评价指标模块

实现论文公式(1) MAPE（平均绝对百分比误差）：
  InferError(j) = sum_i sum_k |GTD^k[i,j] - GTD_bar^k[i,j]| / GTD^k[i,j] * 100 / s

  - 只计算未采集区域（selection_matrix = 0 的位置）
  - s 为未采集区域的总数量
"""

import numpy as np


def mape(ground_truth: np.ndarray,
         inferred: np.ndarray,
         selection_matrix: np.ndarray,
         eps: float = 0.1) -> float:
    """
    计算未采集区域的 MAPE（论文公式 1）。

    注意：eps 默认值设为 0.1，用于跳过归一化数据中真实值过小的区域，
    避免 MinMax 归一化后分母接近 0 导致 MAPE 爆炸。

    Args:
        ground_truth:     真实数据矩阵，形状 (m, n_tasks) 或 (m,)。
        inferred:         推断数据矩阵，形状与 ground_truth 相同。
        selection_matrix: 采集标记矩阵，1 = 已采集，0 = 未采集，形状同上。
        eps:              真实值绝对值低于此阈值的数据点将被跳过，防止除以接近 0 的值。

    Returns:
        MAPE 值（百分比，float）。如果未采集区域为空则返回 0.0。
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    inferred = np.asarray(inferred, dtype=np.float64)
    selection_matrix = np.asarray(selection_matrix, dtype=np.float64)

    # 未采集掩码，同时跳过真实值过小的区域（归一化数据中接近 0 的点）
    not_collected = (selection_matrix == 0) & (np.abs(ground_truth) >= eps)
    s = not_collected.sum()
    if s == 0:
        return 0.0

    # 只计算未采集且真实值足够大的区域
    gt_vals = ground_truth[not_collected]
    inf_vals = inferred[not_collected]

    error = np.abs(gt_vals - inf_vals) / np.abs(gt_vals)
    return float(error.sum() / s * 100.0)


def mape_per_task(ground_truth: np.ndarray,
                  inferred: np.ndarray,
                  selection_matrix: np.ndarray,
                  eps: float = 0.1) -> np.ndarray:
    """
    逐任务计算 MAPE。

    Args:
        ground_truth:     形状 (m, n_tasks)。
        inferred:         形状 (m, n_tasks)。
        selection_matrix: 形状 (m, n_tasks) 或 (m,)（若为 (m,) 则对所有任务使用同一掩码）。
        eps:              防除零值。

    Returns:
        形状 (n_tasks,) 的 numpy 数组，每个元素为对应任务的 MAPE。
    """
    n_tasks = ground_truth.shape[1]
    results = np.zeros(n_tasks, dtype=np.float64)
    for k in range(n_tasks):
        if selection_matrix.ndim == 2:
            sel_k = selection_matrix[:, k]
        else:
            sel_k = selection_matrix
        results[k] = mape(ground_truth[:, k], inferred[:, k], sel_k, eps)
    return results


def overall_mape(ground_truth: np.ndarray,
                 inferred: np.ndarray,
                 selection_matrix: np.ndarray,
                 eps: float = 0.1) -> float:
    """
    计算所有任务的整体平均 MAPE。

    Args:
        ground_truth:     形状 (m, n_tasks)。
        inferred:         形状 (m, n_tasks)。
        selection_matrix: 形状 (m, n_tasks) 或 (m,)。
        eps:              真实值绝对值低于此阈值的数据点将被跳过。

    Returns:
        所有任务 MAPE 的均值（float）。
    """
    per_task = mape_per_task(ground_truth, inferred, selection_matrix, eps)
    return float(per_task.mean())


def mae(ground_truth: np.ndarray,
        inferred: np.ndarray,
        selection_matrix: np.ndarray) -> float:
    """
    计算未采集区域的 MAE（平均绝对误差）。

    归一化数据上的 MAE 合理范围约 0.01-0.2，适合作为训练奖励信号，
    不会因真实值接近 0 而爆炸（与 MAPE 相比更稳定）。

    Args:
        ground_truth:     真实数据矩阵，形状 (m, n_tasks) 或 (m,)。
        inferred:         推断数据矩阵，形状与 ground_truth 相同。
        selection_matrix: 采集标记矩阵，1 = 已采集，0 = 未采集，形状同上。

    Returns:
        MAE 值（float）。如果未采集区域为空则返回 0.0。
    """
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    inferred = np.asarray(inferred, dtype=np.float64)
    selection_matrix = np.asarray(selection_matrix, dtype=np.float64)

    not_collected = (selection_matrix == 0)
    s = not_collected.sum()
    if s == 0:
        return 0.0

    gt_vals = ground_truth[not_collected]
    inf_vals = inferred[not_collected]
    return float(np.abs(gt_vals - inf_vals).sum() / s)


def mae_per_task(ground_truth: np.ndarray,
                 inferred: np.ndarray,
                 selection_matrix: np.ndarray) -> np.ndarray:
    """
    逐任务计算 MAE。

    Args:
        ground_truth:     形状 (m, n_tasks)。
        inferred:         形状 (m, n_tasks)。
        selection_matrix: 形状 (m, n_tasks) 或 (m,)（若为 (m,) 则对所有任务使用同一掩码）。

    Returns:
        形状 (n_tasks,) 的 numpy 数组，每个元素为对应任务的 MAE。
    """
    n_tasks = ground_truth.shape[1]
    results = np.zeros(n_tasks, dtype=np.float64)
    for k in range(n_tasks):
        if selection_matrix.ndim == 2:
            sel_k = selection_matrix[:, k]
        else:
            sel_k = selection_matrix
        results[k] = mae(ground_truth[:, k], inferred[:, k], sel_k)
    return results


def overall_mae(ground_truth: np.ndarray,
                inferred: np.ndarray,
                selection_matrix: np.ndarray) -> float:
    """
    计算所有任务的整体平均 MAE。

    适合用于训练过程中的奖励信号，归一化数据上数值稳定，
    合理范围约 0.01-0.2。

    Args:
        ground_truth:     形状 (m, n_tasks)。
        inferred:         形状 (m, n_tasks)。
        selection_matrix: 形状 (m, n_tasks) 或 (m,)。

    Returns:
        所有任务 MAE 的均值（float）。
    """
    per_task = mae_per_task(ground_truth, inferred, selection_matrix)
    return float(per_task.mean())
