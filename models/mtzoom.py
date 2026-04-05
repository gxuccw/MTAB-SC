"""
MTZOOM — 多任务训练数据更新模块

对应论文 Section IV-A。

核心思想：
  城市数据受天气、节假日等因素影响，历史数据不能准确反映未来状态。
  MTZOOM 从两个维度对训练数据进行实时更新：
    1. 时空渐变性（Spatio-Temporal Gradual Change）：相邻周期数据分布相似
    2. 数据集间相似性（Inter-dataset Similarity）：不同类型数据在同一区域强相关

更新公式（论文公式 5）：
  y_{k,i} = TD^k[i,-1] * (λ_t * avg(SGTD^k[:,-1]) / avg(TD^k[:,-1])
             + Σ_{n≠k} λ_s^n * SGTD^n[i,-1] / TD^n[i,-1])

  - 有采集数据的区域：结合时空渐变和数据相似性更新
  - 无采集数据的区域：仅使用时空渐变更新
  - 滑动窗口：移除最早的时间步，加入最新估计值

权重规则（λ_t + Σλ_s^n = 1）：
  - 当只有 1 种额外数据时：λ_t = λ_s = 0.5
  - 当有 2 种额外数据时：λ_t = 0.5，λ_s^n = 0.25
"""

import numpy as np


class MTZOOM:
    """
    多任务训练数据更新（Multi-Task ZOOM）。

    使用滑动窗口维护每个任务的训练数据 TD^k，
    每个感知周期利用稀疏采集数据 SGTD^k 对训练数据进行更新。
    """

    def __init__(self, n_tasks: int, lambda_t: float = 0.5):
        """
        Args:
            n_tasks: 任务数 n。
            lambda_t: 时空渐变权重 λ_t，默认 0.5。
                      剩余权重 (1 - λ_t) 均分给所有其他任务的相似性权重。
        """
        self.n_tasks = n_tasks
        self.lambda_t = lambda_t

        # 当只有 1 种额外数据时，λ_s = 0.5；有 2 种时，各为 0.25
        if n_tasks > 1:
            self.lambda_s = (1.0 - lambda_t) / (n_tasks - 1)
        else:
            self.lambda_s = 0.0

    def update(self,
               td_list: list,
               sgtd_list: list,
               selection_list: list) -> list:
        """
        执行一次训练数据滑动窗口更新。

        Args:
            td_list:        长度为 n 的列表，每个元素是 TD^k，形状 (m, j)。
                            j 为当前训练数据窗口长度（td_length）。
            sgtd_list:      长度为 n 的列表，每个元素是当前周期的稀疏采集数据 SGTD^k，
                            形状 (m,)，未采集区域填 0。
            selection_list: 长度为 n 的列表，每个元素是采集标记向量，形状 (m,)，
                            1=已采集，0=未采集。

        Returns:
            new_td_list: 更新后的训练数据列表，每个元素形状 (m, j)
                         （最旧列被移除，新估计列被追加）。
        """
        n = self.n_tasks
        m = td_list[0].shape[0]
        new_last_cols = []  # 每个任务估计出的新一列数据

        for k in range(n):
            td_k = td_list[k]            # (m, j)
            sgtd_k = sgtd_list[k]        # (m,)  当前周期稀疏采集值
            sel_k = selection_list[k]    # (m,)  采集标记

            # 当前训练数据最后一列（最近历史）
            td_last = td_k[:, -1]        # (m,)

            # ── 时空渐变比率 ──────────────────────────────────────────────────
            avg_sgtd_k = np.mean(sgtd_k[sel_k > 0]) if sel_k.sum() > 0 else 1.0
            avg_td_last = np.mean(td_last) if np.mean(td_last) != 0 else 1.0
            grad_ratio = avg_sgtd_k / avg_td_last  # 标量，全局渐变比率

            # ── 逐区域估计 ────────────────────────────────────────────────────
            y_k = np.zeros(m, dtype=np.float32)
            for i in range(m):
                td_ki = td_last[i]
                if td_ki == 0:
                    td_ki = 1e-8  # 防止除零

                if sel_k[i] > 0:
                    # 有采集数据：结合时空渐变 + 数据集间相似性
                    inter_term = 0.0
                    for n_idx in range(n):
                        if n_idx == k:
                            continue
                        sgtd_n = sgtd_list[n_idx]    # (m,)
                        td_n = td_list[n_idx][:, -1]  # (m,)
                        denom = td_n[i] if td_n[i] != 0 else 1e-8
                        inter_term += self.lambda_s * (sgtd_n[i] / denom)

                    y_k[i] = td_ki * (self.lambda_t * grad_ratio + inter_term)
                else:
                    # 无采集数据：仅使用时空渐变
                    y_k[i] = td_ki * self.lambda_t * grad_ratio

            new_last_cols.append(y_k)

        # ── 滑动窗口：移除最早时间步，追加新估计列 ────────────────────────────
        new_td_list = []
        for k in range(n):
            new_col = new_last_cols[k].reshape(m, 1)   # (m, 1)
            new_td = np.concatenate([td_list[k][:, 1:], new_col], axis=1)
            new_td_list.append(new_td.astype(np.float32))

        return new_td_list

    def initialize_td(self, gtd: np.ndarray, window: int) -> list:
        """
        用冷启动阶段数据初始化训练数据窗口。

        Args:
            gtd:    真实数据矩阵，形状 (m, t_cold, n)。
            window: 训练数据窗口长度（td_length）。

        Returns:
            td_list: 长度为 n 的列表，每个元素形状 (m, window)。
        """
        m, t, n = gtd.shape
        td_list = []
        for k in range(n):
            # 取冷启动阶段最后 window 个周期
            start = max(0, t - window)
            td_k = gtd[:, start:, k].astype(np.float32)   # (m, <=window)
            # 若冷启动数据不足，前向填充
            if td_k.shape[1] < window:
                pad = np.repeat(td_k[:, :1], window - td_k.shape[1], axis=1)
                td_k = np.concatenate([pad, td_k], axis=1)
            td_list.append(td_k)
        return td_list
