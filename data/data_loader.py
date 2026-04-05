"""
数据加载与预处理模块

支持两个数据集：
  - 交通数据集（Portland-Vancouver）：VHT、Occupancy、Volume，28 个区域，168 个周期
  - 空气质量数据集（北京）：PM2.5、PM10，35 个区域，168 个周期

主要功能：
  1. 加载原始 CSV 数据，转换为 GTD 矩阵 (m × t × n)
  2. 基于地理距离构建空间邻接矩阵
  3. 数据归一化（MinMax）
  4. 训练集 / 测试集划分（前 12 周期冷启动，后 156 周期执行）
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def build_adjacency_matrix(coords: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    基于地理坐标构建空间邻接矩阵。

    Args:
        coords: 形状 (m, 2) 的坐标数组，每行为 (经度, 纬度)。
        threshold: 归一化距离阈值，距离小于该值的区域视为相邻。

    Returns:
        形状 (m, m) 的二值邻接矩阵（不含自环）。
    """
    dist_matrix = cdist(coords, coords, metric='euclidean')
    # 归一化到 [0, 1]
    max_dist = dist_matrix.max()
    if max_dist > 0:
        dist_matrix = dist_matrix / max_dist
    adj = (dist_matrix < threshold).astype(np.float32)
    np.fill_diagonal(adj, 0)  # 去除自环
    return adj


def normalize_data(data: np.ndarray):
    """
    对 GTD 矩阵 (m × t × n) 逐任务做 MinMax 归一化。

    Returns:
        normalized_data: 归一化后的矩阵，形状同 data。
        scalers: 每个任务对应的 MinMaxScaler 列表，用于反归一化。
    """
    m, t, n = data.shape
    normalized = np.zeros_like(data, dtype=np.float32)
    scalers = []
    for k in range(n):
        scaler = MinMaxScaler()
        task_data = data[:, :, k]  # (m, t)
        normalized[:, :, k] = scaler.fit_transform(task_data)
        scalers.append(scaler)
    return normalized, scalers


# ─────────────────────────────────────────────────────────────────────────────
# 交通数据集（Portland-Vancouver）
# ─────────────────────────────────────────────────────────────────────────────

class TrafficDataLoader:
    """
    加载 Portland-Vancouver 交通数据。

    原始数据来源：PORTAL 交通数据存档
    https://portal.its.pdx.edu / https://adus.github.io/portal-documentation/

    数据格式预期（CSV）：
      列包含：area_id, cycle, VHT, Occupancy, Volume
    """

    TASK_NAMES = ["VHT", "Occupancy", "Volume"]
    M_AREAS = 28
    T_CYCLES = 168

    def __init__(self, data_dir: str, cold_start: int = 12,
                 adj_threshold: float = 0.5):
        """
        Args:
            data_dir: 数据目录，包含 traffic_data.csv 和 traffic_coords.csv。
            cold_start: 冷启动周期数。
            adj_threshold: 构建邻接矩阵的距离阈值。
        """
        self.data_dir = data_dir
        self.cold_start = cold_start
        self.adj_threshold = adj_threshold

    def load(self):
        """
        加载并预处理数据。

        Returns:
            gtd: 真实数据矩阵，形状 (m, t, n)，归一化后。
            adj: 邻接矩阵，形状 (m, m)。
            scalers: 反归一化用的 scaler 列表。
            cold_gtd: 冷启动数据，形状 (m, cold_start, n)。
            exec_gtd: 执行阶段数据，形状 (m, t-cold_start, n)。
        """
        data_path = os.path.join(self.data_dir, "traffic_data.csv")
        coords_path = os.path.join(self.data_dir, "traffic_coords.csv")

        df = pd.read_csv(data_path)
        gtd = self._build_gtd(df)

        # 归一化
        gtd, scalers = normalize_data(gtd)

        # 邻接矩阵
        if os.path.exists(coords_path):
            coords_df = pd.read_csv(coords_path)
            coords = coords_df[["lon", "lat"]].values.astype(np.float32)
        else:
            # 若无坐标文件，随机生成（用于调试）
            np.random.seed(42)
            coords = np.random.rand(self.M_AREAS, 2).astype(np.float32)
        adj = build_adjacency_matrix(coords, self.adj_threshold)

        cold_gtd = gtd[:, :self.cold_start, :]
        exec_gtd = gtd[:, self.cold_start:, :]
        return gtd, adj, scalers, cold_gtd, exec_gtd

    def _build_gtd(self, df: pd.DataFrame) -> np.ndarray:
        """将 DataFrame 转换为 GTD 矩阵 (m × t × n)。"""
        gtd = np.zeros((self.M_AREAS, self.T_CYCLES, len(self.TASK_NAMES)),
                       dtype=np.float32)
        for area_idx in range(self.M_AREAS):
            area_df = df[df["area_id"] == area_idx].sort_values("cycle")
            for k, task in enumerate(self.TASK_NAMES):
                if task in area_df.columns:
                    vals = area_df[task].values[:self.T_CYCLES]
                    gtd[area_idx, :len(vals), k] = vals
        return gtd


# ─────────────────────────────────────────────────────────────────────────────
# 空气质量数据集（北京）
# ─────────────────────────────────────────────────────────────────────────────

class AirQualityDataLoader:
    """
    加载北京多站点空气质量数据。

    原始数据来源：UCI Machine Learning Repository
    https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

    数据格式预期（CSV）：
      列包含：area_id, cycle, PM2.5, PM10
    """

    TASK_NAMES = ["PM2.5", "PM10"]
    M_AREAS = 35
    T_CYCLES = 168

    def __init__(self, data_dir: str, cold_start: int = 12,
                 adj_threshold: float = 0.5):
        """
        Args:
            data_dir: 数据目录，包含 air_quality_data.csv 和 air_quality_coords.csv。
            cold_start: 冷启动周期数。
            adj_threshold: 构建邻接矩阵的距离阈值。
        """
        self.data_dir = data_dir
        self.cold_start = cold_start
        self.adj_threshold = adj_threshold

    def load(self):
        """
        加载并预处理数据。

        Returns:
            gtd: 真实数据矩阵，形状 (m, t, n)，归一化后。
            adj: 邻接矩阵，形状 (m, m)。
            scalers: 反归一化用的 scaler 列表。
            cold_gtd: 冷启动数据，形状 (m, cold_start, n)。
            exec_gtd: 执行阶段数据，形状 (m, t-cold_start, n)。
        """
        data_path = os.path.join(self.data_dir, "air_quality_data.csv")
        coords_path = os.path.join(self.data_dir, "air_quality_coords.csv")

        df = pd.read_csv(data_path)
        gtd = self._build_gtd(df)

        # 归一化
        gtd, scalers = normalize_data(gtd)

        # 邻接矩阵
        if os.path.exists(coords_path):
            coords_df = pd.read_csv(coords_path)
            coords = coords_df[["lon", "lat"]].values.astype(np.float32)
        else:
            np.random.seed(42)
            coords = np.random.rand(self.M_AREAS, 2).astype(np.float32)
        adj = build_adjacency_matrix(coords, self.adj_threshold)

        cold_gtd = gtd[:, :self.cold_start, :]
        exec_gtd = gtd[:, self.cold_start:, :]
        return gtd, adj, scalers, cold_gtd, exec_gtd

    def _build_gtd(self, df: pd.DataFrame) -> np.ndarray:
        """将 DataFrame 转换为 GTD 矩阵 (m × t × n)。"""
        gtd = np.zeros((self.M_AREAS, self.T_CYCLES, len(self.TASK_NAMES)),
                       dtype=np.float32)
        for area_idx in range(self.M_AREAS):
            area_df = df[df["area_id"] == area_idx].sort_values("cycle")
            for k, task in enumerate(self.TASK_NAMES):
                if task in area_df.columns:
                    vals = area_df[task].values[:self.T_CYCLES]
                    gtd[area_idx, :len(vals), k] = vals
        return gtd


# ─────────────────────────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────────────────────────

def get_data_loader(dataset: str, data_dir: str, cold_start: int = 12,
                    adj_threshold: float = 0.5):
    """
    根据数据集名称返回对应的数据加载器实例。

    Args:
        dataset: 数据集名称，"traffic" 或 "air_quality"。
        data_dir: 数据目录路径。
        cold_start: 冷启动周期数。
        adj_threshold: 邻接矩阵距离阈值。

    Returns:
        对应的 DataLoader 实例（TrafficDataLoader 或 AirQualityDataLoader）。
    """
    if dataset == "traffic":
        return TrafficDataLoader(data_dir, cold_start, adj_threshold)
    elif dataset == "air_quality":
        return AirQualityDataLoader(data_dir, cold_start, adj_threshold)
    else:
        raise ValueError(f"未知数据集：{dataset}，请选择 'traffic' 或 'air_quality'")
