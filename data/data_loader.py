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

    原始数据来源：北京市生态环境监测中心（通过 Quotsoft 获取）
    主要来源：https://quotsoft.net/air/
    备用来源：https://github.com/HeQinWill/CNEMC
    原始来源：http://www.bjmemc.com.cn/

    包含北京市 35 个监测站的小时级 PM2.5、PM10 数据。
    可通过 data/download_air_quality.py 脚本自动下载并转换为标准格式。

    数据格式预期（标准 CSV）：
      列包含：area_id, cycle, PM2.5, PM10

    Quotsoft 原始 CSV 格式（可通过 load_from_raw_quotsoft() 直接读取）：
      列包含：date, hour, type（污染物类型）, <站点名1>, <站点名2>, ...
      type 取值：PM2.5、PM10 等
    """

    TASK_NAMES = ["PM2.5", "PM10"]
    # 与论文一致：35 个北京监测站
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
        加载并预处理标准格式数据（air_quality_data.csv）。

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
        # 缺失值处理：线性插值 + 前后填充
        df = self._fill_missing(df)
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

    def load_from_raw_quotsoft(self, raw_files: list,
                                max_stations: int = 35):
        """
        直接读取从 Quotsoft 下载的原始 CSV 文件，无需预先运行转换脚本。

        Quotsoft 原始 CSV 格式：
          每行 = 一个时刻 × 一种污染物
          列：date, hour, type, <站点名1>, <站点名2>, ...
          type 取值：PM2.5、PM10、SO2、NO2、CO、O3 等

        Args:
            raw_files: Quotsoft 原始 CSV 文件路径列表（按日期排序）。
                       例如 ["data/beijing_all_20240101.csv",
                              "data/beijing_all_20240102.csv", ...]
            max_stations: 最多使用的监测站数量（默认 35）。

        Returns:
            gtd: 真实数据矩阵，形状 (m, t, n)，归一化后。
            adj: 邻接矩阵，形状 (m, m)。
            scalers: 反归一化用的 scaler 列表。
            cold_gtd: 冷启动数据，形状 (m, cold_start, n)。
            exec_gtd: 执行阶段数据，形状 (m, t-cold_start, n)。
            stations: 实际使用的监测站名称列表。
        """
        # 读取并合并所有原始文件
        frames = []
        for i, fpath in enumerate(raw_files):
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"原始文件不存在：{fpath}")
            day_df = pd.read_csv(fpath)
            # 添加全局周期序号（第 i 天 × 24 + 当天小时）
            day_df["cycle"] = day_df["hour"].astype(int) + i * 24
            frames.append(day_df)

        raw_df = pd.concat(frames, ignore_index=True)

        # 确定要使用的监测站列
        meta_cols = {"date", "hour", "type", "cycle"}
        all_stations = [c for c in raw_df.columns if c not in meta_cols]
        # 优先选坐标已知的站点
        from data.download_air_quality import STATION_COORDS
        known = [s for s in all_stations if s in STATION_COORDS]
        unknown = [s for s in all_stations if s not in STATION_COORDS]
        stations = (known + unknown)[:max_stations]

        # 分别提取 PM2.5 和 PM10
        pm25_df = raw_df[raw_df["type"] == "PM2.5"][["cycle"] + stations].copy()
        pm10_df = raw_df[raw_df["type"] == "PM10"][["cycle"] + stations].copy()

        pm25_df = pm25_df.sort_values("cycle").drop_duplicates("cycle").set_index("cycle")
        pm10_df = pm10_df.sort_values("cycle").drop_duplicates("cycle").set_index("cycle")

        # 对齐到完整的 T_CYCLES 个周期
        all_cycles = pd.RangeIndex(self.T_CYCLES)
        pm25_df = pm25_df.reindex(all_cycles)
        pm10_df = pm10_df.reindex(all_cycles)

        # 构建标准格式 DataFrame
        records = []
        for area_idx, station in enumerate(stations):
            # 提前检查列是否存在，避免在循环内重复判断（O(n) 而非 O(n²)）
            has_pm25 = station in pm25_df.columns
            has_pm10 = station in pm10_df.columns
            for cycle in range(self.T_CYCLES):
                pm25_val = pm25_df.loc[cycle, station] if has_pm25 else np.nan
                pm10_val = pm10_df.loc[cycle, station] if has_pm10 else np.nan
                records.append({
                    "area_id": area_idx,
                    "cycle":   cycle,
                    "PM2.5":   pm25_val,
                    "PM10":    pm10_val,
                })

        df = pd.DataFrame(records)
        # 缺失值处理
        df = self._fill_missing(df)

        # 构建 GTD 矩阵（实际使用的站点数可能少于 M_AREAS）
        actual_m = len(stations)
        gtd = np.zeros((actual_m, self.T_CYCLES, len(self.TASK_NAMES)),
                       dtype=np.float32)
        for area_idx in range(actual_m):
            area_df = df[df["area_id"] == area_idx].sort_values("cycle")
            for k, task in enumerate(self.TASK_NAMES):
                if task in area_df.columns:
                    vals = area_df[task].values[:self.T_CYCLES]
                    gtd[area_idx, :len(vals), k] = vals

        # 归一化
        gtd, scalers = normalize_data(gtd)

        # 邻接矩阵（使用内置坐标）
        coords_path = os.path.join(self.data_dir, "air_quality_coords.csv")
        if os.path.exists(coords_path):
            coords_df = pd.read_csv(coords_path)
            coords = coords_df[["lon", "lat"]].values[:actual_m].astype(np.float32)
        else:
            # 从内置坐标表构建
            from data.download_air_quality import STATION_COORDS, build_coords_csv
            coords_df = build_coords_csv(stations)
            coords = coords_df[["lon", "lat"]].values.astype(np.float32)

        adj = build_adjacency_matrix(coords, self.adj_threshold)

        cold_gtd = gtd[:, :self.cold_start, :]
        exec_gtd = gtd[:, self.cold_start:, :]
        return gtd, adj, scalers, cold_gtd, exec_gtd, stations

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对标准格式 DataFrame 进行缺失值处理。

        处理策略：按 area_id 分组，对 PM2.5 和 PM10 列分别进行
        线性插值，再用前向/后向填充处理首尾缺失。

        Args:
            df: 标准格式 DataFrame，列：area_id, cycle, PM2.5, PM10

        Returns:
            缺失值处理后的 DataFrame。
        """
        filled_frames = []
        for area_id in df["area_id"].unique():
            area_df = df[df["area_id"] == area_id].copy().sort_values("cycle")
            for col in self.TASK_NAMES:
                if col in area_df.columns:
                    area_df[col] = (
                        area_df[col]
                        .interpolate(method="linear", limit_direction="both")
                        .ffill()
                        .bfill()
                    )
            filled_frames.append(area_df)

        if filled_frames:
            return pd.concat(filled_frames, ignore_index=True)
        return df

    def _build_gtd(self, df: pd.DataFrame) -> np.ndarray:
        """将标准格式 DataFrame 转换为 GTD 矩阵 (m × t × n)。"""
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
