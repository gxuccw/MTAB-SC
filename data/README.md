# 数据集说明与下载指南

本目录用于存放 MTAB-SC 框架所需的原始数据文件。

---

## 交通数据集（Portland-Vancouver）

- **来源**：PORTAL 交通数据存档
  - 主站：<https://portal.its.pdx.edu>
  - 文档：<https://adus.github.io/portal-documentation/>
- **字段**：VHT（车辆行驶时间）、Occupancy（占有率）、Volume（交通量）
- **规模**：28 个感知区域，168 个周期（1 周，每小时 1 个周期）
- **预期文件**：
  - `traffic_data.csv`：包含列 `area_id, cycle, VHT, Occupancy, Volume`
  - `traffic_coords.csv`：包含列 `area_id, lon, lat`（感知区域地理坐标）

---

## 空气质量数据集（北京多站点）

- **来源**：UCI Machine Learning Repository
  - <https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data>
- **字段**：PM2.5、PM10
- **规模**：35 个感知区域（监测站），168 个周期（1 周，每小时 1 个周期）
- **预期文件**：
  - `air_quality_data.csv`：包含列 `area_id, cycle, PM2.5, PM10`
  - `air_quality_coords.csv`：包含列 `area_id, lon, lat`（监测站地理坐标）

---

## 数据预处理说明

1. 按上述格式准备好原始 CSV 文件，放入本目录（`data/`）。
2. `data_loader.py` 会自动完成以下处理：
   - 将原始数据转换为 GTD 矩阵，形状 `(m, t, n)`（m 个区域 × t 个周期 × n 个任务）。
   - 基于地理坐标构建空间邻接矩阵（欧式距离 + 归一化阈值过滤）。
   - 对每个任务独立执行 MinMax 归一化。
   - 将数据划分为：前 12 个周期（冷启动）+ 后 156 个周期（执行阶段）。
