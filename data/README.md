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

- **主要来源**：Quotsoft 中国空气质量历史数据
  - <https://quotsoft.net/air/>
  - 包含北京市所有监测站（35 个）的小时级 PM2.5、PM10 数据
  - CSV 格式，可按日下载，例如：
    `https://quotsoft.net/air/data/beijing_all_20240101.csv`
  - 需要下载连续 7 天（168 小时）的数据
- **备用来源**：CNEMC GitHub 工具
  - <https://github.com/HeQinWill/CNEMC>
  - 可自动化抓取中国环境监测总站数据
- **原始来源**：北京市生态环境监测中心
  - <http://www.bjmemc.com.cn/>
- **字段**：PM2.5、PM10
- **规模**：35 个感知区域（监测站），168 个周期（1 周，每小时 1 个周期）
- **预期文件**：
  - `air_quality_data.csv`：包含列 `area_id, cycle, PM2.5, PM10`
  - `air_quality_coords.csv`：包含列 `area_id, lon, lat`（监测站地理坐标）

### 数据下载与预处理步骤

#### 方式一：使用自动下载脚本（推荐）

本项目提供了 `data/download_air_quality.py` 脚本，可自动完成数据下载、合并和格式转换：

```bash
# 默认下载最近 7 天数据（168 小时）
python data/download_air_quality.py

# 指定起始日期和天数
python data/download_air_quality.py --start-date 20240101 --days 7

# 指定输出目录
python data/download_air_quality.py --start-date 20240101 --days 7 --output-dir data/
```

脚本会自动生成：
- `air_quality_data.csv`（列：`area_id, cycle, PM2.5, PM10`）
- `air_quality_coords.csv`（列：`area_id, lon, lat`）

#### 方式二：手动下载并转换

1. 从 Quotsoft 下载连续 7 天的 CSV 文件（文件名格式 `beijing_all_YYYYMMDD.csv`）：
   ```
   https://quotsoft.net/air/data/beijing_all_20240101.csv
   https://quotsoft.net/air/data/beijing_all_20240102.csv
   ...（共 7 个文件）
   ```

2. 将下载的文件放入本目录。

3. 使用 `data_loader.py` 中的 `load_from_raw_quotsoft()` 方法直接加载原始文件，
   或运行下载脚本将文件转换为标准格式。

#### Quotsoft 原始 CSV 格式说明

Quotsoft 下载的原始文件格式如下：

| date | hour | type | 万寿西宫 | 定陵 | 东四 | … |
|------|------|------|----------|------|------|---|
| 20240101 | 0 | PM2.5 | 35.0 | 12.0 | 45.0 | … |
| 20240101 | 0 | PM10  | 60.0 | 20.0 | 80.0 | … |

- `type` 列：污染物类型（如 `PM2.5`、`PM10`）
- 其余各列：各监测站名称（如 `万寿西宫`、`定陵`、`东四` 等）

---

## 数据预处理说明

1. 按上述格式准备好原始 CSV 文件，放入本目录（`data/`）。
2. `data_loader.py` 会自动完成以下处理：
   - 将原始数据转换为 GTD 矩阵，形状 `(m, t, n)`（m 个区域 × t 个周期 × n 个任务）。
   - 基于地理坐标构建空间邻接矩阵（欧式距离 + 归一化阈值过滤）。
   - 对每个任务独立执行 MinMax 归一化。
   - 将数据划分为：前 12 个周期（冷启动）+ 后 156 个周期（执行阶段）。
   - 缺失值处理：线性插值 + 前后填充。
