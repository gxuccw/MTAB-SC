"""
北京空气质量数据下载与转换脚本

数据来源：Quotsoft 中国空气质量历史数据
  https://quotsoft.net/air/

功能：
  1. 自动从 Quotsoft 下载指定日期范围的北京空气质量 CSV 数据
  2. 合并多天数据，筛选 PM2.5 和 PM10 两种污染物
  3. 选取前 35 个有效监测站
  4. 输出标准格式：
     - air_quality_data.csv   （列：area_id, cycle, PM2.5, PM10）
     - air_quality_coords.csv （列：area_id, lon, lat）
  5. 缺失值处理：线性插值 + 前后填充

用法：
  python data/download_air_quality.py
  python data/download_air_quality.py --start-date 20240101 --days 7
  python data/download_air_quality.py --start-date 20240101 --days 7 --output-dir data/
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────────────────────
# 北京监测站经纬度坐标映射表（35 个站点）
# 坐标来源：北京市生态环境监测中心及公开地图数据
# ─────────────────────────────────────────────────────────────────────────────

STATION_COORDS = {
    "万寿西宫":    (116.366, 39.878),
    "定陵":        (116.220, 40.292),
    "东四":        (116.417, 39.929),
    "天坛":        (116.407, 39.886),
    "农展馆":      (116.461, 39.937),
    "官园":        (116.339, 39.929),
    "海淀区万柳":  (116.287, 39.987),
    "顺义新城":    (116.655, 40.127),
    "怀柔镇":      (116.628, 40.328),
    "昌平镇":      (116.230, 40.217),
    "奥体中心":    (116.397, 39.982),
    "古城":        (116.184, 39.914),
    "平谷镇":      (117.121, 40.143),
    "大兴":        (116.404, 39.718),
    "亦庄":        (116.506, 39.795),
    "通州":        (116.663, 39.886),
    "房山":        (116.136, 39.742),
    "延庆":        (115.972, 40.453),
    "密云":        (116.832, 40.370),
    "门头沟":      (116.106, 39.937),
    "永乐店":      (116.783, 39.712),
    "琉璃河":      (116.000, 39.581),
    "前门":        (116.395, 39.899),
    "永定门内":    (116.394, 39.876),
    "西直门北":    (116.349, 39.954),
    "南三环":      (116.368, 39.856),
    "东四环":      (116.483, 39.939),
    "北部新区":    (116.174, 40.090),
    "丰台花园":    (116.279, 39.863),
    "云岗":        (116.146, 39.824),
    "石景山古城":  (116.184, 39.914),
    "植物园":      (116.207, 39.998),
    "丰台小屯":    (116.288, 39.848),
    "榆垡":        (116.300, 39.520),
    "京东南永乐":  (116.783, 39.712),
}

# Quotsoft 数据下载 URL 模板
QUOTSOFT_URL_TEMPLATE = "https://quotsoft.net/air/data/beijing_all_{date}.csv"

# 最大重试次数
MAX_RETRIES = 3

# 请求超时（秒）
REQUEST_TIMEOUT = 30

# 未知站点随机坐标偏移范围（北京地理范围约：东西跨度 ~1.0°，南北跨度 ~0.6°）
BEIJING_LON_OFFSET = 0.5   # 经度偏移最大值（度）
BEIJING_LAT_OFFSET = 0.3   # 纬度偏移最大值（度）


# ─────────────────────────────────────────────────────────────────────────────
# 数据下载函数
# ─────────────────────────────────────────────────────────────────────────────

def download_daily_csv(date_str: str, save_dir: str = None) -> pd.DataFrame:
    """
    从 Quotsoft 下载指定日期的北京空气质量 CSV 数据。

    Args:
        date_str: 日期字符串，格式 YYYYMMDD，如 "20240101"。
        save_dir: 若指定，则将原始 CSV 保存到该目录下（可选）。

    Returns:
        包含当天所有监测站、所有污染物类型的 DataFrame。
        列：date, hour, type, <站点1>, <站点2>, ...

    Raises:
        RuntimeError: 下载或解析失败时抛出。
    """
    url = QUOTSOFT_URL_TEMPLATE.format(date=date_str)
    print(f"  正在下载：{url}")

    # 带重试的 HTTP 请求
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            break
        except requests.exceptions.Timeout:
            print(f"  [警告] 请求超时（第 {attempt}/{MAX_RETRIES} 次）")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"下载 {url} 超时，已重试 {MAX_RETRIES} 次")
            time.sleep(2 * attempt)
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP 错误：{e}（URL：{url}）")
        except requests.exceptions.ConnectionError as e:
            print(f"  [警告] 连接错误（第 {attempt}/{MAX_RETRIES} 次）：{e}")
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"无法连接到 {url}，请检查网络")
            time.sleep(2 * attempt)

    # 如果需要，保存原始 CSV
    if save_dir is not None:
        raw_path = os.path.join(save_dir, f"beijing_all_{date_str}.csv")
        with open(raw_path, "wb") as f:
            f.write(response.content)
        print(f"  原始文件已保存：{raw_path}")

    # 解析 CSV 内容
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        raise RuntimeError(f"解析 CSV 失败（{date_str}）：{e}")

    return df


def download_date_range(start_date: str, days: int,
                        save_dir: str = None) -> pd.DataFrame:
    """
    下载连续多天的数据并合并。

    Args:
        start_date: 起始日期，格式 YYYYMMDD。
        days: 下载天数（默认 7 天 = 168 小时）。
        save_dir: 原始 CSV 保存目录（可选）。

    Returns:
        合并后的 DataFrame，新增 `cycle` 列（从 0 开始的小时序号）。
    """
    start = datetime.strptime(start_date, "%Y%m%d")
    all_frames = []

    print(f"开始下载 {days} 天数据（起始：{start_date}）...")

    for i in range(days):
        current_date = start + timedelta(days=i)
        date_str = current_date.strftime("%Y%m%d")
        try:
            df = download_daily_csv(date_str, save_dir=save_dir)
            # 为每行添加全局周期序号（第 i 天的第 h 小时 → cycle = i*24 + h）
            df["cycle"] = df["hour"].astype(int) + i * 24
            all_frames.append(df)
            print(f"  ✓ {date_str}：{len(df)} 条记录")
        except RuntimeError as e:
            print(f"  ✗ {date_str} 下载失败：{e}，跳过该天")

    if not all_frames:
        raise RuntimeError("所有日期下载均失败，无法生成数据集")

    merged = pd.concat(all_frames, ignore_index=True)
    print(f"合并完成，共 {len(merged)} 条记录")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# 数据转换函数
# ─────────────────────────────────────────────────────────────────────────────

def select_stations(df: pd.DataFrame, max_stations: int = 35) -> list:
    """
    从原始 DataFrame 中选取有效监测站列名。

    Quotsoft CSV 中，除 date、hour、type、cycle 外，其余列均为监测站名称。
    若站点数超过 max_stations，优先选取坐标已知的站点，再补充其他站点。

    Args:
        df: 原始合并 DataFrame。
        max_stations: 最多选取的站点数。

    Returns:
        站点列名列表（长度 ≤ max_stations）。
    """
    # 排除元数据列，得到站点列
    meta_cols = {"date", "hour", "type", "cycle"}
    all_stations = [c for c in df.columns if c not in meta_cols]

    if not all_stations:
        raise ValueError("未找到任何监测站列，请检查原始数据格式")

    # 优先选坐标已知的站点
    known = [s for s in all_stations if s in STATION_COORDS]
    unknown = [s for s in all_stations if s not in STATION_COORDS]

    selected = known[:max_stations]
    if len(selected) < max_stations:
        # 补充坐标未知的站点
        selected += unknown[: max_stations - len(selected)]

    print(f"选取 {len(selected)} 个监测站（共发现 {len(all_stations)} 个）")
    return selected


def convert_to_standard_format(raw_df: pd.DataFrame,
                                stations: list,
                                t_cycles: int = 168) -> pd.DataFrame:
    """
    将 Quotsoft 原始格式转换为标准格式。

    原始格式：每行 = 一个时刻 × 一种污染物，各站点为列
    标准格式：每行 = 一个站点 × 一个周期，列为 area_id, cycle, PM2.5, PM10

    Args:
        raw_df: 原始合并 DataFrame。
        stations: 要保留的监测站列名列表。
        t_cycles: 总周期数（默认 168）。

    Returns:
        标准格式 DataFrame，列：area_id, cycle, PM2.5, PM10
    """
    # 筛选 PM2.5 和 PM10 两种污染物
    pm25_df = raw_df[raw_df["type"] == "PM2.5"][["cycle"] + stations].copy()
    pm10_df = raw_df[raw_df["type"] == "PM10"][["cycle"] + stations].copy()

    # 按周期排序并去重（取第一条）
    pm25_df = pm25_df.sort_values("cycle").drop_duplicates("cycle").set_index("cycle")
    pm10_df = pm10_df.sort_values("cycle").drop_duplicates("cycle").set_index("cycle")

    # 确保有完整的 168 个周期（不足的用 NaN 填充）
    all_cycles = pd.RangeIndex(t_cycles)
    pm25_df = pm25_df.reindex(all_cycles)
    pm10_df = pm10_df.reindex(all_cycles)

    # 构建标准格式 DataFrame
    records = []
    for area_idx, station in enumerate(stations):
        for cycle in range(t_cycles):
            pm25_val = pm25_df.loc[cycle, station] if station in pm25_df.columns else np.nan
            pm10_val = pm10_df.loc[cycle, station] if station in pm10_df.columns else np.nan
            records.append({
                "area_id": area_idx,
                "cycle":   cycle,
                "PM2.5":   pm25_val,
                "PM10":    pm10_val,
            })

    result_df = pd.DataFrame(records)
    return result_df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    处理缺失值：先线性插值，再用前后值填充边界缺失。

    Args:
        df: 标准格式 DataFrame，列：area_id, cycle, PM2.5, PM10

    Returns:
        缺失值处理后的 DataFrame。
    """
    filled_frames = []
    for area_id in df["area_id"].unique():
        area_df = df[df["area_id"] == area_id].copy().sort_values("cycle")
        # 对 PM2.5 和 PM10 分别进行插值
        for col in ["PM2.5", "PM10"]:
            area_df[col] = (
                area_df[col]
                .interpolate(method="linear", limit_direction="both")  # 线性插值
                .ffill()   # 前向填充（处理头部缺失）
                .bfill()   # 后向填充（处理尾部缺失）
            )
        filled_frames.append(area_df)

    return pd.concat(filled_frames, ignore_index=True)


def build_coords_csv(stations: list) -> pd.DataFrame:
    """
    生成监测站坐标文件。

    坐标优先使用内置 STATION_COORDS 映射表；
    对于未知站点，使用北京市中心坐标并加入随机偏移（仅作占位，不影响功能）。

    Args:
        stations: 监测站列名列表。

    Returns:
        coords DataFrame，列：area_id, lon, lat
    """
    np.random.seed(42)
    # 北京市中心参考坐标
    beijing_center = (116.405, 39.905)

    records = []
    for area_idx, station in enumerate(stations):
        if station in STATION_COORDS:
            lon, lat = STATION_COORDS[station]
        else:
            # 未知站点：在北京市范围内随机偏移
            lon = beijing_center[0] + np.random.uniform(-BEIJING_LON_OFFSET, BEIJING_LON_OFFSET)
            lat = beijing_center[1] + np.random.uniform(-BEIJING_LAT_OFFSET, BEIJING_LAT_OFFSET)
            print(f"  [警告] 站点 '{station}' 无内置坐标，使用随机偏移坐标")
        records.append({"area_id": area_idx, "lon": lon, "lat": lat})

    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从 Quotsoft 下载北京空气质量数据并转换为标准格式"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="起始日期，格式 YYYYMMDD（默认：7 天前）",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="下载天数（默认：7，即 168 小时）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(os.path.abspath(__file__)),
        help="输出目录（默认：脚本所在的 data/ 目录）",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        default=False,
        help="是否保存原始下载的 CSV 文件（默认：不保存）",
    )
    parser.add_argument(
        "--max-stations",
        type=int,
        default=35,
        help="最多选取的监测站数量（默认：35）",
    )
    args = parser.parse_args()

    # 确定起始日期（默认为 7 天前）
    if args.start_date is None:
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d")
    else:
        start_date = args.start_date

    # 验证日期格式
    try:
        datetime.strptime(start_date, "%Y%m%d")
    except ValueError:
        print(f"[错误] 日期格式无效：{start_date}，应为 YYYYMMDD")
        sys.exit(1)

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    raw_save_dir = args.output_dir if args.save_raw else None

    print("=" * 60)
    print("北京空气质量数据下载脚本")
    print(f"起始日期：{start_date}，下载天数：{args.days}")
    print(f"输出目录：{args.output_dir}")
    print("=" * 60)

    # 步骤1：下载数据
    raw_df = download_date_range(start_date, args.days, save_dir=raw_save_dir)

    # 步骤2：选取监测站
    stations = select_stations(raw_df, max_stations=args.max_stations)

    # 步骤3：转换为标准格式
    print("正在转换数据格式...")
    standard_df = convert_to_standard_format(raw_df, stations, t_cycles=args.days * 24)

    # 步骤4：处理缺失值
    print("正在处理缺失值...")
    standard_df = fill_missing_values(standard_df)

    # 步骤5：截取前 168 个周期（若 days > 7 则截断）
    t_cycles = min(args.days * 24, 168)
    standard_df = standard_df[standard_df["cycle"] < t_cycles]

    # 步骤6：保存标准格式数据文件
    data_output_path = os.path.join(args.output_dir, "air_quality_data.csv")
    standard_df.to_csv(data_output_path, index=False)
    print(f"✓ 数据文件已保存：{data_output_path}")
    print(f"  形状：{len(standard_df)} 行 × {len(standard_df.columns)} 列")

    # 步骤7：生成并保存坐标文件
    print("正在生成坐标文件...")
    coords_df = build_coords_csv(stations)
    coords_output_path = os.path.join(args.output_dir, "air_quality_coords.csv")
    coords_df.to_csv(coords_output_path, index=False)
    print(f"✓ 坐标文件已保存：{coords_output_path}")

    # 汇总统计
    print("\n" + "=" * 60)
    print("下载完成！数据统计：")
    print(f"  监测站数量：{len(stations)}")
    print(f"  周期数：{t_cycles}")
    print(f"  PM2.5 缺失率：{standard_df['PM2.5'].isna().mean():.2%}")
    print(f"  PM10  缺失率：{standard_df['PM10'].isna().mean():.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
