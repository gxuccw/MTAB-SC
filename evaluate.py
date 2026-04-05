"""
MTAB-SC 评估脚本

功能：
  - 计算各任务和整体的 MAPE（只计算未采集区域，公式 1）
  - 支持不同预算（5-9）下的性能对比
  - 支持与基线方法（RANDOM、GREEDY、CoDC 等）的对比
  - 可视化不同预算下的 MAPE 曲线

用法示例：
  python evaluate.py --dataset traffic --checkpoint checkpoints/ --budgets 5 6 7 8 9
"""

import argparse
import os
import random
import numpy as np
import torch

from config import config
from data.data_loader import get_data_loader
from models.mtzoom import MTZOOM
from models.mgstnet import MGSTNet
from agents.budget_agent import BudgetAgent
from agents.collection_agent import CollectionAgents
from utils.metrics import mape_per_task, overall_mape
from utils.visualization import plot_mape_vs_budget


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MTAB-SC 评估脚本")
    parser.add_argument("--dataset", type=str, default="traffic",
                        choices=["traffic", "air_quality"],
                        help="评估数据集")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录")
    parser.add_argument("--checkpoint", type=str, default="checkpoints",
                        help="模型检查点目录")
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=[5, 6, 7, 8, 9],
                        help="评估的预算值列表")
    parser.add_argument("--n_eval_cycles", type=int, default=20,
                        help="评估的感知周期数")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="结果图保存路径（可选）")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# 基线方法
# ─────────────────────────────────────────────────────────────────────────────

def random_collection(m_areas: int, budget: int,
                      gtd_current: np.ndarray,
                      n_tasks: int) -> tuple:
    """
    RANDOM 基线：随机选择区域采集所有任务。

    Returns:
        (sgtd_list, coverage_list)：各任务稀疏采集数据和覆盖向量。
    """
    selected = random.sample(range(m_areas), min(budget, m_areas))
    sgtd_list = []
    coverage_list = []
    for k in range(n_tasks):
        sgtd = np.zeros(m_areas, dtype=np.float32)
        cov = np.zeros(m_areas, dtype=np.float32)
        for idx in selected:
            sgtd[idx] = gtd_current[idx, k]
            cov[idx] = 1.0
        sgtd_list.append(sgtd)
        coverage_list.append(cov)
    return sgtd_list, coverage_list


def greedy_multi_collection(m_areas: int, budget: int,
                             gtd_current: np.ndarray,
                             td_list: list,
                             n_tasks: int) -> tuple:
    """
    GREEDY-M 基线：贪心选择与历史训练数据差异最大的区域（多任务联合）。

    Returns:
        (sgtd_list, coverage_list)
    """
    # 计算各区域各任务的历史均值（最后一时间步）
    scores = np.zeros(m_areas, dtype=np.float32)
    for k in range(n_tasks):
        hist_last = td_list[k][:, -1]  # (m,)
        diff = np.abs(gtd_current[:, k] - hist_last)
        scores += diff

    # 选取得分最高的 budget 个区域
    selected = np.argsort(scores)[::-1][:budget]
    sgtd_list = []
    coverage_list = []
    for k in range(n_tasks):
        sgtd = np.zeros(m_areas, dtype=np.float32)
        cov = np.zeros(m_areas, dtype=np.float32)
        for idx in selected:
            sgtd[idx] = gtd_current[idx, k]
            cov[idx] = 1.0
        sgtd_list.append(sgtd)
        coverage_list.append(cov)
    return sgtd_list, coverage_list


# ─────────────────────────────────────────────────────────────────────────────
# 评估单个预算下的 MAPE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_budget(method: str,
                    budget: int,
                    exec_gtd: np.ndarray,
                    td_list: list,
                    mgstnet: MGSTNet,
                    adj: np.ndarray,
                    mtzoom: MTZOOM,
                    cfg_dataset: dict,
                    n_eval_cycles: int,
                    device: str,
                    budget_agent: BudgetAgent = None,
                    collection_agents: CollectionAgents = None) -> float:
    """
    对指定预算和方法进行 n_eval_cycles 周期的评估，返回平均 MAPE。

    Args:
        method:            方法名，支持 "AB-CoDC"、"RANDOM"、"GREEDY-M"。
        budget:            总预算 B_total。
        exec_gtd:          执行阶段数据，形状 (m, T, n)。
        td_list:           初始训练数据列表。
        mgstnet:           已训练的 MGSTNet。
        adj:               邻接矩阵。
        mtzoom:            MTZOOM 实例。
        cfg_dataset:       数据集配置字典。
        n_eval_cycles:     评估周期数。
        device:            计算设备。
        budget_agent:      预训练的预算分配智能体（AB-CoDC 专用）。
        collection_agents: 预训练的数据采集智能体（AB-CoDC 专用）。

    Returns:
        n_eval_cycles 周期的平均 MAPE（%）。
    """
    m_areas = cfg_dataset["m_areas"]
    n_tasks = cfg_dataset["n_tasks"]
    td_local = [td.copy() for td in td_list]  # 深拷贝，不影响原始数据
    T = exec_gtd.shape[1]

    total_mape = 0.0
    for cycle_idx in range(min(n_eval_cycles, T)):
        gtd_current = exec_gtd[:, cycle_idx, :]  # (m, n_tasks)

        if method == "AB-CoDC" and budget_agent and collection_agents:
            # 使用训练好的智能体（贪心模式，ε=0）
            eps_saved = collection_agents.epsilon
            collection_agents.epsilon = 0.0

            # 预算分配
            budget_state = np.zeros(n_tasks, dtype=np.float32)
            remaining = budget
            allocated = []
            for k in range(n_tasks):
                bk = budget_agent.select_action(budget_state, remaining)
                bk = min(bk, remaining)
                allocated.append(bk)
                remaining = max(0, remaining - bk)
                budget_state[k] = bk

            # 协同数据采集
            sgtd_list = [np.zeros(m_areas, dtype=np.float32) for _ in range(n_tasks)]
            coverage_list = [np.zeros(m_areas, dtype=np.float32) for _ in range(n_tasks)]
            budget_rem = allocated.copy()
            max_b = max(allocated) if allocated else 1

            for t in range(1, max_b + 1):
                obs_list = [
                    CollectionAgents.encode_obs(
                        [i for i, v in enumerate(coverage_list[k]) if v > 0],
                        coverage_list[k]
                    )
                    for k in range(n_tasks)
                ]
                actions = collection_agents.select_actions(obs_list, budget_rem)
                for k in range(n_tasks):
                    a = actions[k]
                    if a <= m_areas and budget_rem[k] > 0:
                        sgtd_list[k][a - 1] = gtd_current[a - 1, k]
                        coverage_list[k][a - 1] = 1.0
                        budget_rem[k] -= 1

            collection_agents.epsilon = eps_saved

        elif method == "RANDOM":
            sgtd_list, coverage_list = random_collection(
                m_areas, budget, gtd_current, n_tasks)

        elif method == "GREEDY-M":
            sgtd_list, coverage_list = greedy_multi_collection(
                m_areas, budget, gtd_current, td_local, n_tasks)

        else:
            # 默认随机
            sgtd_list, coverage_list = random_collection(
                m_areas, budget, gtd_current, n_tasks)

        # 推断
        input_td = [
            np.concatenate([
                td_local[k][:, 1:],
                sgtd_list[k].reshape(m_areas, 1)
            ], axis=1)
            for k in range(n_tasks)
        ]
        inferred = mgstnet.infer(input_td, adj, device)  # (m, n_tasks)

        # 选择矩阵
        sel_matrix = np.stack(coverage_list, axis=1)  # (m, n_tasks)

        cycle_mape = overall_mape(gtd_current, inferred, sel_matrix)
        total_mape += cycle_mape

        # 更新训练数据（MTZOOM）
        td_local = mtzoom.update(td_local, sgtd_list, coverage_list)

    return total_mape / min(n_eval_cycles, T) if n_eval_cycles > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 主评估函数
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args):
    cfg_dataset = config[args.dataset]
    m_areas = cfg_dataset["m_areas"]
    n_tasks = cfg_dataset["n_tasks"]
    cold_start = cfg_dataset["cold_start_cycles"]
    td_length = config["td_length"]
    device = args.device

    # ── 1. 数据加载 ──────────────────────────────────────────────────────────
    print(f"[INFO] 加载 {args.dataset} 数据集 ...")
    loader = get_data_loader(args.dataset, args.data_dir, cold_start)
    gtd, adj, scalers, cold_gtd, exec_gtd = loader.load()

    # ── 2. 初始化 MTZOOM 和训练数据 ──────────────────────────────────────────
    mtzoom = MTZOOM(n_tasks=n_tasks, lambda_t=config["lambda_t"])
    td_list = mtzoom.initialize_td(cold_gtd, window=td_length)

    # ── 3. 加载 MGSTNet ──────────────────────────────────────────────────────
    mgstnet = MGSTNet(
        m_areas=m_areas,
        n_tasks=n_tasks,
        j_history=td_length,
    ).to(device)
    mgstnet_path = os.path.join(args.checkpoint, "mgstnet.pth")
    if os.path.exists(mgstnet_path):
        mgstnet.load_state_dict(torch.load(mgstnet_path, map_location=device))
        print(f"[INFO] 已加载 MGSTNet 权重：{mgstnet_path}")
    else:
        print("[WARN] 未找到 MGSTNet 权重，使用随机初始化参数。")
    mgstnet.eval()

    # ── 4. 加载 AB-CoDC 智能体（可选）──────────────────────────────────────
    budget_agent = None
    collection_agents = None
    b_agent_path = os.path.join(args.checkpoint, "budget_agent.pth")
    c_agent_path = os.path.join(args.checkpoint, "collection_agents.pth")
    mixer_path = os.path.join(args.checkpoint, "qmix_mixer.pth")
    max_budget = max(args.budgets) if args.budgets else 9

    if all(os.path.exists(p) for p in [b_agent_path, c_agent_path, mixer_path]):
        budget_agent = BudgetAgent(
            n_tasks=n_tasks, B_total=max_budget, device=device)
        budget_agent.load_state_dict(
            torch.load(b_agent_path, map_location=device))

        collection_agents = CollectionAgents(
            n_tasks=n_tasks, m_areas=m_areas,
            state_dim=m_areas * n_tasks, device=device)
        collection_agents.eval_nets.load_state_dict(
            torch.load(c_agent_path, map_location=device))
        collection_agents.eval_mixer.load_state_dict(
            torch.load(mixer_path, map_location=device))
        print("[INFO] 已加载 AB-CoDC 智能体权重。")
    else:
        print("[WARN] 未找到 AB-CoDC 智能体权重，AB-CoDC 方法将跳过。")

    # ── 5. 各预算下的评估 ────────────────────────────────────────────────────
    methods = ["RANDOM", "GREEDY-M"]
    if budget_agent and collection_agents:
        methods = ["AB-CoDC"] + methods

    results = {m: [] for m in methods}

    for budget in args.budgets:
        print(f"\n[INFO] 预算 = {budget}")
        for method in methods:
            mape_val = evaluate_budget(
                method=method,
                budget=budget,
                exec_gtd=exec_gtd,
                td_list=td_list,
                mgstnet=mgstnet,
                adj=adj,
                mtzoom=mtzoom,
                cfg_dataset=cfg_dataset,
                n_eval_cycles=args.n_eval_cycles,
                device=device,
                budget_agent=budget_agent if method == "AB-CoDC" else None,
                collection_agents=collection_agents if method == "AB-CoDC" else None,
            )
            results[method].append(mape_val)
            print(f"  {method:15s}  MAPE = {mape_val:.2f}%")

    # ── 6. 汇总输出 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"  数据集：{args.dataset}  评估周期数：{args.n_eval_cycles}")
    print("=" * 50)
    print(f"{'Budget':>8}", end="")
    for m in methods:
        print(f"  {m:>12}", end="")
    print()
    for i, b in enumerate(args.budgets):
        print(f"{b:>8}", end="")
        for m in methods:
            print(f"  {results[m][i]:>12.2f}%", end="")
        print()

    # ── 7. 可视化 ────────────────────────────────────────────────────────────
    if args.save_fig or len(args.budgets) > 1:
        plot_mape_vs_budget(
            budgets=args.budgets,
            method_results=results,
            title=f"MAPE vs Budget ({args.dataset})",
            save_path=args.save_fig,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
