"""
MTAB-SC 训练主流程

完整实现论文 Algorithm 1。

算法流程：
  输入: MGSTNet, TD（训练数据）
  输出: evalAgent_B, evalAgent_C, evalMixing

  1. 初始化参数更新频率 P_B 和 P_C
  2. 初始化经验池 D_B (大小 M_B) 和 D_C (大小 M_C)
  3. 初始化 evalAgent_B, evalAgent_C, evalMixing
  4. 复制到 targetAgent_B, targetAgent_C, targetMixing

  主循环 (rl_epochs 次):
    // 预算分配 (Lines 5-12)
    for k = 1 to n:
      获取 s_B^k，用 ε-greedy 分配 budget^k
      存入 D_B（中间步骤 r_B=0，最后一步计算 r_B）

    // 获取 GTD_train (Line 13)
    GTD_train = MGSTNet(SGTD[:, -j+1:], 0_vec)

    // 协同数据采集 (Lines 14-32)
    for t = 1 to max(budgets):
      获取全局状态 s_C
      for k = 1 to n: 用 ε-greedy 选择 a_C^k（若 t > budget^k 则 no-op）
      更新 SGTD_train
      计算 r_C (公式 16)
      存入 D_C
      按频率 P_C 更新采集智能体

    // 更新预算分配智能体 (Lines 33-39)
    按频率 P_B 更新 evalAgent_B

用法示例：
  python train.py --dataset traffic --budget 7 --epochs 1000 --device cpu
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import config
from data.data_loader import get_data_loader
from models.mtzoom import MTZOOM
from models.mgstnet import MGSTNet
from agents.budget_agent import BudgetAgent
from agents.collection_agent import CollectionAgents
from agents.replay_buffer import ReplayBufferB, ReplayBufferC
from utils.metrics import overall_mape


# ─────────────────────────────────────────────────────────────────────────────
# 命令行参数
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="MTAB-SC 训练脚本")
    parser.add_argument("--dataset", type=str, default="traffic",
                        choices=["traffic", "air_quality"],
                        help="使用的数据集")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="数据目录路径")
    parser.add_argument("--budget", type=int, default=7,
                        help="总预算 B_total")
    parser.add_argument("--epochs", type=int, default=None,
                        help="强化学习训练轮数（覆盖 config）")
    parser.add_argument("--net_epochs", type=int, default=None,
                        help="MGSTNet 预训练轮数（覆盖 config）")
    parser.add_argument("--device", type=str, default="cpu",
                        help="计算设备，如 cpu 或 cuda")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                        help="模型保存目录")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MGSTNet 预训练
# ─────────────────────────────────────────────────────────────────────────────

def pretrain_mgstnet(mgstnet: MGSTNet,
                     td_list: list,
                     gtd_labels: np.ndarray,
                     adj: np.ndarray,
                     n_epochs: int,
                     batch_size: int,
                     device: str) -> MGSTNet:
    """
    用冷启动阶段数据对 MGSTNet 进行预训练。

    Args:
        mgstnet:     MGSTNet 网络实例。
        td_list:     训练数据列表，每个元素形状 (m, j)。
        gtd_labels:  目标标签，形状 (m, n_tasks)，为最后一个时间步的真实值。
        adj:         邻接矩阵，形状 (m, m)。
        n_epochs:    预训练轮数。
        batch_size:  mini-batch 大小。
        device:      计算设备。

    Returns:
        预训练后的 MGSTNet。
    """
    mgstnet.train()
    optimizer = optim.Adam(mgstnet.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)

    m = td_list[0].shape[0]
    n_tasks = len(td_list)

    # 构造固定的 batch（冷启动数据量小，直接用全部）
    sgtd_tensors = [
        torch.tensor(td, dtype=torch.float32, device=device).unsqueeze(0)
        for td in td_list
    ]  # 每个元素 (1, m, j)
    labels = torch.tensor(gtd_labels, dtype=torch.float32,
                          device=device).unsqueeze(0)  # (1, m, n_tasks)

    for epoch in range(n_epochs):
        mgstnet.train()
        pred = mgstnet(sgtd_tensors, adj_tensor)  # (1, m, n_tasks)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"  [MGSTNet pretrain] Epoch {epoch+1}/{n_epochs}  "
                  f"Loss: {loss.item():.6f}")

    return mgstnet


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：根据采集动作更新稀疏采集数据
# ─────────────────────────────────────────────────────────────────────────────

def apply_actions(actions: list,
                  sgtd_train_list: list,
                  gtd_current: np.ndarray,
                  coverage_list: list,
                  budget_remaining: list,
                  m_areas: int) -> tuple:
    """
    根据各智能体动作更新稀疏采集数据矩阵和覆盖向量。

    Args:
        actions:          各智能体动作列表（1-indexed）。
        sgtd_train_list:  当前稀疏采集数据，长度 n_tasks，每个元素形状 (m,)。
        gtd_current:      当前周期真实数据，形状 (m, n_tasks)。
        coverage_list:    各任务覆盖向量列表，长度 n_tasks，每个元素形状 (m,)。
        budget_remaining: 各任务剩余预算列表。
        m_areas:          感知区域数 m。

    Returns:
        (updated_sgtd_list, updated_coverage_list, updated_budget_remaining)
    """
    n_tasks = len(actions)
    for k in range(n_tasks):
        action = actions[k]
        if action <= m_areas and budget_remaining[k] > 0:
            area_idx = action - 1  # 转为 0-indexed
            # 采集该区域真实数据
            sgtd_train_list[k][area_idx] = gtd_current[area_idx, k]
            coverage_list[k][area_idx] = 1.0
            budget_remaining[k] -= 1
    return sgtd_train_list, coverage_list, budget_remaining


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    cfg_dataset = config[args.dataset]
    m_areas = cfg_dataset["m_areas"]
    n_tasks = cfg_dataset["n_tasks"]
    cold_start = cfg_dataset["cold_start_cycles"]
    B_total = args.budget
    rl_epochs = args.epochs or config["rl_epochs"]
    net_epochs = args.net_epochs or config["net_epochs"]
    td_length = config["td_length"]
    device = args.device

    os.makedirs(args.save_dir, exist_ok=True)

    # ── 1. 数据加载 ──────────────────────────────────────────────────────────
    print(f"[INFO] 加载 {args.dataset} 数据集 ...")
    loader = get_data_loader(args.dataset, args.data_dir, cold_start)
    gtd, adj, scalers, cold_gtd, exec_gtd = loader.load()
    # gtd:      (m, t, n)
    # cold_gtd: (m, cold_start, n)
    # exec_gtd: (m, t-cold_start, n)

    # ── 2. 初始化 MTZOOM 和训练数据 ──────────────────────────────────────────
    mtzoom = MTZOOM(n_tasks=n_tasks, lambda_t=config["lambda_t"])
    td_list = mtzoom.initialize_td(cold_gtd, window=td_length)
    # td_list: 长度 n，每个元素 (m, td_length)

    # ── 3. 初始化 MGSTNet 并预训练 ───────────────────────────────────────────
    mgstnet = MGSTNet(
        m_areas=m_areas,
        n_tasks=n_tasks,
        j_history=td_length,
        gcn_hidden=64,
        gcn_out=32,
        gru_hidden=64,
    ).to(device)

    # 预训练标签：冷启动最后一个时间步的真实数据 (m, n_tasks)
    pretrain_labels = cold_gtd[:, -1, :]
    print(f"[INFO] MGSTNet 预训练 {net_epochs} 轮 ...")
    pretrain_mgstnet(mgstnet, td_list, pretrain_labels, adj,
                     net_epochs, config["net_batch_size"], device)

    # ── 4. 初始化强化学习智能体 ──────────────────────────────────────────────
    state_dim_c = m_areas * n_tasks
    budget_agent = BudgetAgent(
        n_tasks=n_tasks,
        B_total=B_total,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        rl_epochs=rl_epochs,
        device=device,
    )
    collection_agents = CollectionAgents(
        n_tasks=n_tasks,
        m_areas=m_areas,
        state_dim=state_dim_c,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        epsilon_start=config["epsilon_start"],
        epsilon_end=config["epsilon_end"],
        rl_epochs=rl_epochs,
        device=device,
    )

    # ── 5. 初始化经验回放池 ──────────────────────────────────────────────────
    buffer_B = ReplayBufferB(capacity=config["replay_buffer_size_B"])
    buffer_C = ReplayBufferC(capacity=config["replay_buffer_size_C"])

    P_B = config["P_B"]
    P_C = config["P_C"]
    step_b = 0
    step_c = 0

    # ── 6. 主训练循环（Algorithm 1）─────────────────────────────────────────
    print(f"[INFO] 开始强化学习训练，总轮数 {rl_epochs} ...")
    adj_np = adj  # numpy 邻接矩阵

    for epoch in tqdm(range(rl_epochs), desc="RL Training"):
        # 随机选取一个执行周期作为当前感知周期
        t_idx = np.random.randint(0, exec_gtd.shape[1])
        gtd_current = exec_gtd[:, t_idx, :]  # (m, n_tasks)

        # ── Lines 5-12：预算分配 ─────────────────────────────────────────
        budget_state = np.zeros(n_tasks, dtype=np.float32)  # 初始状态：各任务预算为 0
        allocated_budgets = []
        remaining = B_total

        for k in range(n_tasks):
            s_B = budget_state.copy()
            budget_k = budget_agent.select_action(s_B, remaining)
            budget_k = min(budget_k, remaining)
            allocated_budgets.append(budget_k)
            remaining = max(0, remaining - budget_k)

            # 更新状态
            budget_state[k] = budget_k
            s_B_next = budget_state.copy()

            # 中间步骤奖励为 0；最后一步稍后计算
            r_B_intermediate = 0.0
            buffer_B.push(s_B, budget_k, s_B_next, r_B_intermediate)

        # ── Line 13：获取 GTD_train（用 MGSTNet 补全稀疏数据）──────────────
        # 初始化稀疏采集数据：全 0
        sgtd_train_list = [np.zeros(m_areas, dtype=np.float32)
                           for _ in range(n_tasks)]
        coverage_list = [np.zeros(m_areas, dtype=np.float32)
                         for _ in range(n_tasks)]
        budget_remaining = allocated_budgets.copy()

        # MGSTNet 初步推断（基于训练数据）
        gtd_inferred_before = mgstnet.infer(td_list, adj_np, device)  # (m, n_tasks)

        # ── Lines 14-32：协同数据采集 ─────────────────────────────────────
        max_budget = max(allocated_budgets) if allocated_budgets else 1
        for t in range(1, max_budget + 1):
            # 全局状态
            global_state = collection_agents.get_global_state(coverage_list)

            # 各智能体局部观察
            obs_list = [
                CollectionAgents.encode_obs(
                    [i for i, v in enumerate(coverage_list[k]) if v > 0],
                    coverage_list[k]
                )
                for k in range(n_tasks)
            ]

            # ε-greedy 动作选择
            actions = collection_agents.select_actions(obs_list, budget_remaining)

            # 执行动作，更新采集数据
            sgtd_train_list, coverage_list, budget_remaining = apply_actions(
                actions, sgtd_train_list, gtd_current,
                coverage_list, budget_remaining, m_areas
            )

            # 更新后 MGSTNet 推断
            # 将 sgtd_train 整合入 td_list 后缀构造输入
            input_td = [
                np.concatenate([
                    td_list[k][:, 1:],
                    sgtd_train_list[k].reshape(m_areas, 1)
                ], axis=1)
                for k in range(n_tasks)
            ]
            gtd_inferred_after = mgstnet.infer(input_td, adj_np, device)

            # 计算选择矩阵（至少一个任务已采集的区域）
            sel_matrix = np.stack(coverage_list, axis=1)  # (m, n_tasks)

            # 奖励 r_C：推断误差变化量（公式 16）
            error_before = overall_mape(gtd_current, gtd_inferred_before, sel_matrix)
            error_after = overall_mape(gtd_current, gtd_inferred_after, sel_matrix)
            r_C = error_before - error_after  # 误差减少则为正奖励

            # 下一状态
            next_global_state = collection_agents.get_global_state(coverage_list)
            next_obs_list = [
                CollectionAgents.encode_obs(
                    [i for i, v in enumerate(coverage_list[k]) if v > 0],
                    coverage_list[k]
                )
                for k in range(n_tasks)
            ]

            # 存入经验池
            buffer_C.push(
                global_state, next_global_state,
                obs_list, next_obs_list,
                actions, r_C
            )

            gtd_inferred_before = gtd_inferred_after

            # 按频率 P_C 更新采集智能体
            step_c += 1
            if step_c % P_C == 0 and len(buffer_C) >= config["rl_batch_size"]:
                batch_C = buffer_C.sample(config["rl_batch_size"])
                collection_agents.update(batch_C)
                collection_agents.update_target()

        # ── Lines 33-39：更新预算分配智能体 ──────────────────────────────
        # 计算最终奖励 r_B^n（公式 15）
        # 所有预算耗尽时：r_B = C * (-InferError)
        C_reward = 1.0  # 奖励缩放系数
        if sum(budget_remaining) == 0:
            final_r_B = C_reward * (-overall_mape(
                gtd_current, gtd_inferred_after, sel_matrix))
        else:
            final_r_B = 0.0

        # 更新最后一步的奖励（重新入队）
        # 注：简化处理：直接存入最终奖励样本
        # 完整实现中应更新之前存入的最后一步奖励
        if len(buffer_B) > 0:
            # 在缓冲区末尾修正最后一个 D_B 条目的奖励
            last_exp = buffer_B.buffer[-1]
            buffer_B.buffer[-1] = (last_exp[0], last_exp[1],
                                   last_exp[2], final_r_B)

        step_b += 1
        if step_b % P_B == 0 and len(buffer_B) >= config["rl_batch_size"]:
            batch_B = buffer_B.sample(config["rl_batch_size"])
            budget_agent.update(batch_B)
            budget_agent.update_target()

        # ── MTZOOM 更新训练数据 ──────────────────────────────────────────
        td_list = mtzoom.update(
            td_list,
            sgtd_train_list,
            [cov.copy() for cov in coverage_list]
        )

        # ε 衰减
        budget_agent.decay_epsilon()
        collection_agents.decay_epsilon()

        # 每 100 轮打印一次
        if (epoch + 1) % 100 == 0:
            mape_val = overall_mape(gtd_current, gtd_inferred_after, sel_matrix)
            print(f"\n[Epoch {epoch+1}/{rl_epochs}]  MAPE: {mape_val:.2f}%  "
                  f"ε_B: {budget_agent.epsilon:.3f}  "
                  f"ε_C: {collection_agents.epsilon:.3f}")

    # ── 7. 保存模型 ──────────────────────────────────────────────────────────
    torch.save(budget_agent.state_dict(),
               os.path.join(args.save_dir, "budget_agent.pth"))
    torch.save(collection_agents.eval_nets.state_dict(),
               os.path.join(args.save_dir, "collection_agents.pth"))
    torch.save(collection_agents.eval_mixer.state_dict(),
               os.path.join(args.save_dir, "qmix_mixer.pth"))
    torch.save(mgstnet.state_dict(),
               os.path.join(args.save_dir, "mgstnet.pth"))
    print(f"[INFO] 模型已保存至 {args.save_dir}/")

    return budget_agent, collection_agents, mgstnet


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    train(args)
