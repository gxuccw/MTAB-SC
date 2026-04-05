"""
数据采集智能体（下层）

对应论文 Section IV-C-2。

算法：QMIX（多智能体协同采集）
  - 每个任务 k 对应一个数据采集智能体
  - 局部观察  O^k = (p^k, c^k)
      * p^k: 已采集区域索引列表（编码为 one-hot 或定长向量）
      * c^k: 覆盖向量（长度 m 的二进制向量，1 表示已采集）
  - 动作  A_C = {1, ..., m, m+1}
      * 前 m 个动作对应采集对应区域
      * m+1 表示预算耗尽，不操作（no-op）
  - 奖励  r_C（公式 16）：每步采集后推断误差的变化量（增量式奖励）

QMIX 在 CTDE（集中训练-分散执行）范式下工作：
  - 训练时：每个智能体的局部 Q 值通过混合网络（QMIXMixer）集成为全局 Q_tot
  - 执行时：各智能体仅依据局部观察独立决策
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.qmix import QMIXMixer


# ─────────────────────────────────────────────────────────────────────────────
# 单智能体 Q 网络
# ─────────────────────────────────────────────────────────────────────────────

class CollectionQNetwork(nn.Module):
    """
    单个数据采集智能体的局部 Q 网络。

    输入：局部观察 o^k = (p^k, c^k)，编码为长度 2*m 的向量
         （p^k one-hot 编码 + c^k 二进制覆盖向量）
    输出：每个动作的 Q 值，形状 (m + 1,)
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: 形状 (batch, obs_dim)。

        Returns:
            Q 值，形状 (batch, action_dim)。
        """
        return self.net(obs)


# ─────────────────────────────────────────────────────────────────────────────
# 多智能体数据采集模块（QMIX）
# ─────────────────────────────────────────────────────────────────────────────

class CollectionAgents:
    """
    基于 QMIX 的多任务协同数据采集智能体组。

    Args:
        n_tasks:         任务数 n（智能体数）。
        m_areas:         感知区域数 m。
        state_dim:       全局状态维度（= m * n_tasks，覆盖向量拼接）。
        learning_rate:   学习率。
        gamma:           折扣因子 γ。
        epsilon_start:   初始探索率 ε。
        epsilon_end:     最终探索率 ε。
        rl_epochs:       总训练轮数（用于 ε 衰减）。
        hidden_dim:      Q 网络隐层维度。
        qmix_hidden:     QMIX 混合网络隐层维度。
        device:          计算设备。
    """

    def __init__(self,
                 n_tasks: int,
                 m_areas: int,
                 state_dim: int = None,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 rl_epochs: int = 1000,
                 hidden_dim: int = 128,
                 qmix_hidden: int = 32,
                 device: str = "cpu"):
        self.n_tasks = n_tasks
        self.m_areas = m_areas
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / rl_epochs
        self.device = device

        # 局部观察维度：c^k (m,) + p^k one-hot (m,) = 2m
        self.obs_dim = 2 * m_areas
        # 动作维度：m 个区域 + 1 个 no-op
        self.action_dim = m_areas + 1
        # 全局状态维度
        self.state_dim = state_dim if state_dim else m_areas * n_tasks

        # eval 网络：每个任务一个
        self.eval_nets = nn.ModuleList([
            CollectionQNetwork(self.obs_dim, self.action_dim, hidden_dim)
            for _ in range(n_tasks)
        ]).to(device)

        # target 网络
        self.target_nets = copy.deepcopy(self.eval_nets).to(device)
        for net in self.target_nets:
            net.eval()

        # QMIX 混合网络（eval + target）
        self.eval_mixer = QMIXMixer(self.state_dim, n_tasks, qmix_hidden).to(device)
        self.target_mixer = copy.deepcopy(self.eval_mixer).to(device)
        self.target_mixer.eval()

        # 优化器（统一管理所有网络参数）
        params = list(self.eval_nets.parameters()) + \
                 list(self.eval_mixer.parameters())
        self.optimizer = optim.Adam(params, lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    # ── 局部观察编码 ────────────────────────────────────────────────────────

    @staticmethod
    def encode_obs(collected_indices: list, coverage: np.ndarray) -> np.ndarray:
        """
        将局部观察编码为固定长度向量。

        Args:
            collected_indices: 已采集区域的索引列表 p^k。
            coverage:          覆盖向量 c^k，形状 (m,)。

        Returns:
            形状 (2*m,) 的观察向量：[p_onehot | c^k]。
        """
        m = len(coverage)
        p_onehot = np.zeros(m, dtype=np.float32)
        for idx in collected_indices:
            if 0 <= idx < m:
                p_onehot[idx] = 1.0
        return np.concatenate([p_onehot, coverage.astype(np.float32)])

    # ── 动作选择 ─────────────────────────────────────────────────────────────

    def select_actions(self,
                       obs_list: list,
                       budget_remaining: list) -> list:
        """
        ε-greedy 动作选择（各智能体独立执行）。

        Args:
            obs_list:         长度 n_tasks 的列表，每个元素为形状 (obs_dim,) 的观察。
            budget_remaining: 长度 n_tasks 的列表，各任务剩余预算。

        Returns:
            长度 n_tasks 的动作列表，每个元素 ∈ {1, ..., m+1}（1-indexed）。
        """
        actions = []
        for k in range(self.n_tasks):
            if budget_remaining[k] <= 0:
                # 预算耗尽，强制执行 no-op
                actions.append(self.m_areas + 1)
                continue

            if random.random() < self.epsilon:
                # 探索：随机选择区域（含 no-op）
                actions.append(random.randint(1, self.m_areas + 1))
            else:
                # 利用：贪心选择
                obs = torch.tensor(obs_list[k], dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_vals = self.eval_nets[k](obs).squeeze(0)  # (m+1,)
                action = q_vals.argmax().item() + 1             # 1-indexed
                actions.append(action)
        return actions

    # ── 参数更新 ─────────────────────────────────────────────────────────────

    def update(self, batch: tuple) -> float:
        """
        执行一次 QMIX 参数更新（公式 18）。

        Args:
            batch: 来自 ReplayBufferC.sample() 的元组：
                   (global_states, next_global_states,
                    obs_batch, next_obs_batch,
                    action_batch, rewards)

        Returns:
            本次更新的损失值。
        """
        (global_states, next_global_states,
         obs_batch, next_obs_batch,
         action_batch, rewards) = batch

        bs = len(rewards)  # batch size

        # 全局状态 tensor
        s = torch.tensor(np.array(global_states),
                         dtype=torch.float32, device=self.device)   # (bs, state_dim)
        ns = torch.tensor(np.array(next_global_states),
                          dtype=torch.float32, device=self.device)

        r = torch.tensor(rewards, dtype=torch.float32,
                         device=self.device).unsqueeze(1)           # (bs, 1)

        # 各智能体局部 Q 值（eval 网络）
        agent_qs = []
        agent_next_qs = []
        for k in range(self.n_tasks):
            obs_k = torch.tensor(
                np.array([o[k] for o in obs_batch]),
                dtype=torch.float32, device=self.device)        # (bs, obs_dim)
            next_obs_k = torch.tensor(
                np.array([o[k] for o in next_obs_batch]),
                dtype=torch.float32, device=self.device)

            actions_k = torch.tensor(
                [a[k] - 1 for a in action_batch],               # 转为 0-indexed
                dtype=torch.long, device=self.device)

            q_vals = self.eval_nets[k](obs_k)                   # (bs, action_dim)
            q_k = q_vals.gather(1, actions_k.unsqueeze(1))      # (bs, 1)
            agent_qs.append(q_k)

            with torch.no_grad():
                q_next_k = self.target_nets[k](next_obs_k).max(dim=1, keepdim=True)[0]
                agent_next_qs.append(q_next_k)

        # 拼接各智能体 Q 值：(bs, n_tasks)
        agent_qs_cat = torch.cat(agent_qs, dim=1)
        agent_next_qs_cat = torch.cat(agent_next_qs, dim=1)

        # 混合网络计算全局 Q_tot
        q_tot = self.eval_mixer(agent_qs_cat, s)                # (bs, 1)
        with torch.no_grad():
            q_tot_next = self.target_mixer(agent_next_qs_cat, ns)
            q_tot_target = r + self.gamma * q_tot_next          # (bs, 1)

        loss = self.loss_fn(q_tot, q_tot_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 eval 网络参数同步到 target 网络（硬更新）。"""
        for k in range(self.n_tasks):
            self.target_nets[k].load_state_dict(
                self.eval_nets[k].state_dict())
        self.target_mixer.load_state_dict(
            self.eval_mixer.state_dict())

    def decay_epsilon(self):
        """线性衰减 ε。"""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon - self.epsilon_decay)

    def get_global_state(self, coverage_list: list) -> np.ndarray:
        """
        构造全局状态向量（各任务覆盖向量拼接）。

        Args:
            coverage_list: 长度 n_tasks 的列表，每个元素为形状 (m,) 的二进制覆盖向量。

        Returns:
            形状 (m * n_tasks,) 的全局状态向量。
        """
        return np.concatenate([c.astype(np.float32) for c in coverage_list])
