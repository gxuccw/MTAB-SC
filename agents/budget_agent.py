"""
DQN 预算分配智能体（上层）

对应论文 Section IV-C-1。

算法：DQN（Deep Q-Network）
  - 状态  S_B : (budget_1, ..., budget_n)，表示各任务已分配的预算，sum = B_total
  - 动作  A_B : {1, 2, ..., B_total}，顺序为每个任务分配一个预算值
  - 奖励  r_B :
      * 当 k < n（中间任务）时：r_B = 0
      * 当 k = n（最后一个任务）且所有预算耗尽时：
          r_B = C * (-InferError(MTZOOM(SGTD_train)[:,-1], MGSTNet(SGTD_train)))
        否则 r_B = 0
  - 损失函数（公式 17）：
      L(θ_B) = (r_B^k + γ * max tAgent_B(s_B^{k+1}, budget^k) - eAgent_B(s_B^k, budget^k))^2

使用 eval 网络（evalAgent_B）和 target 网络（targetAgent_B）。
按频率 P_B 将 eval 网络参数同步到 target 网络。
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────────────────────────────────────────
# Q 网络结构
# ─────────────────────────────────────────────────────────────────────────────

class BudgetQNetwork(nn.Module):
    """
    预算分配 DQN 的 Q 网络。

    输入：状态向量（各任务已分配预算的拼接），形状 (batch, n_tasks)
    输出：Q 值向量，形状 (batch, action_dim = B_total)
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: 形状 (batch, state_dim)。

        Returns:
            形状 (batch, action_dim) 的 Q 值。
        """
        return self.net(state)


# ─────────────────────────────────────────────────────────────────────────────
# DQN 预算分配智能体
# ─────────────────────────────────────────────────────────────────────────────

class BudgetAgent:
    """
    DQN 预算分配智能体（上层）。

    Args:
        n_tasks:         任务数 n。
        B_total:         总预算。
        learning_rate:   学习率。
        gamma:           折扣因子 γ。
        epsilon_start:   初始探索率 ε。
        epsilon_end:     最终探索率 ε。
        rl_epochs:       总训练轮数（用于 ε 衰减计算）。
        hidden_dim:      Q 网络隐层维度。
        device:          计算设备。
    """

    def __init__(self,
                 n_tasks: int,
                 B_total: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 rl_epochs: int = 1000,
                 hidden_dim: int = 128,
                 device: str = "cpu"):
        self.n_tasks = n_tasks
        self.B_total = B_total
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        # ε 线性衰减步长
        self.epsilon_decay = (epsilon_start - epsilon_end) / rl_epochs
        self.device = device

        # 状态维度 = n_tasks（各任务已分配预算）
        state_dim = n_tasks
        action_dim = B_total

        # eval 网络 & target 网络
        self.eval_net = BudgetQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_net = copy.deepcopy(self.eval_net).to(device)
        self.target_net.eval()

        self.optimizer = optim.Adam(self.eval_net.parameters(),
                                    lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray,
                      remaining: int) -> int:
        """
        使用 ε-greedy 策略为当前任务选择预算。

        Args:
            state:     当前状态向量，形状 (n_tasks,)，各任务已分配预算。
            remaining: 剩余可用预算（当前任务至多分配这么多）。

        Returns:
            选择的预算值（1-indexed，范围 [1, remaining]）。
        """
        if remaining <= 0:
            # 无预算可分配，直接返回 0
            return 0

        if random.random() < self.epsilon:
            # 探索：在剩余预算范围内随机选择
            return random.randint(1, remaining)

        # 利用：贪心选择 Q 值最大的动作
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.eval_net(state_tensor).squeeze(0)  # (B_total,)
        # 只考虑合法动作（预算 ≥ 1，≤ remaining）
        mask = torch.zeros(self.B_total, device=self.device)
        mask[:remaining] = 1.0
        q_values = q_values * mask + (1 - mask) * (-1e9)
        action = q_values.argmax().item() + 1  # 转为 1-indexed
        return min(action, remaining)

    def update(self, batch: tuple) -> float:
        """
        执行一次 DQN 参数更新（公式 17）。

        Args:
            batch: (states, budgets, next_states, rewards) 各自的列表，
                   来自 ReplayBufferB.sample()。

        Returns:
            本次更新的损失值。
        """
        states, budgets, next_states, rewards = batch

        # 转为 tensor
        s = torch.tensor(np.array(states), dtype=torch.float32,
                         device=self.device)             # (batch, n_tasks)
        a = torch.tensor(budgets, dtype=torch.long,
                         device=self.device) - 1         # 转为 0-indexed
        # 安全 clamp，防止经验池中的异常数据（如 budget=0）导致索引越界
        a = torch.clamp(a, 0, self.B_total - 1)
        ns = torch.tensor(np.array(next_states), dtype=torch.float32,
                          device=self.device)
        r = torch.tensor(rewards, dtype=torch.float32,
                         device=self.device)

        # eval 网络计算当前 Q 值
        q_eval = self.eval_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # target 网络计算下一状态 max Q 值
        with torch.no_grad():
            q_next = self.target_net(ns).max(dim=1)[0]

        q_target = r + self.gamma * q_next

        loss = self.loss_fn(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """将 eval 网络参数复制到 target 网络（硬更新）。"""
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def decay_epsilon(self):
        """每个 epoch 后线性衰减 ε。"""
        self.epsilon = max(self.epsilon_end,
                           self.epsilon - self.epsilon_decay)

    def state_dict(self):
        """返回 eval 网络参数字典（用于保存模型）。"""
        return self.eval_net.state_dict()

    def load_state_dict(self, state_dict: dict):
        """加载 eval 网络参数。"""
        self.eval_net.load_state_dict(state_dict)
        self.update_target()
