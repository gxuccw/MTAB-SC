"""
QMIX 混合网络

对应论文 Section IV-C-2 中的 QMIX 混合网络（论文公式 18）。

QMIX 将各智能体的局部 Q 值（Q^k）通过超网络（Hypernetwork）混合为全局 Q_tot。
关键约束：单调性（保证局部最优 = 全局最优，CTDE 框架下的充分条件）：
  ∂Q_tot / ∂Q^k ≥ 0，对所有 k 成立

实现参考：QMIX: Monotonic Value Function Factorisation for CTDE（Rashid et al., 2018）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 超网络（Hypernetwork）
# ─────────────────────────────────────────────────────────────────────────────

class HyperNetwork(nn.Module):
    """
    超网络：以全局状态为输入，生成混合网络的权重矩阵。

    为保证单调性约束，生成的权重通过 abs() 保证非负。
    """

    def __init__(self, state_dim: int, n_agents: int,
                 qmix_hidden: int = 32):
        """
        Args:
            state_dim:    全局状态维度。
            n_agents:     智能体数量（= n_tasks）。
            qmix_hidden:  QMIX 混合网络隐层维度。
        """
        super().__init__()
        self.n_agents = n_agents
        self.qmix_hidden = qmix_hidden

        # 生成第一层权重 W1：形状 (n_agents, qmix_hidden)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden),
            nn.ReLU(),
            nn.Linear(qmix_hidden, n_agents * qmix_hidden),
        )
        # 生成第一层偏置 b1：形状 (qmix_hidden,)
        self.hyper_b1 = nn.Linear(state_dim, qmix_hidden)

        # 生成第二层权重 W2：形状 (qmix_hidden, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden),
            nn.ReLU(),
            nn.Linear(qmix_hidden, qmix_hidden),
        )
        # 生成第二层偏置 b2：标量
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, qmix_hidden),
            nn.ReLU(),
            nn.Linear(qmix_hidden, 1),
        )

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            agent_qs: 各智能体的局部 Q 值，形状 (batch, n_agents)。
            state:    全局状态，形状 (batch, state_dim)。

        Returns:
            Q_tot：全局 Q 值，形状 (batch, 1)。
        """
        batch = state.size(0)

        # ── 第一层 ─────────────────────────────────────────────────────────
        # W1: (batch, n_agents, qmix_hidden)，非负化保证单调性
        w1 = torch.abs(self.hyper_w1(state))
        w1 = w1.view(batch, self.n_agents, self.qmix_hidden)
        b1 = self.hyper_b1(state).view(batch, 1, self.qmix_hidden)

        # agent_qs: (batch, 1, n_agents)
        qs = agent_qs.unsqueeze(1)                   # (batch, 1, n_agents)
        hidden = F.elu(torch.bmm(qs, w1) + b1)      # (batch, 1, qmix_hidden)

        # ── 第二层 ─────────────────────────────────────────────────────────
        # W2: (batch, qmix_hidden, 1)，非负化
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch, self.qmix_hidden, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)

        q_tot = torch.bmm(hidden, w2) + b2           # (batch, 1, 1)
        return q_tot.view(batch, 1)                  # (batch, 1)


# ─────────────────────────────────────────────────────────────────────────────
# QMIX 混合网络主类
# ─────────────────────────────────────────────────────────────────────────────

class QMIXMixer(nn.Module):
    """
    QMIX 混合网络，包装 HyperNetwork，对外提供统一接口。

    Args:
        state_dim:   全局状态维度。
        n_agents:    智能体数（= n_tasks）。
        qmix_hidden: 混合网络隐层维度。
    """

    def __init__(self, state_dim: int, n_agents: int,
                 qmix_hidden: int = 32):
        super().__init__()
        self.hyper = HyperNetwork(state_dim, n_agents, qmix_hidden)

    def forward(self, agent_qs: torch.Tensor,
                state: torch.Tensor) -> torch.Tensor:
        """
        计算全局 Q_tot。

        Args:
            agent_qs: (batch, n_agents)。
            state:    (batch, state_dim)。

        Returns:
            Q_tot:    (batch, 1)。
        """
        return self.hyper(agent_qs, state)
