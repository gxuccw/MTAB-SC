"""
MGSTNet — 多图时空融合推断网络

对应论文 Section IV-B，使用 PyTorch 实现。

网络由三个串联模块组成：

  1. Spatial Block（空间模块，GCN）— 论文公式(6)
     - 每个任务 k 有独立的两层 GCN
     - 输入：SGTD^k_{m×j} 和归一化邻接矩阵 Â
     - Â = D^{-1/2} (A+I) D^{-1/2}
     - f(X, A) = σ(Â σ(Â X W_0) W_1)

  2. Temporal Block（时间模块，GRU）— 论文公式(7-11)
     - 将所有任务的 GCN 特征转置并在特征维度拼接
     - 输入 GRU 学习时间依赖

  3. Fusion Block（融合模块，Self-Attention）— 论文公式(12-13)
     - Q = W_q H, K = W_k H, V = W_v H
     - Attention(Q,K,V) = Softmax(QKᵀ / √d_k) V
     - 最终通过全连接层输出推断结果

输入维度：每个任务的 SGTD^k 形状 (m, j)（m 个区域，j 个历史周期）
输出维度：(m, n_tasks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# GCN 相关
# ─────────────────────────────────────────────────────────────────────────────

def normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    计算归一化邻接矩阵 Â = D^{-1/2} (A + I) D^{-1/2}。

    Args:
        adj: 形状 (m, m) 的邻接矩阵（对称，不含自环亦可）。

    Returns:
        形状 (m, m) 的归一化邻接矩阵。
    """
    # 加自环
    adj_hat = adj + torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    degree = adj_hat.sum(dim=1)  # 度向量 (m,)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(d_inv_sqrt)   # (m, m)
    return D_inv_sqrt @ adj_hat @ D_inv_sqrt  # Â


class GCNLayer(nn.Module):
    """单层图卷积网络（GCN Layer）。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        节点特征矩阵，形状 (batch, m, in_features)。
            adj_norm: 归一化邻接矩阵，形状 (m, m)。

        Returns:
            形状 (batch, m, out_features)。
        """
        support = self.weight(x)                          # (batch, m, out)
        out = torch.matmul(adj_norm.unsqueeze(0), support)  # (batch, m, out)
        return F.relu(out)


class TaskGCN(nn.Module):
    """
    单任务两层 GCN，对应论文公式(6)。

    输入：(batch, m, j)
    输出：(batch, m, gcn_hidden)
    """

    def __init__(self, in_features: int, hidden: int, out_features: int):
        super().__init__()
        self.gcn1 = GCNLayer(in_features, hidden)
        self.gcn2 = GCNLayer(hidden, out_features)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adj_norm)
        h = self.gcn2(h, adj_norm)
        return h  # (batch, m, out_features)


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Block（GRU）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalGRU(nn.Module):
    """
    时间模块：将各任务 GCN 特征拼接后输入 GRU。

    各任务 GCN 输出形状：(batch, m, gcn_out)
    拼接后按时间步展开：(batch, m, n_tasks * gcn_out) → 按 m 个节点处理
    GRU 输入：(seq_len=m, batch, input_size=n_tasks * gcn_out)
    GRU 输出：(seq_len=m, batch, gru_hidden)
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size,
                          num_layers=num_layers, batch_first=False)
        self.hidden_size = hidden_size

    def forward(self, gcn_outputs: list) -> torch.Tensor:
        """
        Args:
            gcn_outputs: 长度为 n_tasks 的列表，每个元素形状 (batch, m, gcn_out)。

        Returns:
            形状 (batch, m, gru_hidden) 的 GRU 输出。
        """
        # 拼接各任务特征：(batch, m, n_tasks * gcn_out)
        concat = torch.cat(gcn_outputs, dim=-1)
        batch, m, feat = concat.shape
        # 转为 GRU 所需格式：(m, batch, feat)
        gru_in = concat.permute(1, 0, 2)  # (m, batch, feat)
        gru_out, _ = self.gru(gru_in)      # (m, batch, hidden)
        # 转回 (batch, m, hidden)
        return gru_out.permute(1, 0, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Fusion Block（Self-Attention）
# ─────────────────────────────────────────────────────────────────────────────

class SelfAttentionFusion(nn.Module):
    """
    融合模块：基于自注意力机制融合多任务特征，对应论文公式(12-13)。

    Q = W_q H, K = W_k H, V = W_v H
    Attention(Q,K,V) = Softmax(QKᵀ / √d_k) V
    """

    def __init__(self, d_model: int, n_tasks: int):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # 最终输出：每个区域预测所有任务的值
        self.fc_out = nn.Linear(d_model, n_tasks)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: GRU 输出，形状 (batch, m, d_model)。

        Returns:
            形状 (batch, m, n_tasks) 的推断结果。
        """
        Q = self.W_q(h)  # (batch, m, d_model)
        K = self.W_k(h)
        V = self.W_v(h)

        scale = self.d_model ** 0.5
        attn_weights = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1)  # (batch, m, m)
        attn_out = torch.bmm(attn_weights, V)                  # (batch, m, d_model)
        out = self.fc_out(attn_out)                            # (batch, m, n_tasks)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# MGSTNet 完整网络
# ─────────────────────────────────────────────────────────────────────────────

class MGSTNet(nn.Module):
    """
    Multi-Graph Spatio-Temporal Inference Network（多图时空融合推断网络）。

    Args:
        m_areas:    感知区域数 m。
        n_tasks:    任务数 n。
        j_history:  输入历史周期数 j（训练数据窗口长度）。
        gcn_hidden: GCN 隐层维度。
        gcn_out:    GCN 输出维度（也是 GRU 每任务输入维度）。
        gru_hidden: GRU 隐层维度（= Self-Attention 的 d_model）。
    """

    def __init__(self,
                 m_areas: int,
                 n_tasks: int,
                 j_history: int,
                 gcn_hidden: int = 64,
                 gcn_out: int = 32,
                 gru_hidden: int = 64):
        super().__init__()
        self.n_tasks = n_tasks
        self.m_areas = m_areas

        # 每个任务独立的两层 GCN
        self.task_gcns = nn.ModuleList([
            TaskGCN(j_history, gcn_hidden, gcn_out)
            for _ in range(n_tasks)
        ])

        # 时间模块（GRU）
        self.temporal = TemporalGRU(
            input_size=n_tasks * gcn_out,
            hidden_size=gru_hidden
        )

        # 融合模块（Self-Attention + FC）
        self.fusion = SelfAttentionFusion(gru_hidden, n_tasks)

    def forward(self,
                sgtd_list: list,
                adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            sgtd_list: 长度为 n_tasks 的列表，每个元素为稀疏采集数据，
                       形状 (batch, m, j)（m 个区域，j 个历史周期）。
            adj:       邻接矩阵，形状 (m, m)。

        Returns:
            推断结果，形状 (batch, m, n_tasks)。
        """
        # 计算归一化邻接矩阵（每次 forward 保证与 adj 同设备）
        adj_norm = normalize_adj(adj)  # (m, m)

        # Spatial Block
        gcn_outs = []
        for k, gcn in enumerate(self.task_gcns):
            gcn_outs.append(gcn(sgtd_list[k], adj_norm))  # (batch, m, gcn_out)

        # Temporal Block
        h = self.temporal(gcn_outs)  # (batch, m, gru_hidden)

        # Fusion Block
        out = self.fusion(h)         # (batch, m, n_tasks)
        return out

    @torch.no_grad()
    def infer(self,
              td_list: list,
              adj: np.ndarray,
              device: str = "cpu") -> np.ndarray:
        """
        推断接口（不计算梯度），输入 numpy 数组，输出 numpy 数组。

        Args:
            td_list: 长度为 n_tasks 的列表，每个元素形状 (m, j)。
            adj:     邻接矩阵 numpy 数组，形状 (m, m)。
            device:  计算设备。

        Returns:
            推断结果 numpy 数组，形状 (m, n_tasks)。
        """
        self.eval()
        adj_tensor = torch.tensor(adj, dtype=torch.float32, device=device)
        sgtd_tensors = [
            torch.tensor(td, dtype=torch.float32, device=device).unsqueeze(0)
            for td in td_list
        ]  # 每个元素形状 (1, m, j)
        out = self.forward(sgtd_tensors, adj_tensor)  # (1, m, n_tasks)
        out = torch.clamp(out, 0.0, 1.0)              # 限制输出在归一化范围 [0, 1] 内
        return out.squeeze(0).cpu().numpy()           # (m, n_tasks)
