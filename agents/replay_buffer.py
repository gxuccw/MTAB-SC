"""
经验回放池模块

实现两个独立的经验回放池：
  - ReplayBufferB (D_B)：预算分配智能体的经验池，容量 M_B
  - ReplayBufferC (D_C)：数据采集智能体的经验池，容量 M_C

存储内容：
  - D_B：(s_B^k, budget^k, s_B^{k+1}, r_B^k)
  - D_C：(s_C, s_C^{t+1}, {o^{k,t}}, {o^{k,t+1}}, {a_C^k}, r_C)
"""

import random
from collections import deque


class ReplayBufferB:
    """
    预算分配智能体经验池 D_B。

    每条经验为 (state, budget, next_state, reward) 的四元组：
      - state (s_B^k)      : 当前预算分配状态（已为各任务分配的预算列表）
      - budget (budget^k)  : 本步为任务 k 分配的预算
      - next_state (s_B^{k+1}): 下一时刻状态
      - reward (r_B^k)     : 奖励（中间步骤为 0，最后一步为公式 15 的值）
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: 经验池容量 M_B。
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, budget, next_state, reward):
        """向经验池中添加一条经验。"""
        self.buffer.append((state, budget, next_state, reward))

    def sample(self, batch_size: int):
        """
        从经验池中随机采样一个 mini-batch。

        Args:
            batch_size: 采样数量。

        Returns:
            states, budgets, next_states, rewards 各自的列表。
        """
        batch = random.sample(self.buffer, batch_size)
        states, budgets, next_states, rewards = zip(*batch)
        return list(states), list(budgets), list(next_states), list(rewards)

    def __len__(self):
        return len(self.buffer)

    @property
    def is_ready(self, min_size: int = None):
        """经验池中是否有足够样本。"""
        return len(self.buffer) > 0


class ReplayBufferC:
    """
    数据采集智能体经验池 D_C（QMIX）。

    每条经验包含：
      - global_state      : 全局状态 s_C，形状 (m,) 或更高维
      - next_global_state : 下一全局状态 s_C^{t+1}
      - obs_list          : 各智能体局部观察列表 {o^{k,t}}，长度 n_tasks
      - next_obs_list     : 各智能体下一局部观察 {o^{k,t+1}}，长度 n_tasks
      - action_list       : 各智能体动作 {a_C^k}，长度 n_tasks
      - reward            : 共享奖励 r_C（公式 16）
    """

    def __init__(self, capacity: int):
        """
        Args:
            capacity: 经验池容量 M_C。
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, global_state, next_global_state,
             obs_list, next_obs_list,
             action_list, reward):
        """向经验池中添加一条多智能体经验。"""
        self.buffer.append((
            global_state, next_global_state,
            obs_list, next_obs_list,
            action_list, reward
        ))

    def sample(self, batch_size: int):
        """
        随机采样一个 mini-batch。

        Returns:
            global_states, next_global_states,
            obs_batch, next_obs_batch,
            action_batch, rewards
        """
        batch = random.sample(self.buffer, batch_size)
        (global_states, next_global_states,
         obs_batch, next_obs_batch,
         action_batch, rewards) = zip(*batch)
        return (list(global_states), list(next_global_states),
                list(obs_batch), list(next_obs_batch),
                list(action_batch), list(rewards))

    def __len__(self):
        return len(self.buffer)
