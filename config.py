"""
超参数配置文件
对应论文 Table V 及实验设置
"""

config = {
    # ── 强化学习参数 ──────────────────────────────────────────────────────────
    "learning_rate": 0.001,          # 学习率
    "epsilon_start": 1.0,            # ε-greedy 起始探索率
    "epsilon_end": 0.05,             # ε-greedy 最终探索率
    "gamma": 0.99,                   # 折扣因子 γ
    "rl_epochs": 1000,               # 强化学习训练轮数
    "rl_batch_size": 64,             # 强化学习 mini-batch 大小

    # ── 推断网络参数 ──────────────────────────────────────────────────────────
    "net_epochs": 300,               # MGSTNet 训练轮数（论文中称 net epoch）
    "net_batch_size": 32,            # MGSTNet mini-batch 大小
    "M_epochs": 100,                 # M epoch（表 V 中的 M）

    # ── MTZOOM 参数 ───────────────────────────────────────────────────────────
    "td_length": 4,                  # 训练数据滑动窗口长度（4 个周期）
    "lambda_t": 0.5,                 # 时空渐变权重 λ_t

    # ── 经验回放池 ────────────────────────────────────────────────────────────
    "replay_buffer_size_B": 100,     # 预算分配智能体经验池容量 M_B
    "replay_buffer_size_C": 100,     # 数据采集智能体经验池容量 M_C

    # ── 网络更新频率 ──────────────────────────────────────────────────────────
    "P_B": 10,                       # 预算分配 target 网络更新频率
    "P_C": 10,                       # 数据采集 target 网络更新频率

    # ── 数据集：交通（Portland-Vancouver）────────────────────────────────────
    "traffic": {
        "m_areas": 28,               # 感知区域数
        "t_cycles": 168,             # 感知周期数（1 周，每小时 1 个周期）
        "n_tasks": 3,                # 任务数：VHT, Occupancy, Volume
        "cold_start_cycles": 12,     # 冷启动周期数（约半天）
        "task_names": ["VHT", "Occupancy", "Volume"],
    },

    # ── 数据集：空气质量（北京）──────────────────────────────────────────────
    "air_quality": {
        "m_areas": 35,               # 感知区域数
        "t_cycles": 168,             # 感知周期数
        "n_tasks": 2,                # 任务数：PM2.5, PM10
        "cold_start_cycles": 12,     # 冷启动周期数
        "task_names": ["PM2.5", "PM10"],
    },
}
