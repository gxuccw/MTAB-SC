# MTAB-SC

> **Adaptive Budgeting for Collaborative Multi-Task Data Collection in Online Sparse Crowdsensing**
>
> IEEE Transactions on Mobile Computing, Vol. 23, No. 7, pp. 7983–7998, 2024
>
> Authors: Chunyu Tu, Zhiyong Yu, Lei Han, Xianwei Guo, Fangwan Huang, Wenzhong Guo, Leye Wang
>
> DOI: [10.1109/TMC.2023.3342206](https://doi.org/10.1109/TMC.2023.3342206)

---

## 简介 / Introduction

**MTAB-SC** 是一个面向**在线稀疏群智感知（Online Sparse Crowdsensing）**的**多任务自适应预算框架**。它利用同类数据的时空相关性和异类数据的互相关性，通过层次化多智能体强化学习实现协同多任务数据采集，以最小的预算实现最优的推断精度。

MTAB-SC is a **Multi-Task Adaptive Budgeting** framework for **Online Sparse Crowdsensing**. It exploits both intra-type spatio-temporal correlations and inter-type data correlations, using hierarchical multi-agent reinforcement learning to collaboratively collect multi-task data with minimum budget and maximum inference accuracy.

### 三大核心模块 / Three Core Modules

| 模块 | 功能 |
|------|------|
| **MTZOOM** | 多任务训练数据更新——结合时空渐变性和数据集间相似性实时更新训练数据 |
| **MGSTNet** | 多图时空融合推断网络——GCN + GRU + Self-Attention 联合推断多任务数据 |
| **AB-CoDC** | 自适应预算协同数据采集——DQN（上层）+ QMIX（下层）层次化多智能体强化学习 |

---

## 项目结构 / Project Structure

```
MTAB-SC/
├── README.md                    # 项目说明（中英文）
├── requirements.txt             # 依赖包
├── config.py                    # 超参数配置
├── train.py                     # 训练主流程（Algorithm 1）
├── evaluate.py                  # 评估脚本
├── data/
│   ├── __init__.py
│   ├── data_loader.py           # 数据加载与预处理
│   └── README.md                # 数据集说明与下载指南
├── models/
│   ├── __init__.py
│   ├── mtzoom.py                # MTZOOM 训练数据更新模块
│   └── mgstnet.py               # MGSTNet 多图时空融合推断网络
├── agents/
│   ├── __init__.py
│   ├── budget_agent.py          # DQN 预算分配智能体（上层）
│   ├── collection_agent.py      # QMIX 数据采集智能体（下层）
│   ├── qmix.py                  # QMIX 混合网络
│   └── replay_buffer.py         # 经验回放池
└── utils/
    ├── __init__.py
    ├── metrics.py               # MAPE 等评价指标
    └── visualization.py         # 可视化工具
```

---

## 环境要求 / Requirements

- Python 3.8+
- PyTorch >= 1.12.0
- torch-geometric >= 2.0.0（用于 GCN，也可使用内置实现）
- numpy, pandas, scipy, matplotlib, scikit-learn, tqdm, tensorboard

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 数据准备 / Data Preparation

请参考 `data/README.md` 下载并预处理数据集，然后将文件放入 `data/` 目录：

| 数据集 | 文件 | 来源 |
|--------|------|------|
| 交通（Portland-Vancouver） | `traffic_data.csv`, `traffic_coords.csv` | https://portal.its.pdx.edu |
| 空气质量（北京） | `air_quality_data.csv`, `air_quality_coords.csv` | https://archive.ics.uci.edu/dataset/501 |

---

## 训练 / Training

```bash
# 使用交通数据集，预算=7
python train.py --dataset traffic --budget 7 --epochs 1000 --device cpu

# 使用空气质量数据集，预算=7，使用 GPU
python train.py --dataset air_quality --budget 7 --epochs 1000 --device cuda
```

常用参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dataset` | 数据集：`traffic` 或 `air_quality` | `traffic` |
| `--data_dir` | 数据目录 | `data` |
| `--budget` | 总预算 B_total | `7` |
| `--epochs` | 强化学习训练轮数 | `1000`（来自 config） |
| `--net_epochs` | MGSTNet 预训练轮数 | `300`（来自 config） |
| `--device` | 计算设备 | `cpu` |
| `--save_dir` | 模型保存目录 | `checkpoints` |

---

## 评估 / Evaluation

```bash
# 评估预算 5-9 下的性能，对比多种方法
python evaluate.py --dataset traffic --checkpoint checkpoints/ --budgets 5 6 7 8 9

# 保存结果图
python evaluate.py --dataset traffic --budgets 5 6 7 8 9 --save_fig results/mape_vs_budget.png
```

---

## 主要超参数 / Key Hyperparameters

参考论文 Table V，详见 `config.py`：

| 参数 | 值 |
|------|----|
| 学习率 | 0.001 |
| ε（探索率） | 1.0 → 0.05 |
| γ（折扣因子） | 0.99 |
| 训练数据窗口 | 4 个周期 |
| RL epoch | 1000 |
| net epoch | 300 |
| λ_t | 0.5 |

---

## 引用 / Citation

如果本代码对您的研究有帮助，请引用：

```bibtex
@article{tu2024adaptive,
  title     = {Adaptive Budgeting for Collaborative Multi-Task Data Collection
               in Online Sparse Crowdsensing},
  author    = {Tu, Chunyu and Yu, Zhiyong and Han, Lei and Guo, Xianwei and
               Huang, Fangwan and Guo, Wenzhong and Wang, Leye},
  journal   = {IEEE Transactions on Mobile Computing},
  volume    = {23},
  number    = {7},
  pages     = {7983--7998},
  year      = {2024},
  doi       = {10.1109/TMC.2023.3342206}
}
```