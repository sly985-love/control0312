
---

# 🌿 PFAL-RLSMPC: 人工智能驱动的植物工厂自主生产管理系统

## 📖 项目简介

本项目是一个针对**全人工光型植物工厂 (PFAL)** 的双层嵌套优化框架。它将宏观层面的运筹学排程与微观层面的先进预测控制相结合，旨在实现产量最大化与能耗成本最小化的平衡。

### 核心亮点

* **双层优化架构**: 上层利用 **Bayesian Optimization (BO)** 锁定最优种植密度与周期；下层利用 **RL-SMPC** 实现高精度环控。
* **RL-SMPC 控制器**: 融合了随机模型预测控制 (SMPC) 的安全性与强化学习 (RL) 的长线预见性。
* **物理引擎驱动**: 基于双区耦合的非线性动力学微分方程，使用 **RK4 积分** 确保仿真精度。
* **无缝模型转译**: 能够自动将 PyTorch 训练的神经网络权重解析为 CasADi 符号函数，实现纳秒级在线优化。

---

## 📂 目录结构

```text
├── common/               # 种植面积平衡与经济学计算引擎
├── configs/              # 环境、模型及优化器的 YAML 配置文件
├── controllers/          # 五大核心控制器 (Baseline, NMPC, SMPC, RL-MPC, RL-SMPC)
├── envs/                 # 数字孪生仿真环境 (Gymnasium 封装)
├── macro_optimizers/     # 宏观战略决策器 (GA, Grid Search)
├── RL/                   # 神经网络架构与权重提取工具
├── experiments/          # 自动化训练、测评与论文出图脚本
├── models/               # 存放训练好的模型权重 (.pth / .zip)
└── run_pipeline.sh       # 一键全自动实验流水线

```

---

## 🛠️ 安装指南


1. **安装依赖**:
```bash
pip install -r requirements.txt

```


*核心依赖包括: `casadi`, `torch`, `stable-baselines3`, `optuna`, `wandb`, `matplotlib`, `pandas*`
2. **初始化 WandB**:
```bash
wandb login

```



---

## 🚀 快速开始

本项目设计为高度自动化，你可以通过以下步骤运行整个科研实验链路：

### 1. 一键运行全链路 (推荐)

直接运行我们预设的流水线脚本，它会自动完成训练、寻优、测评和出图：

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

### 2. 分步执行

如果你想单独调试某个模块：

* **训练 AI 大脑**: `python -m experiments.train_rl`
* **宏观排程寻优**: `python -m macro_optimizers.bo_scheduler`
* **运行算法对比**: `python -m experiments.evaluate_benchmark --algo RL_SMPC`
* **生成论文图表**: `python -m experiments.visualize`

---

## 🔬 核心数学原理

本项目的微观控制器目标函数 $J$ 遵循以下结构：

$$J = \min_{\mathbf{u}_0, \dots, \mathbf{u}_{N-1}} \sum_{s=1}^{N_s} P_s \left[ \sum_{k=0}^{N_p-1} \gamma^k \cdot \ell(x_{k,s}, u_k) + \gamma^{N_p} \cdot V_{\theta}(x_{N_p,s}) \right]$$

其中：

* $\ell(\cdot)$ 是包含电费、产量收益和约束惩罚的即时阶段代价。
* $V_{\theta}(\cdot)$ 是由 **PPO 算法** 训练出的深度价值网络，为控制器提供“无限未来”的估值指引。

---

## 📊 实验结果展示

运行完成后，你可以在 `/plots` 目录下找到如下分析图表：

* `exp1_macro_convergence.png`: 展示遗传算法如何快速锁定最优种植排程。
* `exp2_micro_behavior.png`: 展示控制器如何在电价高峰前进行“提前蓄冷”。
* `exp3_robustness_stress.png`: 展示系统在遭遇 40°C 突发热浪时的生存能力。

---


# 环境安装
conda create -n control_env python=3.11
conda activate control_env 
pip install -r requirements.txt
pip install -r requirements.txt --no-build-isolation -i https://pypi.tuna.tsinghua.edu.cn/simple
# 设置系统代理
git config --global --unset http.proxy 
git config --global --unset https.proxy
git config --global http.proxy http://127.0.0.1:7897 #（注意这个端口号一定是【手动设置代理开启的端口号】）
git config --global -l
git clone https://github.com/Tim-Salzmann/l4casadi.git
pip install -r requirements_build.txt
pip install . --no-build-isolation
pip install optuna -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

conda activate control_env 
wandb login
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\train_rl_wandb.py
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\macro_optimizers\ga_scheduler.py

python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo BASELINE
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo NMPC
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo SMPC
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo PURE_RL
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo RL_MPC
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\evaluate_benchmark.py  --algo RL_SMPC

python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\exp1_macro_results.py
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\exp2_micro_behaviors.py
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\exp3_robustness_test.py
python C:\Users\29341\Desktop\lettuce_control\GA_RLSMPC_PFAL\experiments\visualize.py
