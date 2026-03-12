
---

### 1. 完整的 `requirements.txt`

这个依赖清单涵盖了底层物理仿真、强化学习、非线性优化器工具链以及可视化工具。

```text
# ==========================================
# GA-RLSMPC-PFAL 项目环境依赖
# 建议使用 Python 3.9 或 3.10
# ==========================================

# 1. 核心科学计算与数据处理
numpy>=1.24.0
pandas>=2.0.0

# 2. 配置文件解析
PyYAML>=6.0

# 3. 强化学习与仿真环境底座
gymnasium>=0.29.1
torch>=2.0.0            # PyTorch (用于定义和运行 Value Network)
stable-baselines3>=2.1.0 # 强化学习算法库 (PPO/SAC)
wandb>=0.15.0           # Weights & Biases (用于实验追踪与可视化)

# 4. 核心数学与非线性优化器
casadi>=3.6.0           # 将 PyTorch 符号化并进行非线性求解的核心库 (含 IPOPT)

# 5. 可视化出图
matplotlib>=3.7.0
seaborn>=0.12.0

```

---

### 2. 完整的 `README.md` (分步执行版)

建议将以下内容完全覆盖你仓库根目录下的 `README.md`。

```markdown
# GA-RLSMPC-PFAL: 植物工厂双层架构数字孪生与控制系统 🥬🤖

本项目实现了一个针对 40 尺集装箱植物工厂（PFAL）的**工业级双层优化控制框架**。
系统旨在解决连续农业生产中存在的“长周期离散排程”与“高频连续环境扰动”之间的时间尺度错位问题。

* **宏观层 (Macro-Level)**：利用**遗传算法 (Genetic Algorithm, GA)** 解决严格遵守物理离散节拍（最大公约数 GCD 流转）的最优排程问题。
* **微观层 (Micro-Level)**：利用**强化学习增强的随机模型预测控制 (RL-SMPC)** 进行逐分钟的声光温电精确控制。利用 Ornstein-Uhlenbeck (OU) 过程模拟气象误差，并引入代数平滑 ReLU 算子与 L-BFGS 拟牛顿法确保算力绝对稳定。

---

## 🛠️ 第一步：环境配置 (Environment Setup)

推荐使用 Anaconda 创建纯净的虚拟环境。

```bash
# 1. 创建并激活 Python 3.10 虚拟环境
conda create -n pfal_rlsmpc python=3.10 -y
conda activate pfal_rlsmpc

# 2. 克隆仓库 (请替换为你的实际仓库地址)
git clone [https://github.com/your-username/GA-RLSMPC-PFAL.git](https://github.com/your-username/GA-RLSMPC-PFAL.git)
cd GA-RLSMPC-PFAL

# 3. 安装所有依赖
pip install -r requirements.txt

# 4. (可选) 登录 WandB 账号以同步强化学习训练数据
wandb login

```

## ⚙️ 第二步：参数配置 (Configuration)

在跑实验之前，你可以在 `configs/` 目录下调整系统的物理和经济常数。
最核心的文件是 **`configs/envs/PFAL_dual.yml`**，你可以修改：

* `constraints`: 设备最大功率、温湿度安全红线。
* `geometry`: 集装箱的物理尺寸与面积。
* `economics`: 峰平谷电价参数（修改这里可对比不同城市的经济效益）。

---

## 🚀 第三步：拆解运行流水线 (Step-by-Step Execution)

为了彻底理解双层架构的信息流，建议你按以下 4 个阶段**依次运行**脚本：

### 阶段 1：训练 AI 价值网络 (RL Training)

**目标**：在物理沙盘中让 PPO 智能体试错，学习出具有“无限未来视野”的价值网络（Value Function）。

```bash
python -m experiments.train_rl

```

* **输出**：系统会自动进行“模型手术”，将黑盒 `.zip` 模型中的权重剥离，保存为 `models/value_network_weights.pth`，供后续的 MPC 控制器调用。

### 阶段 2：宏观排程寻优 (Macro Scheduling via GA)

**目标**：充当“工厂 CEO”，在统一的 30 天财务核算期内，寻找利润最高的育苗天数、成株天数与种植密度。

```bash
python -m macro_optimizers.ga_scheduler

```

* **机制**：GA 会调用极速基线控制器，在物理底座中模拟评估数百种方案的真实电费与产能。内置沙盒重置机制，防止内存泄漏。
* **输出**：进化出的最强排程方案会落盘到 `configs/macro_best/ga_best_schedule.yml`。

### 阶段 3：微观控制鲁棒性测评 (Micro Benchmark Arena)

**目标**：宏观计划已定，现在考核各种环控算法（车间主任）在恶劣天气和分时电价下的控场能力。

```bash
# 评测基线规则控制器
python -m experiments.evaluate_benchmark --algo BASELINE

# 评测传统非线性 MPC
python -m experiments.evaluate_benchmark --algo NMPC

# 评测我们提出的终极强化学习增强随机 MPC
python -m experiments.evaluate_benchmark --algo RL_SMPC

```

* **机制**：引入 Ornstein-Uhlenbeck (OU) 过程，模拟带有均值回归特性的真实气象预报误差。
* **输出**：各算法的逐分钟状态数据与最终经济报表会保存在 `results/` 目录下。

### 阶段 4：结果可视化 (Visualization)

**目标**：读取阶段 3 生成的实验数据，绘制学术论文级别的对比图表。

```bash
python -m experiments.visualize

```

* **输出**：高清图表保存在 `plots/` 目录下。

---

## ⚡ 终极替代方案：一键执行 (One-Click Pipeline)

如果你已经配置好环境并确认参数无误，可以通过提供的 Shell 脚本一键走完上述所有流程（炼丹 -> 排程 -> 测评 -> 出图）：

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh

```

---

## 📂 核心代码架构导读 (Project Structure)

```text
GA-RLSMPC-PFAL/
├── common/                  # 公共组件库
│   ├── area_solver.py       # (关键) 核心物理批次(GCD)面积解算器
│   └── economics.py         # 包含峰平谷(ToU)规则的经济核算引擎
├── configs/                 # YAML 配置中心
├── controllers/             # 控制器武库
│   ├── rl_smpc.py           # (核心) 融入代数平滑 ReLU 与 L-BFGS 的 RL-SMPC
│   └── baseline_...         # 对比基线
├── envs/                    # 物理孪生环境
│   ├── pfal_env_dual.py     # 包含双时钟重置与量纲风险防御机制的 Gym 环境
│   └── pfal_dynamics...     # 底层温湿度与生物量微分方程
├── experiments/             # 实验与评测脚本
├── macro_optimizers/        # 宏观排程优化器
│   └── ga_scheduler.py      # (核心) 遗传算法调度器 (含内存泄漏防御)
├── RL/                      # 强化学习模块
│   ├── extract_sb3_vf.py    # 动态解析任意深度的 PyTorch 网络权重剥离器
│   └── rl_network.py        # 网络拓扑基类
└── run_pipeline.sh          # 一键自动化测试流水线

```

---

**维护与贡献**：本环境高度依赖 CasADi 进行跨语言 C++ 符号计算，若需修改底层状态维度 (nx) 或控制维度 (nu)，请务必在修改 `envs` 的同时同步更新 `configs` 及 `controllers` 内的张量定义。

```
