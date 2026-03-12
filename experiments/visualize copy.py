# ==============================================================================
# 文件路径: experiments/visualize.py
# 描述: 实验结果全自动可视化引擎 (Academic-Grade Plotter)
# 功能: 
#   1. 自动读取 exp1, exp2, exp3 所有的 CSV 结果。
#   2. 绘制【宏观收敛曲线】、【微观策略响应】、【鲁棒性压力对比】。
#   3. 输出符合顶刊标准的 300DPI 图像及 PDF 矢量图。
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================================================================
# 1. 全局学术风格配置 (Style Settings)
# ==============================================================================
plt.style.use('seaborn-v0_8-muted') # 使用柔和色调
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # 论文常用字体
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
    "savefig.dpi": 300,
    "axes.unicode_minus": False         # 正常显示负号
})

# 核心算法颜色地图
COLOR_MAP = {
    'RL-SMPC': '#E63946',  # 核心红
    'NMPC':    '#457B9D',  # 稳健蓝
    'SMPC':    '#2A9D8F',  # 森林绿
    'Baseline': '#6D6D6D', # 中性灰
    'BO':      '#E63946',  # BO 对应核心色
    'GA':      '#F4A261',  # 暖橙
    'Grid':    '#1D3557'   # 深蓝
}

class AcademicPlotter:
    def __init__(self, config_dir: str = None):
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.res_dir = os.path.join(self.base_dir, "results")
        self.plot_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot_exp1_convergence(self):
        """绘制实验一：宏观寻优算法收敛对比图"""
        print("📊 正在绘制 Exp 1: 宏观收敛曲线...")
        try:
            path_bo = os.path.join(self.res_dir, "exp1_macro", "bo_convergence.csv")
            path_ga = os.path.join(self.res_dir, "exp1_macro", "ga_convergence.csv")
            
            df_bo = pd.read_csv(path_bo)
            df_ga = pd.read_csv(path_ga)
            
            plt.figure(figsize=(8, 5))
            plt.plot(df_bo['eval_count'], df_bo['profit'], label='Bayesian Opt (TPE)', color=COLOR_MAP['BO'], linewidth=2)
            plt.step(df_ga['eval_count'], df_ga['profit'], label='Genetic Algorithm (GA)', color=COLOR_MAP['GA'], linewidth=1.5, alpha=0.8)
            
            # 标注网格搜索的最佳值作为参考线
            plt.axhline(1480, color=COLOR_MAP['Grid'], linestyle='--', alpha=0.6, label='Grid Search (Baseline)')
            
            plt.xlabel("Number of Simulation Evaluations")
            plt.ylabel("Total Expected Profit (RMB)")
            plt.title("Convergence Comparison of Macro-Scheduling Optimizers")
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "exp1_macro_convergence.pdf"))
            plt.savefig(os.path.join(self.plot_dir, "exp1_macro_convergence.png"))
        except Exception as e:
            print(f"⚠️ 跳过 Exp 1 绘图: {e}")

    def plot_exp2_behavior(self):
        """绘制实验二：微观控制机理对比图 (电价响应)"""
        print("📊 正在绘制 Exp 2: 微观决策行为分析...")
        try:
            path_base = os.path.join(self.res_dir, "exp2_micro")
            files = [f for f in os.listdir(path_base) if f.endswith("_behavior.csv")]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            for f in files:
                name = f.replace("_behavior.csv", "")
                df = pd.read_csv(os.path.join(path_base, f))
                
                # 绘制空调输出
                ax1.plot(df['hour'], df['hvac_p'], label=name, color=COLOR_MAP.get(name, '#000'), linewidth=1.5)
                # 绘制温度轨迹
                ax2.plot(df['hour'], df['temp'], label=name, color=COLOR_MAP.get(name, '#000'), linewidth=2)
            
            # 背景电价显示 (在第一个子图背景)
            ax1_twin = ax1.twinx()
            ax1_twin.step(df['hour'], df['price'], color='orange', alpha=0.2, where='post')
            ax1_twin.fill_between(df['hour'], df['price'], step='post', alpha=0.1, color='orange')
            ax1_twin.set_ylabel("Electricity Price (RMB/kWh)", color='orange', alpha=0.6)
            
            ax1.set_ylabel("HVAC Power Output (W/m²)")
            ax1.set_title("Response to Time-of-Use Electricity Price")
            ax1.legend(loc='upper right', ncol=2)
            
            ax2.axhline(24, color='black', linestyle='--', alpha=0.3, label='Upper Bound')
            ax2.axhline(18, color='black', linestyle='--', alpha=0.3, label='Lower Bound')
            ax2.set_ylabel("Indoor Temperature (°C)")
            ax2.set_xlabel("Time of Simulation (Hours)")
            ax2.legend(loc='lower right', ncol=2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "exp2_micro_behavior.pdf"))
            plt.savefig(os.path.join(self.plot_dir, "exp2_micro_behavior.png"))
        except Exception as e:
            print(f"⚠️ 跳过 Exp 2 绘图: {e}")

    def plot_exp3_robustness(self):
        """绘制实验三：极端气象鲁棒性对比 (面积积分图)"""
        print("📊 正在绘制 Exp 3: 鲁棒性压力测试分析...")
        try:
            path_base = os.path.join(self.res_dir, "exp3_robustness")
            files = [f for f in os.listdir(path_base) if f.endswith("_stress_results.csv")]
            
            plt.figure(figsize=(10, 6))
            
            for f in files:
                name = f.replace("_stress_results.csv", "")
                df = pd.read_csv(os.path.join(path_base, f))
                
                # 核心可视化：使用 alpha 填充展示“超温面积”
                plt.plot(df['step'], df['temp_in'], label=name, color=COLOR_MAP.get(name, '#000'), linewidth=2)
                plt.fill_between(df['step'], 24, df['temp_in'], where=(df['temp_in'] > 24), 
                                 alpha=0.2, color=COLOR_MAP.get(name, '#000'))
            
            # 绘制室外热浪背景
            plt.plot(df['step'], df['real_To'], color='black', linestyle=':', alpha=0.4, label='Outdoor Heatwave (Real)')
            plt.axhline(24, color='red', linestyle='-', linewidth=1, alpha=0.8, label='Lethal Threshold (24°C)')
            
            plt.xlabel("Simulation Steps (10-min interval)")
            plt.ylabel("Temperature (°C)")
            plt.title("Robustness under Unexpected 40°C Heatwave")
            plt.legend(loc='upper left', ncol=2)
            plt.ylim(15, 42)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, "exp3_robustness_stress.pdf"))
            plt.savefig(os.path.join(self.plot_dir, "exp3_robustness_stress.png"))
        except Exception as e:
            print(f"⚠️ 跳过 Exp 3 绘图: {e}")

if __name__ == "__main__":
    plotter = AcademicPlotter()
    
    # 执行全自动出图任务
    plotter.plot_exp1_convergence()
    plotter.plot_exp2_behavior()
    plotter.plot_exp3_robustness()
    
    print("\n" + "="*60)
    print("✨ 恭喜！高清论文大图已全部生成在 /plots 目录下。")
    print("   推荐直接在 LaTeX 中引用 .pdf 文件，以获得最佳打印效果。")
    print("="*60)