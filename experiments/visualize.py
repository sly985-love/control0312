# ==============================================================================
# 文件路径: experiments/visualize.py
# 描述: 实验结果可视化中心 (The Digital Twin Plotter)
# 功能: 
#   1. 读取各算法生成的 metrics.csv 数据。
#   2. 绘制微观控制轨迹对比图 (温度追踪、电价响应)。
#   3. 绘制宏观经济效益对比柱状图 (利润、能效、违规率)。
# 架构优势:
#   - 论文级质量: 默认 300 DPI, 支持 PDF/PNG 输出，线宽与字体大小经过优化。
#   - 自动数据对齐: 自动处理不同算法运行长度不一致的问题。
# ==============================================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置全局绘图风格 (学术论文常用风格)
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "figure.figsize": (12, 8),
    "savefig.dpi": 300
})

# 定义算法对应的颜色和标签，确保对比鲜明
COLOR_MAP = {
    'BASELINE': '#7f8c8d',  # 灰色 (基线)
    'NMPC':     '#2980b9',  # 蓝色
    'SMPC':     '#27ae60',  # 绿色
    'PURE_RL':  '#f39c12',  # 橙色
    'RL_MPC':   '#8e44ad',  # 紫色
    'RL_SMPC':  '#e74c3c'   # 红色 (核心算法：醒目)
}

class ResultVisualizer:
    def __init__(self, config_dir: str = None):
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.results_root = os.path.join(self.base_dir, "results")
        self.output_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.algos = [d for d in os.listdir(self.results_root) if os.path.isdir(os.path.join(self.results_root, d))]
        self.data = {}
        self._load_data()

    def _load_data(self):
        """加载所有算法的测评 CSV"""
        for algo in self.algos:
            csv_path = os.path.join(self.results_root, algo, "metrics.csv")
            if os.path.exists(csv_path):
                self.data[algo] = pd.read_csv(csv_path)
        if not self.data:
            print("⚠️ [Plotter] 未在 results/ 目录下发现任何 CSV 数据，请先运行 evaluate_benchmark.py")

    def plot_trajectory_comparison(self, start_step=0, end_step=288):
        """
        绘制微观轨迹对比 (例如展示 24-48 小时内的温度控制表现)
        """
        if not self.data: return
        
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(14, 10))
        
        # 图1: 温度追踪表现
        ax_temp = axes[0]
        # 绘制安全边界 (Setpoint Range)
        ax_temp.axhline(24, color='black', linestyle='--', alpha=0.5, label='Upper Bound (24°C)')
        ax_temp.axhline(18, color='black', linestyle='--', alpha=0.5, label='Lower Bound (18°C)')
        
        for algo, df in self.data.items():
            df_slice = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
            ax_temp.plot(df_slice['step'], df_slice['temp'], 
                        label=algo, color=COLOR_MAP.get(algo, '#000'), linewidth=2)
            
        ax_temp.set_ylabel("Temperature (°C)")
        ax_temp.set_title(f"Micro-Control Trajectory (Steps {start_step} to {end_step})")
        ax_temp.legend(loc='upper right', ncol=3)

        # 图2: 累计利润/成本走向
        ax_profit = axes[1]
        for algo, df in self.data.items():
            df_slice = df[(df['step'] >= start_step) & (df['step'] <= end_step)]
            # 计算切片内的累计利润
            cumulative_profit = df_slice['profit'].cumsum()
            ax_profit.plot(df_slice['step'], cumulative_profit, 
                          label=algo, color=COLOR_MAP.get(algo, '#000'), linewidth=2)
            
        ax_profit.set_ylabel("Cumulative Profit")
        ax_profit.set_xlabel("Simulation Steps (10-min interval)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "trajectory_comparison.png"))
        print("📊 轨迹对比图已保存。")

    def plot_bar_summary(self):
        """
        绘制全周期综合指标对比柱状图
        """
        if not self.data: return
        
        summary_data = []
        for algo, df in self.data.items():
            summary_data.append({
                'Algorithm': algo,
                'Total Profit': df['profit'].sum(),
                'Energy Cost': df['energy'].sum(),
                'Violation Rate (%)': df['violation'].mean() * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 创建 1x3 的柱状图矩阵
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        metrics = ['Total Profit', 'Energy Cost', 'Violation Rate (%)']
        titles = ['Economic Profit', 'Energy Consumption', 'Reliability (Constraint Violation)']
        
        for i, metric in enumerate(metrics):
            sns.barplot(x='Algorithm', y=metric, data=summary_df, 
                        palette=[COLOR_MAP.get(a, '#ccc') for a in summary_df['Algorithm']],
                        ax=axes[i], hue='Algorithm', legend=False)
            axes[i].set_title(titles[i])
            axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary_bars.png"))
        print("📊 综合性能柱状图已保存。")

# ==============================================================================
# 执行画图程序
# ==============================================================================
if __name__ == "__main__":
    print("================================================================")
    print("🎨 PFAL 结果可视化引擎启动...")
    print("================================================================")
    
    # 检查结果目录是否存在
    visualizer = ResultVisualizer()
    
    if not visualizer.data:
        print("\n❌ 错误: 未发现实验数据！")
        print("请按以下顺序操作:")
        print("1. 运行 experiments/train_rl.py 训练模型")
        print("2. 运行 experiments/evaluate_benchmark.py --algo XXX 生成数据")
    else:
        # 1. 绘制头两天的轨迹 (144步/天 * 2 = 288步)
        visualizer.plot_trajectory_comparison(start_step=0, end_step=288)
        
        # 2. 绘制全周期对比柱状图
        visualizer.plot_bar_summary()
        
        print("\n✨ 绘图任务全部完成！请查看 /plots 文件夹获取高清大图。")
    print("================================================================")