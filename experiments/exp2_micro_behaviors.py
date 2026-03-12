# ==============================================================================
# 文件路径: experiments/exp2_micro_behaviors.py
# 描述: 实验二 - 微观控制决策行为深度分析
# 功能: 
#   1. 模拟一个 48 小时的典型运行窗口，重点观察“电价高峰期”前后的决策。
#   2. 对比 Baseline (反应式) 与 RL-SMPC (预测式) 的空调功率输出曲线。
#   3. 量化证明预测控制如何利用“热惯性”在低价期蓄冷。
# ==============================================================================

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# 导入底层环境与各种控制器
from envs.pfal_env_dual import PFALEnvDual
from controllers.baseline_rule_controller import BaselineRuleController
from controllers.rl_smpc import RLSMPCController
from controllers.nmpc import NMPCController

class MicroBehaviorExperiment:
    def __init__(self, config_dir: str = None):
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.exp_results_dir = os.path.join(self.base_dir, "results", "exp2_micro")
        os.makedirs(self.exp_results_dir, exist_ok=True)
        
        # 实例化测试环境 (关闭 RL 模式，开启物理精确模式)
        self.env = PFALEnvDual(config_dir=self.base_dir, rl_mode=False)
        
        # 设定对比选手
        self.controllers = {
            'Baseline': BaselineRuleController(config_dir=self.base_dir),
            'NMPC': NMPCController(config_dir=self.base_dir),
            'RL-SMPC': RLSMPCController(
                vf_model_path=os.path.join(self.base_dir, "models", "value_network_weights.pth"),
                config_dir=self.base_dir
            )
        }

    def run_scenario(self, days=2):
        """运行一个高分辨率的 48 小时场景测试"""
        total_steps = days * 144  # 144步/天 (10分钟采样)
        
        # 预设电价策略 (Time-of-Use): 
        # 00:00 - 08:00 (低谷: 0.3元) 
        # 08:00 - 14:00 (高峰: 1.5元) 
        # 14:00 - 24:00 (平段: 0.7元)
        
        all_results = {}

        for name, ctrl in self.controllers.items():
            print(f"🧐 正在监测算法行为: {name}...")
            obs, _ = self.env.reset(seed=42) # 保证起点完全相同
            
            history = []
            
            for step in range(total_steps):
                # 构造环境预测 (在此实验中，我们假设预测是准的，重点看响应逻辑)
                # 在这行上方，动态获取当前正在运行的控制器的 Np (如果没有则默认为 24)
                current_Np = getattr(ctrl, 'Np', 24)
                # 将 Np 传给生成预测的函数 (注意：你需要一并修改 _generate_forecast 函数接收这个参数，见下一步)
                tvp_forecast = self._get_deterministic_forecast(step, Np=current_Np)
                # tvp_forecast = self._get_deterministic_forecast(step, Np=current_Np )
                
                # 记录当前状态
                temp_in = obs[5]
                # electricity_price = tvp_forecast['price'][0, 0] if 'SMPC' in name else tvp_forecast['price'][0]
                electricity_price = tvp_forecast['price'][0, 0]
                
                # 提取当前环境的宏观排程参数
                macro_params = {
                    'A_h': self.env.A_h, 
                    'A_l': self.env.A_l,
                    'rho_h': self.env.rho_h, 
                    'rho_l': self.env.rho_l,
                    'xDs_l_initial': self.env.xDs_l_initial
                }
                if "SMPC" in name:
                    action = ctrl.compute_action(obs, macro_params, tvp_forecast)
                elif "MPC" in name:
                    # 确定性 MPC 取名义场景
                    nom_forecast = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 1 else v for k, v in tvp_forecast.items()}
                    action = ctrl.compute_action(obs, macro_params, nom_forecast)
                else:
                    action = ctrl.compute_action(obs)

                # # 计算动作
                # if name == 'Baseline':
                #     action = ctrl.compute_action(obs)
                # elif name == 'NMPC':
                #     action = ctrl.compute_action(obs, {}, tvp_forecast)
                # else: # RL-SMPC
                #     action = ctrl.compute_action(obs, {}, tvp_forecast)
                
                # 提取空调动作 (U4) - 负值代表制冷
                hvac_power = action[4]
                
                # 执行
                obs, reward, _, _, info = self.env.step(action)
                
                history.append({
                    'step': step,
                    'hour': (step * 10) / 60,
                    'temp': temp_in,
                    'hvac_p': hvac_power,
                    'price': electricity_price,
                    'reward': reward
                })
            
            all_results[name] = pd.DataFrame(history)
            all_results[name].to_csv(os.path.join(self.exp_results_dir, f"{name}_behavior.csv"), index=False)

        self._plot_behavior(all_results)

    def _get_deterministic_forecast(self, current_step, Np=24):
        """构造带有剧烈电价波动的未来预测序列"""
        # Np = 24
        price_seq = np.zeros(Np)
        for i in range(Np):
            # 简单的时钟逻辑模拟电价跳变
            hour = ((current_step + i) * 10 / 60) % 24
            if 8 <= hour < 14: price_seq[i] = 1.5
            elif 0 <= hour < 8: price_seq[i] = 0.3
            else: price_seq[i] = 0.7
            
        # 兼容 SMPC 格式
        return {
            'To': np.full((3, Np), 28.0),
            'Co': np.full((3, Np), 0.0006),
            'Ho': np.full((3, Np), 0.008),
            'T_min': np.full((3, Np), 18.0),
            'T_max': np.full((3, Np), 24.0),
            'photo': np.full((3, Np), 1.0),
            'price': np.tile(price_seq, (3, 1))
        }

    def _plot_behavior(self, all_results):
        """绘制微观行为对比图"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        # 子图 1: 电价背景 (所有算法通用)
        ref_df = all_results['Baseline']
        axes[0].step(ref_df['hour'], ref_df['price'], color='orange', linewidth=2, label='Electricity Price (RMB/kWh)')
        axes[0].set_ylabel("Electricity Price")
        axes[0].set_title("Time-of-Use Electricity Price & Controller Response")
        axes[0].fill_between(ref_df['hour'], ref_df['price'], alpha=0.1, color='orange')
        
        # 子图 2: 空调输出对比 (U4)
        for name, df in all_results.items():
            axes[1].plot(df['hour'], df['hvac_p'], label=f"{name} HVAC Output", linewidth=1.5)
        axes[1].set_ylabel("HVAC Power (W/m2)")
        axes[1].axhline(0, color='black', alpha=0.3)
        axes[1].legend(loc='lower right')
        
        # 子图 3: 室内温度轨迹对比
        for name, df in all_results.items():
            axes[2].plot(df['hour'], df['temp'], label=f"{name} Temp", linewidth=2)
        axes[2].axhline(24, color='red', linestyle='--', alpha=0.5, label='Constraint Upper')
        axes[2].axhline(18, color='blue', linestyle='--', alpha=0.5, label='Constraint Lower')
        axes[2].set_ylabel("Indoor Temperature (°C)")
        axes[2].set_xlabel("Time (Hours)")
        axes[2].legend(loc='lower right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_results_dir, "micro_behavior_analysis.png"), dpi=300)
        print(f"📊 行为分析图已存至: {self.exp_results_dir}")

if __name__ == "__main__":
    exp = MicroBehaviorExperiment()
    exp.run_scenario(days=2)