# ==============================================================================
# 文件路径: experiments/exp3_robustness_test.py
# 描述: 实验三 - 极端气象鲁棒性与预报误差压力测试
# 功能: 
#   1. 模拟“错误的预报” vs “真实的极端天气”。
#   2. 对比 NMPC(确定性) 与 RL-SMPC(随机性) 在面对热浪袭击时的安全性。
#   3. 量化统计约束违规率 (Violation Rate) 与违规强度 (Integral of Violation)。
# ==============================================================================

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入环境与控制器
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from envs.pfal_env_dual import PFALEnvDual
from controllers.nmpc import NMPCController
from controllers.rl_smpc import RLSMPCController
from controllers.baseline_rule_controller import BaselineRuleController

class RobustnessExperiment:
    def __init__(self, config_dir: str = None):
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.exp_results_dir = os.path.join(self.base_dir, "results", "exp3_robustness")
        os.makedirs(self.exp_results_dir, exist_ok=True)
        
        # 物理环境
        self.env = PFALEnvDual(config_dir=self.base_dir, rl_mode=False)
        
        # 核心对比：不看预报的基线、只看一个预报的NMPC、看多个可能性的RL-SMPC
        self.controllers = {
            'NMPC': NMPCController(config_dir=self.base_dir),
            'RL-SMPC': RLSMPCController(
                vf_model_path=os.path.join(self.base_dir, "models", "value_network_weights.pth"),
                config_dir=self.base_dir,
                n_scenarios=3
            )
        }

    def run_heatwave_stress_test(self, steps=144):
        """
        模拟 24 小时内的极端热浪压力测试。
        预报说明天 28℃，但实际中午突然飙升到 40℃。
        """
        print("🔥 [Stress Test] 正在启动“极端热浪”压力测试...")
        
        results = {}

        for name, ctrl in self.controllers.items():
            print(f"🧐 正在测试算法鲁棒性: {name}...")
            obs, _ = self.env.reset(seed=100) # 统一宇宙起点
            
            history = []
            
            for k in range(steps):
                # 1. 生成“误导性”的预报 (Forecast)
                # 预报认为天气比较温和 (28℃)
                # forecast = self._generate_misleading_forecast(k, bias=0.0) current_step, Np=24, bias=0.0
                # 动态获取 Np
                current_Np = getattr(ctrl, 'Np', 24) 
                # 把 current_Np 传进天气生成函数
                forecast = self._generate_misleading_forecast(k, Np=current_Np, bias=0.0)
                
                # 2. 生成“真实的极端”天气 (Reality)
                # 在中午 (第 60-90 步) 注入 40℃ 的极端高温
                real_outdoor_temp = 28.0
                if 60 <= k <= 90:
                    real_outdoor_temp = 40.0 # 突发热浪
                
                # 强制干预环境的室外温度
                self.env.dynamics.To_val = real_outdoor_temp 

                # 提取当前环境的宏观排程参数
                macro_params = {
                    'A_h': self.env.A_h, 
                    'A_l': self.env.A_l,
                    'rho_h': self.env.rho_h, 
                    'rho_l': self.env.rho_l,
                    'xDs_l_initial': self.env.xDs_l_initial
                }

                # 根据算法类型分发正确维度的预测数据
                if "SMPC" in name:
                    # SMPC 需要完整的全景预测 (2D数组: 场景数 x 预测步数)
                    # 注意：你生成天气的变量名可能是 forecast 或者 tvp_forecast，请对应修改
                    action = ctrl.compute_action(obs, macro_params, forecast) 
                elif "MPC" in name:
                    # 确定性 NMPC 只需要提取第 0 个名义场景 (1D数组)
                    nom_forecast = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 1 else v for k, v in forecast.items()}
                    action = ctrl.compute_action(obs, macro_params, nom_forecast)
                else:
                    # Baseline 规则控制器，不需要预测
                    action = ctrl.compute_action(obs)
                
                # # 3. 控制器根据“错误预报”决策
                # if name == 'NMPC':
                #     # NMPC 只看名义场景 (Scenario 0)
                #     nom_forecast = {key: val[0] if val.ndim > 1 else val for key, val in forecast.items()}
                #     # action = ctrl.compute_action(obs, {}, nom_forecast)
                #     action = ctrl.compute_action(obs, macro_params, nom_forecast)
                # else:
                #     # RL-SMPC 决策时已考虑到 40℃ 发生的可能性 (Scenario 1)
                #     # action = ctrl.compute_action(obs, {}, forecast)
                #     action = ctrl.compute_action(obs, macro_params, nom_forecast)
                
                # 4. 环境步进
                obs, reward, _, _, info = self.env.step(action)
                
                temp_in = obs[5]
                violation = max(0, temp_in - 24.0) # 假设上限是 24℃
                
                history.append({
                    'step': k,
                    'real_To': real_outdoor_temp,
                    'temp_in': temp_in,
                    'violation': violation,
                    'action_hvac': action[4]
                })
            
            results[name] = pd.DataFrame(history)
            results[name].to_csv(os.path.join(self.exp_results_dir, f"{name}_stress_results.csv"), index=False)

        self._plot_robustness(results)

    # def _generate_misleading_forecast(self, current_step, bias=0.0):
    def _generate_misleading_forecast(self, current_step, Np=24, bias=0.0):
        """
        构造预测视界。
        SMPC 会看到三个场景：[28℃(名义), 40℃(极端热), 15℃(极端冷)]
        NMPC 只会看到 28℃。
        """
        # Np = 24
        forecast = {
            'To': np.zeros((3, Np)),
            'Co': np.full((3, Np), 0.0006),
            'Ho': np.full((3, Np), 0.008),
            'T_min': np.full((3, Np), 18.0),
            'T_max': np.full((3, Np), 24.0),
            'photo': np.full((3, Np), 1.0),
            'price': np.full((3, Np), 0.7)
        }
        forecast['To'][0, :] = 28.0 + bias # 场景0: 误导性的平庸预报
        forecast['To'][1, :] = 40.0        # 场景1: 潜在的热浪风险
        forecast['To'][2, :] = 15.0        # 场景2: 潜在的寒潮风险
        return forecast

    def _plot_robustness(self, results):
        """对比展示鲁棒性差异"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # 子图 1: 室外真实气温 vs 室内温度追踪
        ax_t = axes[0]
        ax_t.fill_between(results['NMPC']['step'], 18, 24, color='green', alpha=0.1, label='Safe Zone')
        
        # 绘制真实室外温度 (背景)
        ax_t.plot(results['NMPC']['step'], results['NMPC']['real_To'], color='black', linestyle=':', alpha=0.4, label='Real Outdoor Temp (Heatwave)')
        
        # 绘制不同算法的室内温度
        for name, df in results.items():
            ax_t.plot(df['step'], df['temp_in'], label=f"Indoor Temp ({name})", linewidth=2)
            
        ax_t.set_ylabel("Temperature (°C)")
        ax_t.set_title("Robustness Test: Response to Unexpected 40°C Heatwave")
        ax_t.legend()

        # 子图 2: 违规强度对比
        ax_v = axes[1]
        for name, df in results.items():
            ax_v.fill_between(df['step'], df['violation'], alpha=0.3, label=f"{name} Violation Area")
            ax_v.plot(df['step'], df['violation'], linewidth=1.5)
            
        ax_v.set_ylabel("Violation Magnitude (°C)")
        ax_v.set_xlabel("Steps (10-min interval)")
        ax_v.set_title("Constraint Violation Intensity (The 'Red Zone')")
        ax_v.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.exp_results_dir, "robustness_comparison.png"), dpi=300)
        
        # 打印量化结果
        print("\n" + "="*40)
        print("📊 鲁棒性量化分析结果:")
        for name, df in results.items():
            total_v = df['violation'].sum()
            max_v = df['violation'].max()
            print(f"  [{name}] 累计超温强度: {total_v:.2f} ℃·step | 最大超温: {max_v:.2f} ℃")
        print("="*40)

if __name__ == "__main__":
    exp = RobustnessExperiment()
    exp.run_heatwave_stress_test()