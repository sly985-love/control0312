# ==============================================================================
# 文件路径: experiments/evaluate_benchmark.py
# 描述: 多算法统一测评引擎 (The Benchmark Arena)
# 功能: 
#   1. 加载指定的控制器 (Baseline, NMPC, SMPC, RL_SMPC 等)。
#   2. 在标准的物理仿真环境下跑完完整的生长周期。
#   3. 收集并保存高维实验数据 (能耗、产量、温控精度、利润)。
# 修正说明:
#   1. 【修复8: 预测过于理想化】抛弃静态偏移，引入 Ornstein-Uhlenbeck (OU) 随机过程，
#      生成具备时间相关性和均值回归特性的真实气象预报误差序列。
#   2. 对齐底层 Env 返回的 info 字典键值，确保评测指标 (电费、惩罚) 被正确记录。
# ==============================================================================

import os
import yaml
import time
import argparse
import numpy as np
import pandas as pd
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
# 导入底层环境
from envs.pfal_env_dual import PFALEnvDual
# 导入所有武库中的控制器
from controllers.baseline_rule_controller import BaselineRuleController
from controllers.nmpc import NMPCController
from controllers.smpc import SMPCController
from controllers.pure_rl_controller import PureRLController
from controllers.rl_mpc import RLMPCController
from controllers.rl_smpc import RLSMPCController

class BenchmarkEvaluator:
    """
    统一算法评测器
    """
    def __init__(self, algo_name: str, config_dir: str = None):
        self.algo_name = algo_name.upper()
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 1. 实例化物理环境 (非 RL 模式，完全基于物理绝对值交互)
        self.env = PFALEnvDual(config_dir=self.base_dir, rl_mode=False)
        # self.Np = 36  # 预测视界 (6小时，基于 dt=10min)
        
        # 2. 控制器工厂：动态实例化
        self.controller = self._get_controller()

        # 3. 动态对齐预测视界 (Np)
        # 尝试从控制器读取 Np，如果该控制器没有 Np 属性 (如 Baseline)，则给默认值 24
        self.Np = getattr(self.controller, 'Np', 24)
        
        # 3. 结果存储路径
        self.results_dir = os.path.join(self.base_dir, "results", self.algo_name)
        os.makedirs(self.results_dir, exist_ok=True)

    def _get_controller(self):
        """控制器工厂模式"""
        model_path_pth = os.path.join(self.base_dir, "models", "value_network_weights.pth")
        model_path_zip = os.path.join(self.base_dir, "models", "best_model.zip")

        if self.algo_name == "BASELINE":
            return BaselineRuleController(config_dir=self.base_dir)
        elif self.algo_name == "NMPC":
            return NMPCController(config_dir=self.base_dir)
        elif self.algo_name == "SMPC":
            return SMPCController(config_dir=self.base_dir, n_scenarios=3)
        elif self.algo_name == "PURE_RL":
            return PureRLController(model_path=model_path_zip, algo='PPO', config_dir=self.base_dir)
            # return PureRLController(model_path=model_path_zip, algo='SAC', config_dir=self.base_dir)
        elif self.algo_name == "RL_MPC":
            return RLMPCController(vf_model_path=model_path_pth, config_dir=self.base_dir)
        elif self.algo_name == "RL_SMPC":
            return RLSMPCController(vf_model_path=model_path_pth, config_dir=self.base_dir, n_scenarios=3)
        else:
            raise ValueError(f"未知算法: {self.algo_name}")

    def run_simulation(self):
        """执行一个完整的生长周期仿真"""
        print(f"🚀 [Arena] 启动 {self.algo_name} 测评流水线...")
        
        # 重置环境，获取宏观排程参数
        obs, _ = self.env.reset()
        macro_params = {
            'A_h': self.env.A_h, 'A_l': self.env.A_l,
            'rho_h': self.env.rho_h, 'rho_l': self.env.rho_l,
            'xDs_l_initial': self.env.xDs_l_initial
        }

        # 准备记录容器 (使用 numpy 预分配内存，提升速度)
        max_steps = self.env.max_steps
        history = {
            'step': np.zeros(max_steps),
            'temp': np.zeros(max_steps),
            'profit': np.zeros(max_steps),
            'energy': np.zeros(max_steps),
            'violation': np.zeros(max_steps)
        }

        step = 0
        done = False
        t_solve_total = 0.0

        while not done and step < max_steps:
            # 1. 准备高精度预测数据 (TVP Forecast)
            tvp_forecast = self._generate_forecast(step)
            
            # 2. 计算动作并计时 (体现各算法的算力开销差异)
            t_start = time.time()
            
            if "SMPC" in self.algo_name:
                # SMPC 类控制器需要多场景全量输入 (Ns, Np)
                action = self.controller.compute_action(obs, macro_params, tvp_forecast)
            elif "MPC" in self.algo_name:
                # 确定性 MPC 只需要名义场景 (Scenario 0)
                nom_forecast = {k: v[0] if isinstance(v, np.ndarray) and v.ndim > 1 else v for k, v in tvp_forecast.items()}
                action = self.controller.compute_action(obs, macro_params, nom_forecast)
            else:
                # Baseline 和 Pure RL 只需要当前观测
                action = self.controller.compute_action(obs)
            
            t_solve_total += (time.time() - t_start)

            # 3. 环境步进
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 4. 记录数据 (键值与 pfal_env_dual.py 严格对齐)
            history['step'][step] = step
            history['temp'][step] = obs[5]    # 室内温度
            history['profit'][step] = reward  # 本步利润
            history['energy'][step] = info.get('step_cost', 0.0) # 修正键名
            history['violation'][step] = 1.0 if info.get('penalty', 0.0) > 0.0 else 0.0 # 修正越界判定
            
            step += 1
            done = terminated or truncated
            if step % 144 == 0: # 每天 (144个10分钟) 打印一次进度
                print(f"  Day {step // 144}: 当前累计利润 = {np.sum(history['profit'][:step]):.2f} RMB")

        # 5. 裁剪冗余内存并保存结果
        for k in history: history[k] = history[k][:step]
        self._save_results(history, t_solve_total / step)

    def _generate_forecast(self, current_step):
        """
        【核心重构】构造未来 N_p 步的随机过程预测序列。
        利用 Ornstein-Uhlenbeck (OU) 过程生成均值回归的动态误差，
        彻底模拟真实气象预报中“预报偏高但中途回落”的跨象限噪声。
        """
        Np = self.Np
        Ns = 3 # 3个场景：S0(名义), S1(高温偏移), S2(低温偏移)
        
        # 1. 安全提取环境中的“真实未来天气”作为基底
        start_idx = current_step
        max_idx = len(self.env.weather_series) - 1
        
        actual_To = np.zeros(Np)
        actual_Co = np.zeros(Np)
        actual_Ho = np.zeros(Np)
        
        for k in range(Np):
            idx = min(start_idx + k, max_idx)
            actual_To[k] = self.env.weather_series[idx, 0]
            actual_Co[k] = self.env.weather_series[idx, 1]
            actual_Ho[k] = self.env.weather_series[idx, 2]

        forecast = {
            'To': np.zeros((Ns, Np)),
            'Co': np.zeros((Ns, Np)),
            'Ho': np.zeros((Ns, Np)),
            'T_min': np.full((Ns, Np), self.env.cst['temp_min']),
            'T_max': np.full((Ns, Np), self.env.cst['temp_max']),
            'photo': np.zeros((Ns, Np)),
            'price': np.zeros((Ns, Np))
        }
        
        # 2. 填充确定性参数 (电价与光周期硬约束)
        for k in range(Np):
            t_hours = ((current_step + k) * self.env.dt) / 3600.0
            t_of_day = t_hours % 24.0
            photo_flag = 1.0 if 6.0 <= t_of_day < (6.0 + self.env.photoperiod) else 0.0
            current_price = self.env.economics.get_electricity_price(t_of_day)
            
            for s in range(Ns):
                forecast['photo'][s, k] = photo_flag
                forecast['price'][s, k] = current_price
                forecast['Co'][s, k] = actual_Co[k]
                forecast['Ho'][s, k] = actual_Ho[k]

        # 3. OU 过程引擎: dx = theta * (mu - x) + sigma * dW
        def generate_ou_noise(length, theta=0.15, mu=0.0, sigma=0.5):
            noise = np.zeros(length)
            x = 0.0 # 初始点预报误差通常较小
            for i in range(1, length):
                dx = theta * (mu - x) + sigma * np.random.normal()
                x += dx
                noise[i] = x
            return noise

        # 4. 生成多场景气象
        for s in range(Ns):
            if s == 0:
                # S0 名义预报：仅叠加极小的独立高斯白噪声
                noise = np.random.normal(0, 0.2, Np) 
            elif s == 1:
                # S1 高温预报场景：OU 均值回归向 +4.0°C 偏移，伴随强波动
                noise = generate_ou_noise(Np, theta=0.2, mu=4.0, sigma=0.8)
            else:
                # S2 低温预报场景：OU 均值回归向 -4.0°C 偏移，伴随强波动
                noise = generate_ou_noise(Np, theta=0.2, mu=-4.0, sigma=0.8)
                
            forecast['To'][s, :] = actual_To + noise

        return forecast

    def _save_results(self, history, avg_time):
        """数据落盘与性能汇总"""
        df = pd.DataFrame(history)
        csv_path = os.path.join(self.results_dir, "metrics.csv")
        df.to_csv(csv_path, index=False)
        
        summary = {
            'algo': self.algo_name,
            'total_profit_RMB': float(np.sum(history['profit'])),
            'avg_solve_time_ms': float(avg_time * 1000),
            'violation_rate': float(np.mean(history['violation'])),
            'total_energy_cost': float(np.sum(history['energy']))
        }
        
        with open(os.path.join(self.results_dir, "summary.yml"), 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, allow_unicode=True)
            
        print(f"\n✅ 测评完成！结果已存入: {self.results_dir}")
        print(f"🏆 总净利润: {summary['total_profit_RMB']:.2f} RMB | 平均算力开销: {summary['avg_solve_time_ms']:.2f} ms")

if __name__ == "__main__":
    # 使用方式: python evaluate_benchmark.py --algo RL_SMPC
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='BASELINE', help='算法名称')
    args = parser.parse_args()
    
    evaluator = BenchmarkEvaluator(algo_name=args.algo)
    evaluator.run_simulation()