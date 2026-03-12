# ==============================================================================
# 文件路径: macro_optimizers/grid_search.py
# 描述: 宏观排程决策层 - 网格暴力搜索基线 (Grid Search Baseline)
# 核心使命: 
#   通过遍历预设的所有参数组合，寻找全局最优排程。
#   主要作用是作为“地主家的傻儿子”基准，用来在论文中衬托贝叶斯优化 (BO) 的极速与高效。
# 架构护城河:
#   1. 运筹学绝对壁垒剪枝: 提前计算空间，直接跳过无效组合，极大缓解组合爆炸。
#   2. 原生类型强转保护: 杜绝 Numpy int64/float32 写入 YAML 时的序列化崩溃。
#   3. 单例环境复用: 彻底杜绝数千次迭代带来的 C++ 底层内存泄漏。
# ==============================================================================

import os
import yaml
import time
import itertools
import numpy as np

# 导入底层物理仿真车间与快速评估基线
from envs.pfal_env_dual import PFALEnvDual
from controllers.baseline_rule_controller import BaselineRuleController
from common.area_solver import AreaSolver

class GridSearchScheduler:
    """
    网格搜索排程器。
    遍历所有可能的变量组合，寻找利润最高的生长周期与密度配置。
    """
    def __init__(self, config_dir: str = None):
        print("[Grid Search] 正在初始化顶层暴力评估沙盒...")
        
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_config_path = os.path.join(self.base_dir, "configs", "envs", "PFAL_dual.yml")
        
        with open(self.env_config_path, 'r', encoding='utf-8') as f:
            self.env_cfg = yaml.safe_load(f)
            
        # 实例化极速评估套件 (严格只实例化一次，杜绝内存泄漏)
        self.env = PFALEnvDual(config_dir=self.base_dir, rl_mode=False)
        self.controller = BaselineRuleController(config_dir=self.base_dir)
        self.area_solver = AreaSolver()

    def run(self):
        """执行网格搜索主干流程"""
        
        # ==========================================================
        # 1. 定义搜索网格 (Search Grid)
        # 注意: 现实中如果每个变量取 10 个值，5 个变量就是 10万次仿真！
        # 这里为了演示基线，我们采用相对稀疏的网格 (3x3x3x3x2 = 162 种组合)
        # ==========================================================
        grid = {
            't_h_days': [10, 15, 20],               # 育苗天数
            't_l_days': [20, 25, 30],               # 成株天数
            'rho_h': [80.0, 100.0, 120.0],          # 育苗密度
            'rho_l': [20.0, 25.0, 30.0],            # 成株密度
            'photoperiod_hours': [12.0, 16.0]       # 光周期
        }
        
        # 提取参数名与对应的值列表
        keys = list(grid.keys())
        values = list(grid.values())
        
        # 使用 itertools.product 生成所有排列组合
        all_combinations = list(itertools.product(*values))
        total_runs = len(all_combinations)
        
        print(f"网格构建完毕！共计 {total_runs} 种物理组合。开始执行暴力推演...\n")
        
        best_profit = -np.inf
        best_params = None
        
        # 统计学记录
        skipped_count = 0
        simulated_count = 0
        start_time = time.time()

        # ==========================================================
        # 2. 开始大循环遍历
        # ==========================================================
        for idx, combination in enumerate(all_combinations):
            # 将当前组合打包为字典
            params = dict(zip(keys, combination))
            
            t_h = params['t_h_days']
            t_l = params['t_l_days']
            rho_h = params['rho_h']
            rho_l = params['rho_l']
            photo = params['photoperiod_hours']

            # 【高能预警：运筹学空间拦截】
            # 在启动极其耗时的仿真前，先问问数学公式：集装箱装得下吗？
            area_res = self.area_solver.solve(t_h, t_l, rho_h, rho_l)
            if not area_res['is_feasible']:
                skipped_count += 1
                continue # 直接无情枪毙，跳过仿真！
            
            # 物理空间可行，启动仿真！
            simulated_count += 1
            
            # 环境状态重置
            obs, _ = self.env.reset(options={
                't_h': t_h, 't_l': t_l, 
                'rho_h': rho_h, 'rho_l': rho_l, 
                'photoperiod': photo
            })
            
            total_profit = 0.0
            done = False
            
            # 极速沙盘推演
            while not done:
                action = self.controller.compute_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_profit += reward
                done = terminated or truncated

            # 更新全局最优解
            if total_profit > best_profit:
                best_profit = total_profit
                # 【防崩溃修复】：强制将类型转换为 Python 原生类型，防止 yaml.dump 报错
                best_params = {
                    't_h_days': int(t_h),
                    't_l_days': int(t_l),
                    'rho_h': float(rho_h),
                    'rho_l': float(rho_l),
                    'photoperiod_hours': float(photo)
                }

            # 打印进度 (每 20 次有效仿真打印一次)
            if simulated_count % 20 == 0:
                print(f"进度: {idx+1}/{total_runs} | 已找到的当前最高利润: {best_profit:.2f} | 因空间不足已剪枝: {skipped_count} 次")

        # ==========================================================
        # 3. 结果汇总与文件落盘
        # ==========================================================
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("🏁 网格暴力搜索完毕！")
        print("="*60)
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"总组合数: {total_runs} | 实际仿真数: {simulated_count} | 运筹学拦截数: {skipped_count}")
        print(f"拦截率 (节约算力): {(skipped_count/total_runs)*100:.1f}%\n")
        
        if best_params is None:
            print("❌ 灾难：所有预设的网格组合在物理空间上均不合法！请重新调整网格范围。")
            return

        print("🏆 全局最优宏观排程：")
        print(f"  💰 估算总净利润: {best_profit:.2f}")
        for k, v in best_params.items():
            print(f"  📌 {k:<18}: {v}")

        # 架构闭环：覆盖 YAML 文件
        output_dir = os.path.join(self.base_dir, "configs", "macro_best")
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "grid_best_schedule.yml")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            yaml.dump({'optimal_scheduling': best_params}, f, allow_unicode=True, default_flow_style=False)
            
        print(f"\n🎯 暴力求解最优排程已落盘至: {out_file}")
        print("================================================================\n")


if __name__ == "__main__":
    scheduler = GridSearchScheduler()
    scheduler.run()