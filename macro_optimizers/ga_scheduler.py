# ==============================================================================
# 文件路径: macro_optimizers/ga_scheduler.py
# 描述: 宏观排程决策层 - 启发式遗传算法排程基线 (Genetic Algorithm Scheduler)
# 修正: 
#   1. 完美对接离散 GCD AreaSolver，直接读取产能惩罚梯度。
#   2. 【修复9: 内存管理隐患】引入 _init_sandbox 机制与显式 gc.collect()，
#      在长期寻优中定期销毁重建物理环境与控制器句柄，彻底根除跨语言内存溢出。
# ==============================================================================

import os
import yaml
import time
import numpy as np
import gc  # 引入 Python 垃圾回收模块
import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
from envs.pfal_env_dual import PFALEnvDual
from controllers.baseline_rule_controller import BaselineRuleController
from common.area_solver import AreaSolver

class GAScheduler:
    def __init__(self, config_dir: str = None, pop_size: int = 20, generations: int = 30):
        print("[GA Scheduler] 正在初始化进化算法评估沙盒...")
        
        self.base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_config_path = os.path.join(self.base_dir, "configs", "envs", "PFAL_dual.yml")
        
        with open(self.env_config_path, 'r', encoding='utf-8') as f:
            self.env_cfg = yaml.safe_load(f)
            
        self.area_solver = AreaSolver()
        self.total_area = self.env_cfg['geometry']['total_growing_area']
        
        # GA 超参数
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = 0.2
        self.crossover_rate = 0.8
        
        self.bounds = np.array([
            [10.0, 25.0],   # t_h
            [15.0, 35.0],   # t_l
            [80.0, 150.0],  # rho_h
            [20.0, 40.0],   # rho_l
            [10.0, 20.0]    # photoperiod
        ])
        self.num_genes = len(self.bounds)
        
        # 初始挂载沙盒
        self._init_sandbox()

    def _init_sandbox(self):
        """【防御级重构】: 定期销毁并重建环境与控制器句柄，强制释放 C++ 内存池"""
        if hasattr(self, 'env'):
            del self.env
        if hasattr(self, 'controller'):
            del self.controller
            
        # 强制唤醒垃圾回收器，清理悬空指针
        gc.collect() 
        
        self.env = PFALEnvDual(config_dir=self.base_dir, rl_mode=False)
        self.controller = BaselineRuleController(config_dir=self.base_dir)

    def _decode_chromosome(self, chromosome: np.ndarray) -> dict:
        c_clipped = np.clip(chromosome, self.bounds[:, 0], self.bounds[:, 1])
        return {
            't_h_days': int(np.round(c_clipped[0])),  
            't_l_days': int(np.round(c_clipped[1])),  
            'rho_h': float(c_clipped[2]),
            'rho_l': float(c_clipped[3]),
            'photoperiod_hours': float(c_clipped[4])
        }

    def _evaluate_fitness(self, chromosome: np.ndarray) -> float:
        params = self._decode_chromosome(chromosome)
        
        area_res = self.area_solver.solve(
            params['t_h_days'], params['t_l_days'], 
            params['rho_h'], params['rho_l']
        )
        
        if not area_res['is_feasible']:
            return -float(area_res['penalty'])
            
        obs, _ = self.env.reset(options={
            't_h': params['t_h_days'], 't_l': params['t_l_days'], 
            'rho_h': params['rho_h'], 'rho_l': params['rho_l'], 
            'photoperiod': params['photoperiod_hours']
        })
        
        total_profit = 0.0
        done = False
        while not done:
            action = self.controller.compute_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_profit += reward
            done = terminated or truncated

        return float(total_profit)

    def run(self):
        print("="*60)
        print(f"🧬 遗传算法排程优化启动 | 种群大小: {self.pop_size} | 繁衍代数: {self.generations}")
        print("="*60)
        start_time = time.time()
        
        population = np.random.uniform(
            low=self.bounds[:, 0], 
            high=self.bounds[:, 1], 
            size=(self.pop_size, self.num_genes)
        )
        
        best_overall_fitness = -np.inf
        best_overall_chromosome = None
        
        for gen in range(self.generations):
            # ==========================================
            # 内存防溢出保护：每 5 代强制刷新一次物理沙盒
            # ==========================================
            if gen > 0 and gen % 5 == 0:
                self._init_sandbox()
                
            fitnesses = np.array([self._evaluate_fitness(ind) for ind in population])
            
            current_best_idx = np.argmax(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]
            
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_chromosome = population[current_best_idx].copy()
                
            print(f"世代 [Gen {gen+1:02d}/{self.generations}] | 本代最高利润: {current_best_fitness:.2f} | 历史最高: {best_overall_fitness:.2f}")
            
            new_population = np.zeros_like(population)
            for i in range(self.pop_size):
                tournament_indices = np.random.choice(self.pop_size, size=3, replace=False)
                winner_idx = tournament_indices[np.argmax(fitnesses[tournament_indices])]
                new_population[i] = population[winner_idx]
                
            for i in range(0, self.pop_size, 2):
                if i+1 < self.pop_size and np.random.rand() < self.crossover_rate:
                    cpt = np.random.randint(1, self.num_genes)
                    parent1, parent2 = new_population[i].copy(), new_population[i+1].copy()
                    new_population[i, cpt:] = parent2[cpt:]
                    new_population[i+1, cpt:] = parent1[cpt:]
                    
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    gene_idx = np.random.randint(self.num_genes)
                    mutation_scale = (self.bounds[gene_idx, 1] - self.bounds[gene_idx, 0]) * 0.1
                    noise = np.random.normal(0, mutation_scale)
                    new_population[i, gene_idx] += noise
                    new_population[i, gene_idx] = np.clip(
                        new_population[i, gene_idx], 
                        self.bounds[gene_idx, 0], 
                        self.bounds[gene_idx, 1]
                    )
                    
            weakest_idx = np.argmin(fitnesses)
            new_population[weakest_idx] = best_overall_chromosome
            
            population = new_population

        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print("🏁 遗传算法进化完毕！")
        print("="*60)
        print(f"总耗时: {elapsed_time:.2f} 秒 (总计评估 {self.pop_size * self.generations} 个个体)")
        
        best_params = self._decode_chromosome(best_overall_chromosome)
        
        final_area_res = self.area_solver.solve(
            best_params['t_h_days'], best_params['t_l_days'], 
            best_params['rho_h'], best_params['rho_l']
        )
        
        print("🏆 进化出的全局最优排程：")
        print(f"  💰 估算总净利润: {best_overall_fitness:.2f}")
        for k, v in best_params.items():
            print(f"  📌 {k:<18}: {v}")
            
        print("\n🏭 【工业批次排程详情】:")
        print(f"  🔄 移栽流转周期 (GCD) : 每 {final_area_res.get('batch_cycle_days', 0)} 天移栽一次")
        print(f"  🌱 单批次植株数量    : {final_area_res.get('plants_per_batch', 0)} 株")
        print(f"  📦 预估日均产能      : {final_area_res.get('daily_yield_kg', 0):.2f} kg/天")

        output_dir = os.path.join(self.base_dir, "configs", "macro_best")
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, "ga_best_schedule.yml")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            yaml.dump({'optimal_scheduling': best_params}, f, allow_unicode=True, default_flow_style=False)
            
        print(f"\n🎯 进化求解最优排程已落盘至: {out_file}")
        print("================================================================\n")

if __name__ == "__main__":
    scheduler = GAScheduler(pop_size=20, generations=30)
    scheduler.run()