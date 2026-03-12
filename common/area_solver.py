# ==============================================================================
# 文件路径: common/area_solver.py
# 描述: 工业级双区植物工厂排程解算器 (引入批次、GCD与速率匹配)
# 作用: 解决连续农业排程的“不可分割”问题，确保物理上完全可行。
# 修正: 修复了不达标时丢失面积与产能返回值的 Bug，确保优化器能获取完整信息。
# ==============================================================================

import yaml
import math
import os
from typing import Dict

class AreaSolver:
    def __init__(self, config_path: str = None, env_config_path: str = None):
        """初始化解算器，加载物理参数"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config_path is None:
            config_path = os.path.join(base_dir, "configs", "optimizers.yml")
        if env_config_path is None:
            env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            opt_config = yaml.safe_load(f)
            self.target_yield = opt_config['constraints']['target_daily_yield_kg']
            self.max_penalty = opt_config['constraints']['max_unfeasible_penalty']
            
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
            self.total_area = env_config['geometry']['total_growing_area']
            
        self.harvest_fresh_weight_per_plant = 0.1  # 每株 0.1 kg (100g)

    def solve(self, t_h: float, t_l: float, rho_h: float, rho_l: float) -> Dict[str, float]:
        """
        基于最大公约数(GCD)和批次匹配的离散排程解析法。
        """
        result = {
            "A_h": 0.0,
            "A_l": 0.0,
            "daily_yield_kg": 0.0,
            "is_feasible": False,
            "penalty": 0.0,
            "batch_cycle_days": 0,
            "plants_per_batch": 0
        }

        # 1. 密度物理拦截: 育苗区密度必须大于成株区密度
        if rho_h <= rho_l:
            result["penalty"] = self.max_penalty
            return result

        # 2. 离散化天数 (确保 T_h 和 T_l 是整数天)
        t_h_int = int(round(t_h))
        t_l_int = int(round(t_l))
        
        if t_h_int <= 0 or t_l_int <= 0:
            result["penalty"] = self.max_penalty
            return result

        # 3. 最大公约数计算流转周期 (Batch Cycle Problem)
        # 例如 14天 和 21天，GCD 为 7天。每 7 天做一次移栽流转。
        d = math.gcd(t_h_int, t_l_int)
        
        # 4. 计算同时在线的批次数
        n_h = t_h_int // d   # 育苗区同时有几个批次
        n_l = t_l_int // d   # 成株区同时有几个批次

        # 5. 求解最大单批次植株数 (Rate Matching & Area Constraint)
        # 公式: N_batch * (n_h / rho_h + n_l / rho_l) <= A_total
        area_factor_per_plant = (n_h / rho_h) + (n_l / rho_l)
        
        max_plants_per_batch = math.floor(self.total_area / area_factor_per_plant)
        
        if max_plants_per_batch <= 0:
            result["penalty"] = self.max_penalty
            return result

        # 6. 反推各区真实占用面积 (离散化后的真实面积，通常会略微小于 A_total)
        A_h = n_h * (max_plants_per_batch / rho_h)
        A_l = n_l * (max_plants_per_batch / rho_l)

        # 7. 计算每日平摊产能
        # 每 d 天产出一个批次，所以日均产能 = 单批次数量 / d
        daily_yield_kg = (max_plants_per_batch / d) * self.harvest_fresh_weight_per_plant

        # 【关键修复】：无论是否达标，都必须将计算出的物理数据写入 result！
        # 否则优化器在产能不足时拿到的 A_h 永远是 0，无法进行有效进化。
        result["A_h"] = A_h
        result["A_l"] = A_l
        result["daily_yield_kg"] = daily_yield_kg
        result["batch_cycle_days"] = d
        result["plants_per_batch"] = max_plants_per_batch

        # 8. 硬约束校验
        if daily_yield_kg < self.target_yield:
            # 产能不达标，赋予惩罚。叠加基于差距的线性惩罚，给 GA 算法指明“进化方向”
            result["penalty"] = self.max_penalty + (self.target_yield - daily_yield_kg) * 10000.0 
            result["is_feasible"] = False
        else:
            # 完全合法
            result["is_feasible"] = True

        return result

# ==============================================================================
# 单体测试
# ==============================================================================
if __name__ == "__main__":
    solver = AreaSolver()
    
    # 模拟一组数据: 育苗 14天, 成株 21天 (GCD=7天), 密度分别为 120 和 30
    print("--- 测试 1: 完美的工业批次排程 (GCD=7) ---")
    res1 = solver.solve(t_h=14, t_l=21, rho_h=120, rho_l=30)
    for k, v in res1.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
            
    # 模拟一组差劲的数据: GCD 突变导致单批次极小
    print("\n--- 测试 2: 糟糕的排程 (GCD=1，引发高额惩罚) ---")
    res2 = solver.solve(t_h=15, t_l=22, rho_h=120, rho_l=30)
    for k, v in res2.items():
        if isinstance(v, float):
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")