# ==============================================================================
# 文件路径: common/economics.py
# 描述: 工业级集装箱植物工厂分时电价与经济核算引擎 (CasADi 符号计算兼容版)
# 修复: 【7. 物理参数硬编码问题】彻底解耦配置，所有经济、能效常数统一从 yaml 加载。
# 作用: 
#   1. 将控制器输出的物理量指令转化为绝对设备能耗 (kWh)。
#   2. 引入真实电网的峰平谷分时电价 (ToU) 计算每一控制步的电费成本。
#   3. 计算核心学术指标: EPI (Energy Performance Index)。
# ==============================================================================

import os
import yaml
import numpy as np
from typing import Dict, Tuple

class EconomicsCalculator:
    def __init__(self, env_config_path: str = None, use_tou: bool = True):
        """
        初始化经济核算器，动态挂载 YAML 配置。
        
        参数:
        env_config_path: 配置文件路径。若留空则自动寻址。
        use_tou: 是否开启峰平谷分时电价计算 (开启后才能体现 RL-SMPC 的时域错峰能力)
        """
        self.use_tou = use_tou
        
        # 1. 自动寻址与配置文件加载
        if env_config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
            
        with open(env_config_path, 'r', encoding='utf-8') as f:
            env_config = yaml.safe_load(f)
            
        # 提取经济配置块，并设置默认值以防配置文件缺失该区块
        econ_cfg = env_config.get('economics', {})
        
        # ---------------------------------------------------------
        # 工业设备能效参数与基础物料价格
        # ---------------------------------------------------------
        self.c_elec_flat = econ_cfg.get('c_elec_flat', 0.6)
        self.c_co2_price = econ_cfg.get('c_co2_price', 0.2)
        self.c_COP       = econ_cfg.get('c_COP', 3.0)
        self.c_dehum_eev = econ_cfg.get('c_dehum_eev', 3.0)
        self.fan_eff     = econ_cfg.get('fan_eff_const', 7.0785)

        # ---------------------------------------------------------
        # 分时电价 (Time-of-Use) 阶梯定义
        # ---------------------------------------------------------
        # 谷段 (Valley): 价格最低，系统会倾向于此时开灯
        self.price_valley = self.c_elec_flat * econ_cfg.get('tou_valley_ratio', 0.5)
        # 平段 (Flat)
        self.price_flat   = self.c_elec_flat         
        # 峰段 (Peak): 价格最高，此时开大功率设备会被重罚
        self.price_peak   = self.c_elec_flat * econ_cfg.get('tou_peak_ratio', 1.5)

    def get_electricity_price(self, time_of_day_hours: float) -> float:
        """
        根据当前一天中的时间 (0.0 ~ 24.0 小时) 返回瞬时电价。
        这正是 RL-SMPC 能够“预测电价并提前蓄冷/错峰亮灯”的数学基础。
        """
        if not self.use_tou:
            return self.c_elec_flat
            
        t = time_of_day_hours % 24.0
        
        # 谷电判定 (22:00 - 06:00)
        if t >= 22.0 or t < 6.0:
            return self.price_valley
        # 峰电判定 (10:00 - 15:00, 18:00 - 22:00)
        elif (10.0 <= t < 15.0) or (18.0 <= t < 22.0):
            return self.price_peak
        # 其余为平电
        else:
            return self.price_flat

    def compute_step_cost(self, 
                          u_unscaled: np.ndarray, 
                          A_h: float, 
                          A_l: float, 
                          A_total: float, 
                          dt_seconds: float, 
                          time_of_day_hours: float) -> Dict[str, float]:
        """
        计算单个控制步长内的绝对能耗 (kWh) 与经济成本。
        """
        u_L_h, u_L_l, u_C, u_D, u_Q, u_V = u_unscaled

        # ==========================================
        # 1. 核心设备绝对能耗计算 (单位统一转化为 kWh)
        # ==========================================
        dt_hours = dt_seconds / 3600.0

        # 双区独立 LED 照明能耗
        e_light_h = (u_L_h * A_h * dt_hours) / 1000.0
        e_light_l = (u_L_l * A_l * dt_hours) / 1000.0
        e_light_total = e_light_h + e_light_l

        # 共享空调(HVAC)能耗
        e_hvac = (abs(u_Q) * A_total * dt_hours) / 1000.0 / self.c_COP

        # 共享除湿机能耗
        water_removed_kg = u_D * A_total * dt_seconds
        e_dehum = water_removed_kg / self.c_dehum_eev

        # 共享通风风机能耗
        e_vent = (u_V / self.fan_eff) * dt_hours * A_total

        # ==========================================
        # 2. 气源与电量汇总
        # ==========================================
        total_elec_kwh = e_light_total + e_hvac + e_dehum + e_vent
        co2_consumed_kg = u_C * A_total * dt_seconds

        # ==========================================
        # 3. 经济成本计算 (引入 ToU 电价)
        # ==========================================
        current_elec_price = self.get_electricity_price(time_of_day_hours)
        
        cost_elec = total_elec_kwh * current_elec_price
        cost_co2 = co2_consumed_kg * self.c_co2_price
        
        total_cost = cost_elec + cost_co2

        return {
            "e_light_kwh": e_light_total,
            "e_hvac_kwh": e_hvac,
            "e_dehum_kwh": e_dehum,
            "e_vent_kwh": e_vent,
            "total_elec_kwh": total_elec_kwh,
            "co2_kg": co2_consumed_kg,
            "current_elec_price": current_elec_price,
            "cost_elec": cost_elec,
            "cost_co2": cost_co2,
            "total_cost": total_cost
        }

    def compute_epi(self, total_energy_kwh: float, total_fresh_yield_kg: float) -> float:
        """
        计算综合能耗指标 EPI (Energy Performance Index)。
        """
        if total_fresh_yield_kg <= 0:
            return float('inf') 
        return total_energy_kwh / total_fresh_yield_kg

# ==============================================================================
# 测试模块
# ==============================================================================
if __name__ == "__main__":
    calc = EconomicsCalculator(use_tou=True)
    
    # 模拟一个相同的设备开启状态: 
    # 光照强开 (80 W/m2), 空调制冷全开 (-150 W/m2), 不开CO2/除湿/风机
    # 假设此时经过解算，育苗区 10 m2，成株区 30 m2
    u_mock = np.array([80.0, 80.0, 0.0, 0.0, -150.0, 0.0])
    Ah, Al, Atot = 10.0, 30.0, 40.0
    dt = 600.0 # 10分钟

    # 测试 1: 在半夜 03:00 (谷电阶段) 运行该指令
    print("--- 半夜 03:00 运行 (谷电) ---")
    res_valley = calc.compute_step_cost(u_mock, Ah, Al, Atot, dt, time_of_day_hours=3.0)
    print(f"总耗电: {res_valley['total_elec_kwh']:.3f} kWh")
    print(f"当前电价: {res_valley['current_elec_price']} RMB/kWh")
    print(f"该步长电费: {res_valley['cost_elec']:.3f} RMB\n")

    # 测试 2: 在中午 13:00 (峰电阶段) 运行同样的指令
    print("--- 中午 13:00 运行 (峰电) ---")
    res_peak = calc.compute_step_cost(u_mock, Ah, Al, Atot, dt, time_of_day_hours=13.0)
    print(f"总耗电: {res_peak['total_elec_kwh']:.3f} kWh")
    print(f"当前电价: {res_peak['current_elec_price']} RMB/kWh")
    print(f"该步长电费: {res_peak['cost_elec']:.3f} RMB\n")
    
    # EPI 测试
    print("--- 宏观 EPI 计算演示 ---")
    epi = calc.compute_epi(total_energy_kwh=1500.0, total_fresh_yield_kg=105.0)
    print(f"30天生产批次的 EPI: {epi:.2f} kWh/kg")