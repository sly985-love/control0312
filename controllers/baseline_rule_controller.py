# ==============================================================================
# 文件路径: controllers/baseline_rule_controller.py
# 描述: 农学经验规则与 PID 追踪控制器 (传统植物工厂基线)
# 核心逻辑:
#   1. 光照与 CO2: 严格绑定光周期 (白天满功率，夜间全关防止浪费)。
#   2. 温度 (HVAC): 基于当前误差的比例控制 (P-Control)，不考虑未来电价和天气。
#   3. 湿度与通风: 阈值触发式控制 (Bang-Bang Control)。
# ==============================================================================

import os
import yaml
import numpy as np

class BaselineRuleController:
    """
    传统植物工厂的经验规则与 PID 混合控制器。
    作为智能算法 (MPC/RL) 的对标基线 (Baseline)，用于证明智能算法的节能与柔性优势。
    """
    def __init__(self, config_dir: str = None):
        # 加载约束配置以获取设备的物理极限
        base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
        
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        cst = self.config['constraints']
        
        # 提取设备的物理动作边界 [uL_h, uL_l, uC, uD, uQ, uV]
        self.u_min = np.array([
            cst['light_h_min'], cst['light_l_min'], cst['co2_supply_min'], 
            cst['dehum_min'], cst['heat_min'], cst['vent_min']
        ])
        self.u_max = np.array([
            cst['light_h_max'], cst['light_l_max'], cst['co2_supply_max'], 
            cst['dehum_max'], cst['heat_max'], cst['vent_max']
        ])

        # ---------------------------------------------------------
        # PID 控制器参数整定 (Tuning Parameters)
        # ---------------------------------------------------------
        # 空调 P 控制器增益 (W/m2 / °C)
        self.Kp_temp = 50.0  
        # CO2 阀门 P 控制器增益
        self.Kp_co2 = 50.0
        # 除湿机 P 控制器增益
        self.Kp_hum = 10.0

    def compute_action(self, obs: np.ndarray) -> np.ndarray:
        """
        根据当前物理环境状态，输出下一时刻的控制指令。
        参数:
        - obs: 13维的纯物理状态数组 (rl_mode=False 时 env.step 返回的值)
          [0~3: 生物量, 4: C, 5: T, 6: H, 7: T_min, 8: T_max, 9: photo, 10: To, 11: Co, 12: Ho]
        返回:
        - action: 6维物理控制指令
        """
        # 1. 状态解析
        C_indoor = obs[4]       # 室内 CO2 浓度 (kg/m3)
        T_indoor = obs[5]       # 室内温度 (°C)
        H_indoor = obs[6]       # 室内绝对湿度 (kg/m3)
        
        T_target_min = obs[7]   # 温度下限
        T_target_max = obs[8]   # 温度上限
        photo_flag = obs[9]     # 光周期标志 (1为白天，0为黑夜)
        
        T_outdoor = obs[10]     # 室外温度
        H_outdoor = obs[12]     # 室外湿度

        # 初始化动作数组
        action = np.zeros(6, dtype=np.float32)

        # =========================================================
        # 规则 1: 光照控制 (Day/Night Bang-Bang Control)
        # 农学常识: 白天必须开灯满足光合作用，夜间必须关灯让植物进行暗期代谢。
        # =========================================================
        if photo_flag == 1.0:
            # 传统做法：只要是白天，灯光直接拉满 (不考虑电价，极度费电)
            action[0] = self.u_max[0]  # 育苗区灯光
            action[1] = self.u_max[1]  # 成株区灯光
        else:
            action[0] = 0.0
            action[1] = 0.0

        # =========================================================
        # 规则 2: CO2 补气控制 (P-Control + 光周期拦截)
        # 农学常识: 夜间植物只呼吸不光合，严禁补 CO2 (防止气源浪费和毒害)。
        # =========================================================
        target_co2 = 0.0018 # 约 1000 ppm 的目标浓度
        if photo_flag == 1.0 and C_indoor < target_co2:
            # P-Control: 浓度差距越大，阀门开度越大
            co2_error = target_co2 - C_indoor
            action[2] = self.Kp_co2 * co2_error
        else:
            action[2] = 0.0

        # =========================================================
        # 规则 3: 温度控制 HVAC (P-Control 死区追踪)
        # 逻辑: 只有当温度突破上下限时，空调才启动发力。
        # 痛点: 传统 PID 看不到未来电价，在正午峰电时如果温度超标，也会硬抗着昂贵电价开满空调。
        # =========================================================
        target_temp_mid = (T_target_min + T_target_max) / 2.0
        
        if T_indoor > T_target_max:
            # 太热了，需要制冷 (uQ 为负数)
            temp_error = T_target_max - T_indoor # 负值
            action[4] = self.Kp_temp * temp_error
        elif T_indoor < T_target_min:
            # 太冷了，需要制热 (uQ 为正数)
            temp_error = T_target_min - T_indoor # 正值
            action[4] = self.Kp_temp * temp_error
        else:
            # 温度在舒适死区内，关闭空调省电
            action[4] = 0.0

        # =========================================================
        # 规则 4: 湿度控制 (P-Control)
        # 逻辑: 高于警戒线时开启除湿机。
        # =========================================================
        hum_target_max = self.config['constraints']['hum_max'] - 0.001 # 留一点安全余量
        if H_indoor > hum_target_max:
            hum_error = H_indoor - hum_target_max
            action[3] = self.Kp_hum * hum_error
        else:
            action[3] = 0.0

        # =========================================================
        # 规则 5: 通风控制 (Rule-based 应急散热/排湿)
        # 逻辑: 如果室内太热/太湿，且室外凉爽/干燥，则开窗通风白嫖大自然冷量。
        # =========================================================
        # 设定基础微风换气 (防止密闭缺氧)
        action[5] = self.u_max[5] * 0.1 
        
        # 应急通风判断
        if (T_indoor > T_target_max and T_outdoor < T_indoor) or \
           (H_indoor > hum_target_max and H_outdoor < H_indoor):
            action[5] = self.u_max[5] * 0.8 # 开大风机

        # =========================================================
        # 终极物理安全限幅 (Hardware Interlock)
        # =========================================================
        action = np.clip(action, self.u_min, self.u_max)
        
        return action

# ==============================================================================
# 控制器仿真测试 (验证基线在不同工况下的反应)
# ==============================================================================
if __name__ == "__main__":
    controller = BaselineRuleController()
    
    print("==================================================")
    print("🌿 传统植物工厂基准控制器 (Baseline) 测试")
    print("==================================================")
    
    # 构造测试状态 1: 白天，极其炎热闷热的温室
    obs_day_hot = np.zeros(13)
    obs_day_hot[4] = 0.0006   # C: CO2被吸光了 (低于 1000ppm)
    obs_day_hot[5] = 30.0     # T: 30度 (远超 24度上限)
    obs_day_hot[6] = 0.020    # H: 湿度极高
    obs_day_hot[7] = 18.0     # T_min
    obs_day_hot[8] = 24.0     # T_max
    obs_day_hot[9] = 1.0      # Photo: 白天
    obs_day_hot[10] = 35.0    # To: 室外更热 (不能通风降温)
    obs_day_hot[12] = 0.010   # Ho: 室外较干
    
    action_1 = controller.compute_action(obs_day_hot)
    print("\n[工况 1] 白天 + 高温超标 + 低CO2 + 室外更热:")
    print(f"  💡 灯光 (H/L) : {action_1[0]:.1f} / {action_1[1]:.1f} (预期: 全开)")
    print(f"  💨 CO2 阀门   : {action_1[2]:.6f} (预期: 开启补气)")
    print(f"  ❄️ 空调功率   : {action_1[4]:.1f} W/m2 (预期: 满负荷制冷 < 0)")
    print(f"  🌪️ 通风机     : {action_1[5]:.2f} m/s (预期: 开启排湿，因为室外干)")
    
    # 构造测试状态 2: 深夜，温度适宜
    obs_night_cool = np.copy(obs_day_hot)
    obs_night_cool[5] = 22.0  # T: 22度 (在 18~24 之间)
    obs_night_cool[9] = 0.0   # Photo: 黑夜
    
    action_2 = controller.compute_action(obs_night_cool)
    print("\n[工况 2] 黑夜 + 温度适宜:")
    print(f"  💡 灯光 (H/L) : {action_2[0]:.1f} / {action_2[1]:.1f} (预期: 全关)")
    print(f"  💨 CO2 阀门   : {action_2[2]:.6f} (预期: 全关，防浪费)")

    print(f"  ❄️ 空调功率   : {action_2[4]:.1f} W/m2 (预期: 0.0，省电)")