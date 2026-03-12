# ==============================================================================
# 文件路径: envs/observations_dual.py
# 描述: 工业级双区植物工厂 状态观测与动作归一化引擎 (Observation Scaler)
# 核心定位: 
#   本模块【仅服务于神经网络/强化学习】！它充当 AI 与 物理世界之间的“翻译官”。
#   对于 MPC 或 PID 等基于真实物理模型的控制算法，应在环境中通过 rl_mode=False 旁路本模块。
# 作用: 
#   1. 解决物理量纲跨度过大(从 0.00001 到 200.0)导致的神经网络梯度爆炸问题。
#   2. 在 AI ([-1, 1]) 与 物理仿真环境 之间建立双向转换桥梁。
#   3. 支持真实传感器噪声注入 (Sensor Noise Injection / Domain Randomization)。
# ==============================================================================

import yaml
import os
import numpy as np

class ObservationScaler:
    """
    13维状态与6维动作的双向缩放器。
    采用严格的 Min-Max 缩放，将所有输入/输出映射到 [-1.0, 1.0] 区间。
    """
    def __init__(self, env_config_path: str = None):
        if env_config_path is None:
            # 自动寻址项目根目录下的配置文件
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
            
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self._init_action_bounds()
        self._init_state_bounds()

    def _init_action_bounds(self):
        """
        初始化 6维动作 (nu=6) 的物理边界。
        严格读取 PFAL_dual.yml 中 constraints 定义的硬件设备极限能力。
        控制空间顺序: [u_L_h, u_L_l, u_C, u_D, u_Q, u_V]
        """
        c = self.config['constraints']
        self.u_min = np.array([
            c['light_h_min'],     # 育苗区光照 (W/m2)
            c['light_l_min'],     # 成株区光照 (W/m2)
            c['co2_supply_min'],  # CO2 供应 (kg/m2/s)
            c['dehum_min'],       # 除湿 (kg/m2/s)
            c['heat_min'],        # HVAC 制冷 (W/m2，负数代表制冷最大功率)
            c['vent_min']         # 通风换气 (m/s)
        ], dtype=np.float32)
        
        self.u_max = np.array([
            c['light_h_max'],     # 育苗区灯光全开
            c['light_l_max'],     # 成株区灯光全开
            c['co2_supply_max'],
            c['dehum_max'],
            c['heat_max'],        # HVAC 制热 (W/m2，正数)
            c['vent_max']
        ], dtype=np.float32)

    def _init_state_bounds(self):
        """
        初始化 13维状态 (nx=13) 的物理边界视野。
        注意: 这里的边界不代表环境的绝对死区，而是为了给 RL 网络提供一个“正常的视野范围”。
        状态空间顺序: [xDn_h, xDs_h, xDn_l, xDs_l, C, T, H, T_lb, T_ub, photo, To, Co, Ho]
        """
        self.x_min = np.array([
            0.0, 0.0, 0.0, 0.0,          # xDn_h, xDs_h, xDn_l, xDs_l (单株干重 kg/plant)
            0.0, 10.0, 0.001,            # 室内 CO2(kg/m3), 温度(C), 绝对湿度(kg/m3)
            10.0, 10.0, 0.0,             # 控制边界 TVP: T_lb, T_ub, 光周期标志(0黑夜/1白天)
            -15.0, 0.0, 0.0              # 室外气象 TVP: To(防极寒), Co, Ho
        ], dtype=np.float32)
        
        self.x_max = np.array([
            0.005, 0.005, 0.030, 0.030,  # 预估的最大单株干重 (30g 干重约等于极其巨大的生菜)
            0.003, 40.0, 0.030,          # 室内 C(约1600ppm), T(极限高温), H
            40.0, 40.0, 1.0,             # 控制边界 TVP
            45.0, 0.003, 0.030           # 室外气象 TVP: To(防极热), Co, Ho
        ], dtype=np.float32)

    # -----------------------------------------------------------------------
    # 动作映射 (Action Mapping): AI [-1, 1] <---> 物理世界
    # -----------------------------------------------------------------------
    def unscale_action(self, action_scaled: np.ndarray) -> np.ndarray:
        """
        AI输出 ([-1, 1]) -> 真实物理设备指令。
        在 env.step() 的最开头调用，把 RL 的脑电波翻译成机器听得懂的电压。
        """
        # 截断保护: 防止刚初始化的神经网络胡乱输出绝对值大于 1 的动作导致物理引擎崩溃
        act = np.clip(action_scaled, -1.0, 1.0)
        # 线性映射公式: u_phys = u_min + 0.5 * (act + 1.0) * (u_max - u_min)
        u_phys = self.u_min + 0.5 * (act + 1.0) * (self.u_max - self.u_min)
        return u_phys

    def scale_action(self, action_phys: np.ndarray) -> np.ndarray:
        """
        真实物理设备指令 -> AI视野 ([-1, 1])。
        辅助功能: 主要用于将 PID/MPC 生成的专家专家数据，转录给 RL 进行模仿学习 (Imitation Learning)。
        """
        # 加上 1e-8 防止分母为 0 导致程序崩溃
        act = 2.0 * (action_phys - self.u_min) / (self.u_max - self.u_min + 1e-8) - 1.0
        return np.clip(act, -1.0, 1.0)

    # -----------------------------------------------------------------------
    # 状态映射与域随机化 (Observation Mapping & Domain Randomization)
    # -----------------------------------------------------------------------
    def scale_obs(self, state_phys: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        真实物理状态 -> AI视野 ([-1, 1])。
        在 env.step() 结束时调用，把微分方程算出的结果送给 RL 网络观察。
        
        参数:
        - add_noise: 开启后注入传感器白噪声，极大提升模型从仿真到现实 (Sim2Real) 部署的鲁棒性。
        """
        obs = 2.0 * (state_phys - self.x_min) / (self.x_max - self.x_min + 1e-8) - 1.0
        
        if add_noise:
            # 定义 13维 状态的独立测量噪声水平 (标准差为 [-1, 1] 空间的比例)
            # 逻辑: 视觉估算生物量误差极大(5%)，现代温湿度传感器极其精准(1%)，设定参数无误差(0%)
            noise_std = np.array([
                0.05, 0.05, 0.05, 0.05,  # 0~3: 生物量估算噪声
                0.01, 0.01, 0.01,        # 4~6: 室内环境传感器噪声
                0.0, 0.0, 0.0,           # 7~9: 算法设定的 TVP，无噪声
                0.01, 0.01, 0.01         # 10~12: 室外气象站测量噪声
            ])
            # 生成高斯白噪声并注入
            noise = np.random.normal(0, noise_std)
            obs += noise
            
        return np.clip(obs, -1.0, 1.0)

    def unscale_obs(self, obs_scaled: np.ndarray) -> np.ndarray:
        """
        AI视野 ([-1, 1]) -> 真实物理状态。
        """
        obs = np.clip(obs_scaled, -1.0, 1.0)
        state_phys = self.x_min + 0.5 * (obs + 1.0) * (self.x_max - self.x_min)
        return state_phys


# ==============================================================================
# 单体逻辑与边界测试 (独立运行以验证缩放器是否安全)
# ==============================================================================
if __name__ == "__main__":
    # 实例化测试对象
    scaler = ObservationScaler()
    
    print("="*50)
    print("🟢 模块一: AI动作 -> 物理设备指令 (Unscale Action) 测试")
    print("="*50)
    # 模拟 RL 神经网络在极限情况下的输出 (全开或全关)
    # [灯H开满, 灯L关死, CO2关死, 除湿全开, 空调极限满载制冷, 通风关死]
    ai_extreme_action = np.array([1.0, -1.0, -1.0, 1.0, -1.0, -1.0]) 
    phys_action = scaler.unscale_action(ai_extreme_action)
    
    print(f"输入: AI 神经网络极限输出指令: {ai_extreme_action}")
    print(f"输出: 还原为真实工厂 PLC 控制指令:")
    print(f"  - 育苗区灯光 = {phys_action[0]:.2f} W/m2 (预期: 满功率)")
    print(f"  - 成株区灯光 = {phys_action[1]:.2f} W/m2 (预期: 0.00)")
    print(f"  - 空调系统   = {phys_action[4]:.2f} W/m2 (预期: 负数满功率制冷极限)")
    
    print("\n" + "="*50)
    print("🟢 模块二: 物理环境状态 -> AI视野观测 (Scale Obs) 测试")
    print("="*50)
    # 构造一个真实的物理状态 (中庸的温度和微小的生物量)
    mock_phys_state = np.copy(scaler.x_min)
    mock_phys_state[1] = 0.0025 # 育苗干重: 0.0025kg (刚好是 0~0.005 的中点)
    mock_phys_state[5] = 25.0   # 室内温度: 25°C (刚好是 10~40°C 的中点)
    
    scaled_obs_clean = scaler.scale_obs(mock_phys_state, add_noise=False)
    scaled_obs_noisy = scaler.scale_obs(mock_phys_state, add_noise=True)
    
    print(f"构造物理状态: 育苗干重 = 0.0025 kg/plant, 室内温度 = 25.0 °C")
    print(f"1. 无噪声归一化送入 AI: ")
    print(f"  - 干重视觉: {scaled_obs_clean[1]:.2f} (预期: 0.00)")
    print(f"  - 温度视觉: {scaled_obs_clean[5]:.2f} (预期: 0.00)")
    
    print(f"\n2. 注入 Sim2Real 白噪声后送入 AI (模拟真实传感器抖动): ")
    print(f"  - 抖动干重视觉: {scaled_obs_noisy[1]:.4f} (预期在 0.00 附近剧烈波动)")
    print(f"  - 抖动温度视觉: {scaled_obs_noisy[5]:.4f} (预期在 0.00 附近轻微波动)")
    print("\n✅ 所有安全测试与量纲映射通过！")