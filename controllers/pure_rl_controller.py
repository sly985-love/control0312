# ==============================================================================
# 文件路径: controllers/pure_rl_controller.py
# 描述: 纯强化学习控制器 (Pure Reinforcement Learning Controller)
# 核心技术: 
#   1. 黑盒策略输出 (Black-box Policy): 依托预训练的深度神经网络直接输出动作。
#   2. 极速推理 (O(1) Inference): 无需在线规划，实现亚毫秒级控制延迟。
# 架构升级:
#   - 增加强制 CPU 推理机制，确保跨硬件边缘部署 (Edge Deployment) 的绝对稳定。
#   - 严密的张量维度 (Shape) 校验与对齐，防止 SB3 向量化导致的张量崩溃。
# ==============================================================================

import os
import warnings
import numpy as np

# 导入强化学习常用算法库
try:
    from stable_baselines3 import SAC, PPO
except ImportError:
    warnings.warn("[警告] 尚未安装 stable_baselines3！请执行 pip install stable-baselines3")

# 导入 AI 与物理世界的“双向翻译官”
from envs.observations_dual import ObservationScaler

class PureRLController:
    """
    纯强化学习策略控制器。
    加载离线训练好的权重文件 (.zip)，在在线评估/部署时执行极速的前向推理。
    """
    def __init__(self, model_path: str, algo: str = 'SAC', config_dir: str = None):
        """
        参数:
        - model_path: 训练好的 SB3 模型权重路径 (例如 'models/sac_best_model.zip')
        - algo: 使用的算法类别 ('SAC' 或 'PPO')
        - config_dir: 配置文件目录路径
        """
        self.algo = algo.upper()
        self.model_path = model_path
        
        # 1. 初始化 Observation Scaler
        # 重要逻辑: 评估时由于环境的 rl_mode=False (输出物理量),
        # 控制器必须自己先用 Scaler 把物理量翻译为 [-1, 1], 才能喂给 AI。
        base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
        self.scaler = ObservationScaler(self.env_config_path)
        
        # 2. 加载预训练大模型 (强制使用 CPU 保证边缘工控机兼容性)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 RL 权重文件: {model_path}。请先执行训练脚本！")
            
        print(f"[PureRLController] 正在加载 {self.algo} 神经网络权重...")
        
        # kwargs 字典用于强制覆盖模型内部的设备设定，防止 CUDA 缺失报错
        custom_objects = {"device": "cpu"} 
        
        if self.algo == 'SAC':
            self.model = SAC.load(model_path, custom_objects=custom_objects)
        elif self.algo == 'PPO':
            self.model = PPO.load(model_path, custom_objects=custom_objects)
        else:
            raise ValueError(f"不支持的强化学习算法类型: {self.algo}")
            
        print("[PureRLController] 模型加载成功！处于极速推理就绪状态。")

    def compute_action(self, obs: np.ndarray, macro_params: dict = None, tvp_forecast: dict = None, return_horizon: bool = False):
        """
        多态核心控制接口：与 Baseline / NMPC / SMPC 保持绝对一致的函数签名！
        
        参数:
        - obs: 当前真实 13 维物理状态 (物理量级)
        - macro_params: 纯 RL 模式忽略 (已隐含在 obs 中的生物量与状态里)
        - tvp_forecast: 纯 RL 模式忽略 (标准 RL 只看当前时刻，不看未来视野)
        """
        # ==========================================================
        # 1. 物理视界 -> AI 视界 (缩放与降噪)
        # ==========================================================
        # 确保输入是 1D 数组 (13,)
        if obs.ndim != 1 or obs.shape[0] != 13:
            raise ValueError(f"[PureRLController] 观测状态维度错误！期望 (13,), 实际 {obs.shape}")
            
        # 部署推理阶段，关闭 add_noise，我们需要绝对确定性的反应
        scaled_obs = self.scaler.scale_obs(obs, add_noise=False)
        
        # 防护机制: 重塑为 (1, 13) 迎合 SB3 底层的预测期望
        scaled_obs_batch = scaled_obs.reshape(1, -1)
        
        # ==========================================================
        # 2. 神经网络极速前向推理
        # ==========================================================
        # deterministic=True: 屏蔽高斯探索噪声，只输出当前认为最高收益的确定性动作
        scaled_action_batch, _states = self.model.predict(scaled_obs_batch, deterministic=True)
        
        # 提取真正的 1D 动作数组 (6,)
        scaled_action = scaled_action_batch.flatten()
        
        # ==========================================================
        # 3. AI 视界 -> 物理世界 (动作还原)
        # ==========================================================
        # 把 [-1, 1] 的神经元激活值翻译成 PLC 听得懂的瓦特/流速
        physical_action = self.scaler.unscale_action(scaled_action)
        
        # 处理多态请求
        if return_horizon:
            # RL 的本质是只顾眼前的“本能反射”，它无法显式描绘未来轨迹
            return physical_action, None, None
            
        return physical_action

# ==============================================================================
# 单体集成测试模块 (自带虚拟大模型锻造器)
# ==============================================================================
if __name__ == "__main__":
    import time
    from stable_baselines3 import SAC
    import gymnasium as gym
    from gymnasium import spaces
    
    print("================================================================")
    print("🟢 Pure RL (纯强化学习控制器) 集成与稳定性测试启动")
    print("================================================================")
    
    temp_model_path = "temp_test_sac_model.zip"
    print("🔧 正在本地极速锻造虚拟 SAC 测试模型 (这只需几秒钟)...")
    
    # 严格遵循 Gymnasium 规范的虚拟环境
    class StandardDummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        def reset(self, seed=None, options=None):
            return np.zeros(13, dtype=np.float32), {}
        def step(self, action):
            return np.zeros(13, dtype=np.float32), 0.0, False, False, {}
            
    # 实例化合规环境并生成空模型
    dummy_env = StandardDummyEnv()
    dummy_model = SAC("MlpPolicy", dummy_env, verbose=0, device="cpu")
    dummy_model.save(temp_model_path)
    print(f"✅ 虚拟模型锻造成功并保存至: {temp_model_path}\n")
    
    # -------------------------------------------------------------
    # 正式推理测试流程
    # -------------------------------------------------------------
    print("加载 RL 控制器 (强制分配至 CPU 计算节点)...")
    rl_controller = PureRLController(model_path=temp_model_path, algo='SAC')
    
    # 构造当前时刻的极度炎热物理状态 (28℃)
    obs_current_physical = np.array([
        0.0002, 0.0005, 0.002, 0.005, # 生物干重
        0.001, 28.0, 0.015,           # CO2, 温度, 湿度
        18.0, 24.0, 1.0,              # TVP边界设定 
        30.0, 0.0006, 0.008           # 室外气象
    ])
    
    print(f"\n[测试状态] 室内真实温度: 28.0 ℃。神经网络蓄势待发。")
    print("发送推理指令并启动亚毫秒级计时...")
    
    t_start = time.perf_counter() # 使用高精度计时器
    u_opt_phys = rl_controller.compute_action(obs_current_physical, macro_params={}, tvp_forecast={})
    solve_time = time.perf_counter() - t_start
    
    print(f"\n✅ 神经网络前向推理完成！(耗时: {solve_time*1000:.4f} ms !!!)")
    print("输出真实物理控制指令 (由于是初生未训练模型，指令结果接近随机):")
    print(f"  💡 育苗区光照 : {u_opt_phys[0]:.2f} W/m2")
    print(f"  💡 成株区光照 : {u_opt_phys[1]:.2f} W/m2")
    print(f"  💨 CO2 补气   : {u_opt_phys[2]:.6f} kg/m2/s")
    print(f"  💧 除湿运行   : {u_opt_phys[3]:.6f} kg/m2/s")
    print(f"  ❄️ 空调功率   : {u_opt_phys[4]:.2f} W/m2")
    print(f"  🌪️ 通风机     : {u_opt_phys[5]:.2f} m/s")
    
    print("\n💡 架构分析：")
    print("1. 我们彻底补齐了 RL 控制器在张量维度的边缘崩溃漏洞 (reshape)。")
    print("2. 无论在哪台服务器上训练的模型，通过强制 CPU 推理，它都能在")
    print("   算力孱弱的边缘网关设备上稳定跑到毫秒级的高频控制响应！")
    print("================================================================\n")
    
    # 清理临时文件
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)