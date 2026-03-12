# ==============================================================================
# 文件路径: envs/pfal_env_dual.py
# 描述: 工业级双区植物工厂强化学习/经典控制仿真环境 (OpenAI Gym 接口)
# 终极修复: 
#   1. 引入干鲜比 (Dry/Fresh Ratio) 转换，彻底拉平生物学与经济学的量纲。
#   2. 引入真实的外部 CSV 气象数据接口与鲁棒的循环填充/降级回退机制。
#   3. 【修复6: 量纲风险防御】在奖励结算环节截断夜间呼吸作用导致的干重负增量，消除 RL 训练方差。

# 特性: 
#   支持通过 rl_mode 开关，无缝切换 神经网络(归一化) 和 MPC/PID(纯物理量) 控制。
#   内置双时钟周期重置 (Dual-Clock Reset)，完美模拟连续生产的流水线流转。
# ==============================================================================

import os
import yaml
import math
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import warnings

from common.area_solver import AreaSolver
from common.economics import EconomicsCalculator
from envs.pfal_dynamics_dual import PFALDynamicsDual
from envs.observations_dual import ObservationScaler 

class PFALEnvDual(gym.Env):
    """
    双区植物工厂仿真环境。
    支持 rl_mode 开关，无缝适配强化学习（归一化空间）与 MPC/PID（物理真实空间）。
    """
    def __init__(self, config_dir: str = None, rl_mode: bool = True):
        super(PFALEnvDual, self).__init__()
        
        self.rl_mode = rl_mode # 控制算法旁路开关
        
        # 1. 自动寻址与配置文件加载
        base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
        self.mpc_config_path = os.path.join(base_dir, "configs", "models", "mpc_dual.yml")
        self.opt_config_path = os.path.join(base_dir, "configs", "optimizers.yml")
        
        with open(self.env_config_path, 'r', encoding='utf-8') as f:
            self.env_cfg = yaml.safe_load(f)
        with open(self.mpc_config_path, 'r', encoding='utf-8') as f:
            self.mpc_cfg = yaml.safe_load(f)['PFAL_Dual']
        with open(self.opt_config_path, 'r', encoding='utf-8') as f:
            self.opt_cfg = yaml.safe_load(f)

        # 2. 实例化底层引擎
        self.dynamics = PFALDynamicsDual(self.env_config_path)
        self.area_solver = AreaSolver()
        self.economics = EconomicsCalculator(use_tou=True)
        self.scaler = ObservationScaler(self.env_config_path) 
        
        # 3. 时间步长与仿真周期规划
        self.nx = self.env_cfg['nx']          
        self.nu = self.env_cfg['nu']          
        self.dt = self.env_cfg['dt']          
        self.n_days = self.env_cfg['n_days']
        # 计算最大仿真步数 (如 30天 * 24小时 * 6个步长/小时)
        self.max_steps = int((self.n_days * 24 * 3600) / self.dt)
        
        # 提取运筹学常数与环境硬边界
        self.fresh_weight_ratio = self.opt_cfg['constraints'].get('fresh_weight_ratio', 0.05)
        self.cst = self.env_cfg['constraints'] 
        
        # 4. 动态构建 Gym Action / Observation Space
        if self.rl_mode:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32)
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(self.nx,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=self.scaler.u_min, high=self.scaler.u_max, dtype=np.float32)
            self.observation_space = spaces.Box(low=self.scaler.x_min, high=self.scaler.x_max, dtype=np.float32)
        
        # 5. 评价体系提取 (保证 RL 和 MPC 的目标函数绝对一致)
        self.weights = self.mpc_cfg['weights']
        self.pen_w = np.array(self.env_cfg['lb_pen_w']) 
        
        # 6. 加载真实外部气象数据 (或回退到模拟数据)
        self.weather_file_path = self.env_cfg.get('weather_file', None)
        self._load_weather_data()
        
        self.state = np.zeros(self.nx)
        self.current_step = 0

    def _load_weather_data(self):
        """
        加载外部气象数据接口。
        优先从 yml 配置的 CSV 文件读取真实的 To(温度), Co(CO2), Ho(湿度)。
        若文件不存在或读取失败，鲁棒降级到正弦波模拟。
        """
        required_len = self.max_steps + 10 # 多备 10 步防越界
        
        # 1. 尝试读取真实 CSV 数据
        if self.weather_file_path and os.path.exists(self.weather_file_path):
            print(f"[PFALEnvDual] 正在加载真实气象历史数据: {self.weather_file_path}")
            try:
                df = pd.read_csv(self.weather_file_path)
                
                # 假设真实的 CSV 中有这三列。可根据实际数据源(如 EnergyPlus) 的表头在此修改映射
                To_data = df['Temp_out'].values
                Co_data = df['CO2_out'].values
                Ho_data = df['Hum_out'].values
                
                # 鲁棒性处理：如果提供的数据长度短于仿真周期，自动进行循环拼接 (Tile)
                if len(To_data) < required_len:
                    repeats = math.ceil(required_len / len(To_data))
                    To_data = np.tile(To_data, repeats)
                    Co_data = np.tile(Co_data, repeats)
                    Ho_data = np.tile(Ho_data, repeats)
                    
                # 截取所需长度并堆叠
                self.weather_series = np.column_stack((
                    To_data[:required_len],
                    Co_data[:required_len],
                    Ho_data[:required_len]
                ))
                print(f"[PFALEnvDual] 气象数据加载成功！(共截取 {required_len} 步)")
                return # 成功后直接返回，跳过模拟生成
                
            except Exception as e:
                warnings.warn(f"[PFALEnvDual 警告] 真实气象文件解析失败: {e}。将启用模拟气象底座！")

        else:
            print("[PFALEnvDual] 未配置真实气象文件路径，系统启用标准的正弦波气象底座。")

        # 2. 回退机制 (Fallback): 模拟标准昼夜节律气象
        self.weather_series = np.zeros((required_len, 3))
        for i in range(len(self.weather_series)):
            t_hour = (i * self.dt) / 3600.0
            # To: 昼夜温差 15~25度 的标准正弦波动
            self.weather_series[i, 0] = 20.0 + 5.0 * math.sin(2 * math.pi * t_hour / 24.0) 
            # Co: 恒定室外大气 CO2 浓度 (~400 ppm 的质量浓度估算)
            self.weather_series[i, 1] = 0.0006  
            # Ho: 恒定室外绝对湿度
            self.weather_series[i, 2] = 0.008   

    def reset(self, seed=None, options=None):
        """环境重置，接收宏观 BO(或 GA) 参数下发，初始化双区时钟"""
        super().reset(seed=seed)
        opts = options or {}
        
        # 1. 接收宏观排程下发的指令 (若无则读默认配置)
        self.T_h_days = opts.get('t_h', self.env_cfg['scheduling']['t_h_days'])
        self.T_l_days = opts.get('t_l', self.env_cfg['scheduling']['t_l_days'])
        self.rho_h = opts.get('rho_h', self.env_cfg['scheduling']['rho_h'])
        self.rho_l = opts.get('rho_l', self.env_cfg['scheduling']['rho_l'])
        self.photoperiod = opts.get('photoperiod', self.env_cfg['scheduling']['photoperiod_hours'])
        
        # 2. 解算真实物理占用面积
        area_res = self.area_solver.solve(self.T_h_days, self.T_l_days, self.rho_h, self.rho_l)
        if not area_res['is_feasible']:
            # 给定合理的安全面积默认值，防止极端宏观参数导致仿真崩溃
            self.A_h, self.A_l = 10.0, 30.0 
        else:
            self.A_h, self.A_l = area_res['A_h'], area_res['A_l']
        
        # 3. 初始化双区物理状态 (前四项为单株质量)
        x0 = np.array(self.env_cfg['x0'])
        self.state = np.copy(x0)
        self.init_xDn_h, self.init_xDs_h = x0[0], x0[1]
        
        # 4. 初始化双区移栽流水线时钟 (秒)
        self.time_h_sec, self.time_l_sec = 0.0, 0.0
        self.cached_xDs_transplant = x0[3] # 移栽苗记忆缓存
        self.xDs_l_initial = x0[3]
        
        self.current_step = 0
        self._update_tvp()
        
        # 5. 返回观测值 (旁路分发)
        if self.rl_mode:
            return self.scaler.scale_obs(self.state, add_noise=False), {}
        else:
            return self.state.copy(), {}

    def _update_tvp(self):
        """更新 13D 状态数组后 6 维的时间变参数 (边界与气象)"""
        t_hours = (self.current_step * self.dt) / 3600.0
        t_of_day = t_hours % 24.0
        # 极简光周期：计算当前时间是否在灯光开启区间内
        photo_flag = 1.0 if 6.0 <= t_of_day < (6.0 + self.photoperiod) else 0.0
        
        self.state[7] = self.cst['temp_min']
        self.state[8] = self.cst['temp_max']
        self.state[9] = photo_flag
        
        # 获取当前步的外部真实/模拟气象数据
        weather = self.weather_series[self.current_step]
        self.state[10], self.state[11], self.state[12] = weather[0], weather[1], weather[2]

    def step(self, action):
        """核心单步仿真执行"""
        # 1. 动作映射与硬约束保护
        if self.rl_mode:
            u_unscaled = self.scaler.unscale_action(action)
        else:
            u_unscaled = np.clip(action, self.scaler.u_min, self.scaler.u_max)
            
        # 黑夜期间硬性切断一切补光
        if self.state[9] == 0.0:
            u_unscaled[0], u_unscaled[1] = 0.0, 0.0

        x_phys = self.state[0:7].copy()
        w_dist = self.state[10:13].copy()
        
        # 2. 高精度 RK4 数值积分求解
        def f_ode(x):
            dx = self.dynamics.compute_derivatives(
                x, u_unscaled, w_dist, 
                self.A_h, self.A_l, self.rho_h, self.rho_l, 
                self.xDs_l_initial, lib=np
            )
            return np.array(dx)

        k1 = f_ode(x_phys)
        k2 = f_ode(x_phys + 0.5 * self.dt * k1)
        k3 = f_ode(x_phys + 0.5 * self.dt * k2)
        k4 = f_ode(x_phys + self.dt * k3)
        x_phys_next = x_phys + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # # 物理量底线截断 (极其重要，防止物理引擎因为极小数值爆炸)
        # x_phys_next = np.maximum(x_phys_next, 1e-8) 
        # self.state[0:7] = x_phys_next

        x_phys_next = x_phys + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 物理量底线截断 
        x_phys_next = np.maximum(x_phys_next, 1e-8) 
        
        # ===================================================================
        # 【引入 Peese 参考代码的熔断保护 (Status != 0 逻辑)】
        # 如果新算出来的状态出现了 NaN/Inf，或者温度超出了极限生存边界 (如 >45度)
        # ===================================================================
        is_invalid = np.any(np.isnan(x_phys_next)) or np.any(np.isinf(x_phys_next))
        is_crashed = x_phys_next[5] > 45.0 or x_phys_next[5] < 5.0  # 温度 T 在索引 5

        if is_invalid or is_crashed:
            # 拒绝更新物理状态，退回到安全状态 (避免网络被 NaN 污染)
            self.state[0:7] = np.nan_to_num(x_phys, nan=1e-8, posinf=45.0, neginf=5.0)
            
            # 给 AI 一记重拳，告诉它这种操作直接导致温室毁坏
            reward = -500.0
            terminated = True  # 强制提前结束该 Episode！
            truncated = False
            info = {"yield_gain_fresh_kg": 0.0, "step_cost": 0.0, "penalty": 500.0, "crash": True}
            
            # 必须缩放观测状态才能返回
            if self.rl_mode:
                return self.scaler.scale_obs(self.state, add_noise=False), reward, terminated, truncated, info
            else:
                return self.state.copy(), reward, terminated, truncated, info

        # 如果系统安全，则正常更新状态
        self.state[0:7] = x_phys_next
        

        # 3. 经济账本与产量核算
        t_hours = (self.current_step * self.dt) / 3600.0
        time_of_day = t_hours % 24.0
        cost_dict = self.economics.compute_step_cost(
            u_unscaled, self.A_h, self.A_l, self.area_solver.total_area, 
            self.dt, self.economics.get_electricity_price(time_of_day)
        )
        step_cost = cost_dict['total_cost']

        # 单步干重增量
        delta_W_h = (x_phys_next[0] + x_phys_next[1]) - (x_phys[0] + x_phys[1])
        delta_W_l = (x_phys_next[2] + x_phys_next[3]) - (x_phys[2] + x_phys[3])
        # 基于总面积和密度转换为全集装箱的干重产能
        yield_gain_dry_kg = (delta_W_h * self.rho_h * self.A_h) + (delta_W_l * self.rho_l * self.A_l)
        
        # ===================================================================
        # 【核心修复 6：量纲风险防御】
        # 植物在夜间（无光照时）只进行呼吸作用消耗非结构碳水化合物，导致 delta_W 为负。
        # 对于底层物理系统，干重下降是真实发生的；但对于经济与 RL 的 Reward 结算，
        # 如果不予截断，这会被除以 0.05 放大 20 倍，形成巨大的夜间惩罚。
        # 截断处理让 AI 知道：“夜间不长肉是被允许的自然规律”。
        # ===================================================================
        yield_gain_dry_reward = max(0.0, yield_gain_dry_kg)
        
        # 【量纲统一核心】将干重彻底转换为商品生菜鲜重
        yield_gain_fresh_kg = yield_gain_dry_reward / self.fresh_weight_ratio
        
        # 4. 生态越界惩罚评估
        penalty = 0.0
        C, T, H = x_phys_next[4], x_phys_next[5], x_phys_next[6]
        
        # 使用平方惩罚平滑过渡，消除硬编码，全部从配置读取
        if C < self.cst['co2_min']: penalty += self.pen_w[4] * (self.cst['co2_min'] - C)**2
        if C > self.cst['co2_max']: penalty += self.pen_w[4] * (C - self.cst['co2_max'])**2
        if T < self.state[7]:       penalty += self.pen_w[5] * (self.state[7] - T)**2
        if T > self.state[8]:       penalty += self.pen_w[5] * (T - self.state[8])**2
        if H < self.cst['hum_min']: penalty += self.pen_w[6] * (self.cst['hum_min'] - H)**2
        if H > self.cst['hum_max']: penalty += self.pen_w[6] * (H - self.cst['hum_max'])**2

        # 5. RL 综合奖励 = 鲜生菜售卖价值 - 能耗账单成本 - 控制失误惩罚
        reward = (yield_gain_fresh_kg * self.weights['alpha_yield']) - (step_cost * self.weights['beta_energy']) - penalty

        # 6. 【核心流水线管理】双时钟重置机制
        self.time_h_sec += self.dt
        self.time_l_sec += self.dt
        
        # 高密度区满期，移出并重置
        if self.time_h_sec >= (self.T_h_days * 24 * 3600):
            self.cached_xDs_transplant = self.state[1] 
            self.state[0], self.state[1] = self.init_xDn_h, self.init_xDs_h
            self.time_h_sec = 0.0
            
        # 低密度区满期，定植新苗并重置
        if self.time_l_sec >= (self.T_l_days * 24 * 3600):
            self.xDs_l_initial = self.cached_xDs_transplant
            # 按比例估算非结构干重分配
            ratio = self.init_xDn_h / (self.init_xDs_h + 1e-8)
            self.state[2] = self.xDs_l_initial * ratio
            self.state[3] = self.xDs_l_initial
            self.time_l_sec = 0.0

        # 7. 步进更新与结束判定
        self.current_step += 1
        self._update_tvp()
        
        terminated = False
        truncated = self.current_step >= self.max_steps
            
        info = {
            "yield_gain_fresh_kg": yield_gain_fresh_kg, # 记录为截断后的无损耗产能
            "step_cost": step_cost,
            "penalty": penalty
        }

        # 8. 旁路输出观测状态
        if self.rl_mode:
            return self.scaler.scale_obs(self.state, add_noise=True), float(reward), terminated, truncated, info
        else:
            return self.state.copy(), float(reward), terminated, truncated, info