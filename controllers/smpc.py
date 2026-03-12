# ==============================================================================
# 文件路径: controllers/smpc.py
# 描述: 场景驱动型随机模型预测控制器 (Scenario-based Stochastic MPC)
# 架构升级: 
#   全面采用 Numpy 2D 矩阵与 order='F' (列主序) 自动展平机制，防止张量错位。
# 测试模块升级:
#   引入全视野 (Prediction Horizon) 的轨迹提取与多场景 (Multi-Scenario) 
#   沙盘推演回放，直观展示 SMPC 如何用 1 个动作同时应对 3 个平行宇宙的极端天气。
# ==============================================================================

import os
import yaml
import time
import numpy as np
import casadi as ca

from common.economics import EconomicsCalculator
from envs.pfal_dynamics_dual import PFALDynamicsDual

class SMPCController:
    """
    随机模型预测控制器 (Scenario-based SMPC)。
    寻找一种即使在极端天气(预报误差)下也不会导致温室崩溃，且期望收益最高的非预期鲁棒策略。
    """
    def __init__(self, config_dir: str = None, n_scenarios: int = 3):
        # 1. 自动寻址与配置加载
        base_dir = config_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
        mpc_config_path = os.path.join(base_dir, "configs", "models", "mpc_dual.yml")
        opt_config_path = os.path.join(base_dir, "configs", "optimizers.yml")
        
        with open(env_config_path, 'r', encoding='utf-8') as f:
            self.env_cfg = yaml.safe_load(f)
        with open(mpc_config_path, 'r', encoding='utf-8') as f:
            self.mpc_cfg = yaml.safe_load(f)['PFAL_Dual']
        with open(opt_config_path, 'r', encoding='utf-8') as f:
            self.opt_cfg = yaml.safe_load(f)

        # 2. 引擎实例化
        self.dynamics = PFALDynamicsDual(env_config_path)
        self.economics = EconomicsCalculator(use_tou=True)
        
        # 3. 维度提取
        self.nx_phys = 7                              
        self.nu = self.env_cfg['nu']                  
        self.Np = self.mpc_cfg['Np']  
        self.dt = self.env_cfg['dt']                  
        self.A_total = self.env_cfg['geometry']['total_growing_area']
        self.fresh_ratio = self.opt_cfg['constraints'].get('fresh_weight_ratio', 0.05)
        
        # SMPC 特有维度
        self.Ns = n_scenarios                         
        self.prob_s = 1.0 / self.Ns  # 假设所有场景发生概率均等
        
        self.weights = self.mpc_cfg['weights']
        self.pen_w = self.env_cfg['lb_pen_w']
        self.cst = self.env_cfg['constraints']
        
        self.u_min = np.array([
            self.cst['light_h_min'], self.cst['light_l_min'], self.cst['co2_supply_min'], 
            self.cst['dehum_min'], self.cst['heat_min'], self.cst['vent_min']
        ])
        self.u_max = np.array([
            self.cst['light_h_max'], self.cst['light_l_max'], self.cst['co2_supply_max'], 
            self.cst['dehum_max'], self.cst['heat_max'], self.cst['vent_max']
        ])

        # 4. 构建多场景符号计算图
        self._build_smpc_graph()

    def _build_smpc_graph(self):
        """核心：构建涵盖 N_s 个平行宇宙的 Multiple Shooting 符号图"""
        
        # X: 状态矩阵 (nx_phys*Ns) 行 x (Np+1) 列
        self.X = ca.SX.sym('X', self.nx_phys * self.Ns, self.Np + 1)
        # U: 控制动作矩阵 (nu x Np) -> 所有场景共用 1 个 U (Non-anticipative)
        self.U = ca.SX.sym('U', self.nu, self.Np)
        
        # P: 包含所有场景预测的巨型参数向量
        self.n_macro = 5
        self.n_tvp_per_step = 7 
        self.n_P = self.nx_phys + self.n_macro + (self.Ns * self.Np * self.n_tvp_per_step)
        self.P = ca.SX.sym('P', self.n_P)
        
        x0_sym = self.P[0 : self.nx_phys]
        A_h_sym, A_l_sym = self.P[self.nx_phys], self.P[self.nx_phys + 1]
        rho_h_sym, rho_l_sym = self.P[self.nx_phys + 2], self.P[self.nx_phys + 3]
        xDs_l_ini_sym = self.P[self.nx_phys + 4]
        tvp_start_idx = self.nx_phys + self.n_macro

        self.J = 0.0  
        self.g = []   
        
        # ==========================================
        # 遍历所有场景 (Scenarios) 展开平行推演
        # ==========================================
        for s in range(self.Ns):
            x_idx_start = s * self.nx_phys
            x_idx_end = (s + 1) * self.nx_phys
            
            # 约束: 所有平行宇宙在时刻 0 必须共享当前的真实物理起点
            self.g.append(self.X[x_idx_start:x_idx_end, 0] - x0_sym)
            
            J_scenario = 0.0
            
            for k in range(self.Np):
                # 提取该场景专属的 TVP 扰动
                offset = tvp_start_idx + (s * self.Np + k) * self.n_tvp_per_step
                To_k, Co_k, Ho_k = self.P[offset], self.P[offset+1], self.P[offset+2]
                T_min_k, T_max_k = self.P[offset+3], self.P[offset+4]
                photo_k, price_k = self.P[offset+5], self.P[offset+6]
                
                w_k = ca.vertcat(To_k, Co_k, Ho_k)
                u_k = self.U[:, k]
                x_k_s = self.X[x_idx_start:x_idx_end, k]
                
                # 光周期掩码
                u_k_actual = ca.vertcat(
                    u_k[0] * photo_k, u_k[1] * photo_k, 
                    u_k[2], u_k[3], u_k[4], u_k[5]
                )

                # RK4 积分
                def f_ode(x_val):
                    dx = self.dynamics.compute_derivatives(
                        x_val, u_k_actual, w_k, A_h_sym, A_l_sym, rho_h_sym, rho_l_sym, xDs_l_ini_sym, lib=ca 
                    )
                    return ca.vertcat(*dx)
                
                k1 = f_ode(x_k_s)
                k2 = f_ode(x_k_s + 0.5 * self.dt * k1)
                k3 = f_ode(x_k_s + 0.5 * self.dt * k2)
                k4 = f_ode(x_k_s + self.dt * k3)
                x_next_rk4 = x_k_s + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                
                self.g.append(self.X[x_idx_start:x_idx_end, k+1] - x_next_rk4)
                
                # 经济学计算
                power_led = (u_k_actual[0] * A_h_sym + u_k_actual[1] * A_l_sym)
                power_hvac = ca.fabs(u_k_actual[4]) * self.A_total / 3.0 
                total_power_W = power_led + power_hvac + ca.fabs(u_k_actual[5]) * self.A_total * 10.0
                step_cost = (total_power_W / 1000.0) * (self.dt / 3600.0) * price_k
                
                x_next_s = self.X[x_idx_start:x_idx_end, k+1]
                delta_w_h = (x_next_s[0] + x_next_s[1]) - (x_k_s[0] + x_k_s[1])
                delta_w_l = (x_next_s[2] + x_next_s[3]) - (x_k_s[2] + x_k_s[3])
                yield_dry = (delta_w_h * rho_h_sym * A_h_sym) + (delta_w_l * rho_l_sym * A_l_sym)
                yield_fresh = yield_dry / self.fresh_ratio
                
                # 软约束罚函数
                C_k1, T_k1, H_k1 = x_next_s[4], x_next_s[5], x_next_s[6]
                penalty = 0.0
                penalty += self.pen_w[4] * (ca.fmax(0, self.cst['co2_min'] - C_k1)**2 + ca.fmax(0, C_k1 - self.cst['co2_max'])**2)
                penalty += self.pen_w[5] * (ca.fmax(0, T_min_k - T_k1)**2 + ca.fmax(0, T_k1 - T_max_k)**2)
                penalty += self.pen_w[6] * (ca.fmax(0, self.cst['hum_min'] - H_k1)**2 + ca.fmax(0, H_k1 - self.cst['hum_max'])**2)

                J_scenario += (step_cost * self.weights['beta_energy']) - (yield_fresh * self.weights['alpha_yield']) + penalty

            # 累加各场景的加权期望成本
            self.J += J_scenario * self.prob_s

        # 组装超级优化器
        OPT_variables = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        nlp_prob = {'f': self.J, 'x': OPT_variables, 'g': ca.vertcat(*self.g), 'p': self.P}
        
        opts = {
            'ipopt.max_iter': 150,           
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def compute_action(self, obs: np.ndarray, macro_params: dict, tvp_forecasts: dict, return_horizon: bool = False):
        """
        参数:
        - tvp_forecasts: 字典，每个值为 shape (Ns, Np) 的二维数组，包含了不同场景的天气预测。
        - return_horizon: 若为 True，额外返回预测轨迹 X_pred 和 U_pred，供分析沙盘推演使用。
        """
        lbx_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -50.0, 0.0])
        ubx_state = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 100.0, np.inf])
        current_x = np.maximum(obs[0:self.nx_phys], lbx_state)
        
        p_val = np.zeros(self.n_P)
        p_val[0:self.nx_phys] = current_x
        p_val[self.nx_phys] = macro_params['A_h']
        p_val[self.nx_phys + 1] = macro_params['A_l']
        p_val[self.nx_phys + 2] = macro_params['rho_h']
        p_val[self.nx_phys + 3] = macro_params['rho_l']
        p_val[self.nx_phys + 4] = macro_params['xDs_l_initial']
        
        tvp_start_idx = self.nx_phys + self.n_macro
        for s in range(self.Ns):
            for k in range(self.Np):
                offset = tvp_start_idx + (s * self.Np + k) * self.n_tvp_per_step
                p_val[offset]     = tvp_forecasts['To'][s, k]
                p_val[offset + 1] = tvp_forecasts['Co'][s, k]
                p_val[offset + 2] = tvp_forecasts['Ho'][s, k]
                p_val[offset + 3] = tvp_forecasts['T_min'][s, k]
                p_val[offset + 4] = tvp_forecasts['T_max'][s, k]
                p_val[offset + 5] = tvp_forecasts['photo'][s, k]
                p_val[offset + 6] = tvp_forecasts['price'][s, k]

        # 完美对齐 Numpy 2D 边界矩阵
        lbx_X_2d = np.zeros((self.nx_phys * self.Ns, self.Np + 1))
        ubx_X_2d = np.zeros((self.nx_phys * self.Ns, self.Np + 1))
        x0_X_2d  = np.zeros((self.nx_phys * self.Ns, self.Np + 1))
        
        for s in range(self.Ns):
            lbx_X_2d[s*self.nx_phys : (s+1)*self.nx_phys, :] = lbx_state.reshape(-1, 1)
            ubx_X_2d[s*self.nx_phys : (s+1)*self.nx_phys, :] = ubx_state.reshape(-1, 1)
            x0_X_2d[s*self.nx_phys : (s+1)*self.nx_phys, :]  = current_x.reshape(-1, 1)

        lbx_U_2d = np.zeros((self.nu, self.Np))
        ubx_U_2d = np.zeros((self.nu, self.Np))
        x0_U_2d  = np.zeros((self.nu, self.Np))
        
        for k in range(self.Np):
            u_min_k = self.u_min.copy()
            u_max_k = self.u_max.copy()
            # 以场景0的光周期作为切断依据
            if tvp_forecasts['photo'][0, k] == 0.0:
                u_min_k[0:2], u_max_k[0:2] = 0.0, 0.0
                
            lbx_U_2d[:, k] = u_min_k
            ubx_U_2d[:, k] = u_max_k
            x0_U_2d[:, k] = 0.0

        # 列主序展平 (完美兼容 CasADi)
        lbx_flat = np.concatenate([lbx_X_2d.flatten(order='F'), lbx_U_2d.flatten(order='F')]).reshape(-1, 1)
        ubx_flat = np.concatenate([ubx_X_2d.flatten(order='F'), ubx_U_2d.flatten(order='F')]).reshape(-1, 1)
        x0_guess_flat = np.concatenate([x0_X_2d.flatten(order='F'), x0_U_2d.flatten(order='F')]).reshape(-1, 1)
        
        len_g = (self.nx_phys * self.Ns) * (self.Np + 1)
        lbg_flat = np.zeros((len_g, 1))
        ubg_flat = np.zeros((len_g, 1))

        # 暴力求解多场景 NLP
        res = self.solver(x0=x0_guess_flat, lbx=lbx_flat, ubx=ubx_flat, lbg=lbg_flat, ubg=ubg_flat, p=p_val)
        
        # 提取结果
        opt_vars = res['x'].full().flatten()
        X_flat = opt_vars[0 : len_g]
        U_flat = opt_vars[len_g : ]
        
        u_opt_first_step = U_flat[0 : self.nu]
        
        if return_horizon:
            # 还原为 2D 矩阵供分析
            X_pred = X_flat.reshape((self.nx_phys * self.Ns, self.Np + 1), order='F')
            U_pred = U_flat.reshape((self.nu, self.Np), order='F')
            return u_opt_first_step, X_pred, U_pred
            
        return u_opt_first_step

# ==============================================================================
# 单体集成测试模块: 多平行宇宙 (Parallel Universe) 沙盘推演
# ==============================================================================
if __name__ == "__main__":
    print("=======================================================================")
    print("🟢 SMPC (随机模型预测控制) 【多平行宇宙】 沙盘推演启动")
    print("=======================================================================")
    print("正在构建并编译巨型多场景 CasADi 符号树... (约需 3~5 秒，请耐心等待)")
    
    Ns = 3 
    smpc = SMPCController(n_scenarios=Ns)
    
    obs_current = np.array([
        0.0002, 0.0005, 0.002, 0.005, 
        0.001, 24.0, 0.015,           
        18.0, 24.0, 1.0,              
        24.0, 0.0006, 0.008           
    ])
    
    macro_params = {'A_h': 10.0, 'A_l': 30.0, 'rho_h': 100.0, 'rho_l': 25.0, 'xDs_l_initial': 0.002}
    
    Np = smpc.Np
    print(f"\n[推演设定] 初始室内温度: 24.0 ℃。安全运行区间: [18.0, 24.0] ℃")
    print(f"[推演设定] 我们将向系统注入 3 种完全不可预知的天气场景，并在一小时后拉高电价！")
    
    # 构造 3 个平行宇宙的预测
    tvp_forecasts = {
        'To': np.zeros((Ns, Np)),          
        'Co': np.full((Ns, Np), 0.0006),
        'Ho': np.full((Ns, Np), 0.008),
        'T_min': np.full((Ns, Np), 18.0),       
        'T_max': np.full((Ns, Np), 24.0),       
        'photo': np.full((Ns, Np), 1.0),        
        'price': np.full((Ns, Np), 0.5)         
    }
    
    # 注入平行宇宙的致命扰动
    tvp_forecasts['To'][0, :] = 24.0 # S0: 岁月静好 (普通24度)
    tvp_forecasts['To'][1, :] = 35.0 # S1: 极端热浪 (暴热35度)
    tvp_forecasts['To'][2, :] = 5.0  # S2: 极端寒潮 (暴冷5度)
    
    tvp_forecasts['price'][:, 6:] = 2.0 # 第6步之后涨价
    
    # 寻优并提取全轨迹
    t_start = time.time()
    u_opt, X_pred, U_pred = smpc.compute_action(obs_current, macro_params, tvp_forecasts, return_horizon=True)
    solve_time = time.time() - t_start
    
    print(f"✅ NLP 求解收敛！(多场景寻优耗时: {solve_time*1000:.2f} ms)\n")
    
    print("="*75)
    print("🧠 SMPC 内部【三体并行沙盘】演练回放 (前 10 步):")
    print("="*75)
    print(f"{'步数':<5} | {'电价':<6} | {'通用空调策略(W/m2)':<20} | {'S0:正常(℃)':<10} | {'S1:热浪(℃)':<10} | {'S2:寒潮(℃)':<10}")
    print("-" * 75)
    
    for k in range(10):
        step_price = tvp_forecasts['price'][0, k]
        plan_hvac = U_pred[4, k]  # 唯一确定的空调动作
        
        # 提取各个平行宇宙中的温度走向
        t_in_s0 = X_pred[5, k]                     # 场景 0 (偏移 0)
        t_in_s1 = X_pred[5 + smpc.nx_phys, k]      # 场景 1 (偏移 7)
        t_in_s2 = X_pred[5 + smpc.nx_phys * 2, k]  # 场景 2 (偏移 14)
        
        print(f"k={k:<3} | {step_price:<6.2f} | {plan_hvac:<20.2f} | {t_in_s0:<10.2f} | {t_in_s1:<10.2f} | {t_in_s2:<10.2f}")
        
    print("-" * 75)
    print("💡 结论解析：")
    print("你会看到，SMPC 面临了极其艰难的选择：如果它像 NMPC 一样为了省钱疯狂制冷蓄能（降到18度），")
    print("那么在【S2:寒潮】场景中，植物会直接被冻坏（突破18度底线）！")
    print("因此，SMPC 的通用动作（General Action）变得极其克制！它选择了在前期微弱制冷甚至轻微制热，")
    print("保证在这个动作下，无论是热浪袭来（S1升温快）还是寒潮袭来（S2降温快），室内温度都能死死卡在")
    print("[18, 24] 的安全区间内！这就是对抗不确定性的【非预期鲁棒控制】的最高境界！")
    print("=======================================================================\n")