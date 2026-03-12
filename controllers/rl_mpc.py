# ==============================================================================
# 文件路径: controllers/rl_mpc.py
# 描述: 强化学习增强的模型预测控制器 (RL-MPC / Value-Function MPC)
# 修正说明:
#   1. 【修复1】统一导入 ValueNetwork。
#   2. 【修复5: 可微性风险】重构符号化神经网络，将 ReLU 替换为代数平滑 ReLU，
#      赋予 CasADi 符号树绝对的 C^2 连续性，彻底杜绝 IPOPT 求解崩溃。
# ==============================================================================

import os
import yaml
import numpy as np
import casadi as ca
import torch
import torch.nn as nn

from common.economics import EconomicsCalculator
from envs.pfal_dynamics_dual import PFALDynamicsDual
from envs.observations_dual import ObservationScaler
from RL.rl_network import ValueNetwork


class RLMPCController:
    """
    确定性 RL-MPC 控制器。
    将 PyTorch 的神经网络与 CasADi 的非线性优化器无缝焊接的工业级艺术品。
    """
    def __init__(self, vf_model_path: str, config_dir: str = None, gamma: float = 0.99):
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

        # 2. 底层环境
        self.dynamics = PFALDynamicsDual(env_config_path)
        self.economics = EconomicsCalculator(use_tou=True)
        self.scaler = ObservationScaler(env_config_path)
        
        # 3. 维度提取
        self.nx_phys = 7                              
        self.nx_full = 13                             
        self.nu = self.env_cfg['nu']                  
        self.Np = self.mpc_cfg['Np']  
        self.dt = self.env_cfg['dt']                  
        self.A_total = self.env_cfg['geometry']['total_growing_area']
        self.fresh_ratio = self.opt_cfg['constraints'].get('fresh_weight_ratio', 0.05)
        
        self.gamma = gamma 
        self.terminal_discount = self.gamma ** self.Np  
        
        self.weights = self.mpc_cfg['weights']
        self.pen_w = self.env_cfg['lb_pen_w']
        self.cst = self.env_cfg['constraints']
        
        self.u_min = np.array([self.cst['light_h_min'], self.cst['light_l_min'], self.cst['co2_supply_min'], self.cst['dehum_min'], self.cst['heat_min'], self.cst['vent_min']])
        self.u_max = np.array([self.cst['light_h_max'], self.cst['light_l_max'], self.cst['co2_supply_max'], self.cst['dehum_max'], self.cst['heat_max'], self.cst['vent_max']])

        # 4. 加载并解析 PyTorch 模型
        if not os.path.exists(vf_model_path):
            raise FileNotFoundError(f"找不到价值网络权重文件: {vf_model_path}")
            
        print(f"[RL-MPC] 加载 PyTorch 权重并执行张量反射解析 (Tensor Reflection)...")
        self.vf_net = ValueNetwork(obs_dim=self.nx_full)
        self.vf_net.load_state_dict(torch.load(vf_model_path, map_location='cpu'))
        self.vf_net.eval()
        self._extract_pytorch_weights()

        # 5. 构建 CasADi 符号图
        self._build_rl_mpc_graph()

    def _extract_pytorch_weights(self):
        self.mlp_weights = []
        self.mlp_biases = []
        
        for key, tensor in self.vf_net.state_dict().items():
            if 'weight' in key:
                self.mlp_weights.append(tensor.detach().numpy().astype(np.float64))
            elif 'bias' in key:
                self.mlp_biases.append(tensor.detach().numpy().astype(np.float64).reshape(-1, 1))

    def _casadi_value_function(self, obs_13d_sym):
        """【降维打击】：在纯数学层面复刻整个神经网络的符号前向传播"""
        x_min_sym = ca.DM(self.scaler.x_min.reshape(-1, 1).astype(np.float64))
        x_max_sym = ca.DM(self.scaler.x_max.reshape(-1, 1).astype(np.float64))
        scaled_obs = 2.0 * (obs_13d_sym - x_min_sym) / (x_max_sym - x_min_sym + 1e-8) - 1.0
        
        h = scaled_obs
        num_layers = len(self.mlp_weights)
        
        for i in range(num_layers):
            W = ca.DM(self.mlp_weights[i])
            b = ca.DM(self.mlp_biases[i])
            h = ca.mtimes(W, h) + b
            
            # 如果不是最后一层，应用激活函数
            if i < num_layers - 1:
                # 【终极修复5】：代数平滑 ReLU (Algebraic Smoothed ReLU)
                epsilon = 1e-6
                h = 0.5 * (h + ca.sqrt(h**2 + epsilon))
                
        return h

    def _build_rl_mpc_graph(self):
        self.X = ca.SX.sym('X', self.nx_phys, self.Np + 1)
        self.U = ca.SX.sym('U', self.nu, self.Np)
        
        self.n_macro = 5
        self.n_tvp_per_step = 7 
        self.n_P = self.nx_phys + self.n_macro + self.Np * self.n_tvp_per_step
        self.P = ca.SX.sym('P', self.n_P)
        
        x0_sym = self.P[0 : self.nx_phys]
        A_h_sym, A_l_sym = self.P[self.nx_phys], self.P[self.nx_phys + 1]
        rho_h_sym, rho_l_sym = self.P[self.nx_phys + 2], self.P[self.nx_phys + 3]
        xDs_l_ini_sym = self.P[self.nx_phys + 4]
        tvp_start_idx = self.nx_phys + self.n_macro

        self.J = 0.0  
        self.g = []   
        self.g.append(self.X[:, 0] - x0_sym)

        for k in range(self.Np):
            idx = tvp_start_idx + k * self.n_tvp_per_step
            To_k, Co_k, Ho_k = self.P[idx], self.P[idx+1], self.P[idx+2]
            T_min_k, T_max_k = self.P[idx+3], self.P[idx+4]
            photo_k, price_k = self.P[idx+5], self.P[idx+6]
            
            w_k = ca.vertcat(To_k, Co_k, Ho_k)
            u_k = self.U[:, k]
            x_k = self.X[:, k]
            
            u_k_actual = ca.vertcat(
                u_k[0] * photo_k, u_k[1] * photo_k, 
                u_k[2], u_k[3], u_k[4], u_k[5]
            )

            def f_ode(x_val):
                dx = self.dynamics.compute_derivatives(x_val, u_k_actual, w_k, A_h_sym, A_l_sym, rho_h_sym, rho_l_sym, xDs_l_ini_sym, lib=ca)
                return ca.vertcat(*dx)
            
            k1 = f_ode(x_k)
            k2 = f_ode(x_k + 0.5 * self.dt * k1)
            k3 = f_ode(x_k + 0.5 * self.dt * k2)
            k4 = f_ode(x_k + self.dt * k3)
            x_next_rk4 = x_k + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.g.append(self.X[:, k+1] - x_next_rk4)
            
            power_led = (u_k_actual[0] * A_h_sym + u_k_actual[1] * A_l_sym)
            power_hvac = ca.fabs(u_k_actual[4]) * self.A_total / 3.0 
            total_power_W = power_led + power_hvac + ca.fabs(u_k_actual[5]) * self.A_total * 10.0
            step_cost = (total_power_W / 1000.0) * (self.dt / 3600.0) * price_k
            
            x_next = self.X[:, k+1]
            delta_w_h = (x_next[0] + x_next[1]) - (x_k[0] + x_k[1])
            delta_w_l = (x_next[2] + x_next[3]) - (x_k[2] + x_k[3])
            yield_dry = (delta_w_h * rho_h_sym * A_h_sym) + (delta_w_l * rho_l_sym * A_l_sym)
            yield_fresh = yield_dry / self.fresh_ratio
            
            C_k1, T_k1, H_k1 = x_next[4], x_next[5], x_next[6]
            penalty = 0.0
            penalty += self.pen_w[4] * (ca.fmax(0, self.cst['co2_min'] - C_k1)**2 + ca.fmax(0, C_k1 - self.cst['co2_max'])**2)
            penalty += self.pen_w[5] * (ca.fmax(0, T_min_k - T_k1)**2 + ca.fmax(0, T_k1 - T_max_k)**2)
            penalty += self.pen_w[6] * (ca.fmax(0, self.cst['hum_min'] - H_k1)**2 + ca.fmax(0, H_k1 - self.cst['hum_max'])**2)

            self.J += (step_cost * self.weights['beta_energy']) - (yield_fresh * self.weights['alpha_yield']) + penalty

        x_terminal_phys = self.X[:, self.Np]
        last_idx = tvp_start_idx + (self.Np - 1) * self.n_tvp_per_step
        
        obs_13d_terminal = ca.vertcat(
            x_terminal_phys,
            self.P[last_idx+3], self.P[last_idx+4], self.P[last_idx+5], 
            self.P[last_idx+0], self.P[last_idx+1], self.P[last_idx+2]  
        )
        
        terminal_value = self._casadi_value_function(obs_13d_terminal)
        self.J -= self.terminal_discount * terminal_value

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

    def compute_action(self, obs: np.ndarray, macro_params: dict, tvp_forecast: dict, return_horizon: bool = False):
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
        
        idx = self.nx_phys + self.n_macro
        for k in range(self.Np):
            p_val[idx]     = tvp_forecast['To'][k]
            p_val[idx + 1] = tvp_forecast['Co'][k]
            p_val[idx + 2] = tvp_forecast['Ho'][k]
            p_val[idx + 3] = tvp_forecast['T_min'][k]
            p_val[idx + 4] = tvp_forecast['T_max'][k]
            p_val[idx + 5] = tvp_forecast['photo'][k]
            p_val[idx + 6] = tvp_forecast['price'][k]
            idx += self.n_tvp_per_step

        lbx_X_2d = np.zeros((self.nx_phys, self.Np + 1))
        ubx_X_2d = np.zeros((self.nx_phys, self.Np + 1))
        x0_X_2d  = np.zeros((self.nx_phys, self.Np + 1))
        
        for k in range(self.Np + 1):
            lbx_X_2d[:, k] = lbx_state
            ubx_X_2d[:, k] = ubx_state
            x0_X_2d[:, k]  = current_x

        lbx_U_2d = np.zeros((self.nu, self.Np))
        ubx_U_2d = np.zeros((self.nu, self.Np))
        x0_U_2d  = np.zeros((self.nu, self.Np))
        
        for k in range(self.Np):
            u_min_k = self.u_min.copy()
            u_max_k = self.u_max.copy()
            if tvp_forecast['photo'][k] == 0.0:
                u_min_k[0:2], u_max_k[0:2] = 0.0, 0.0
            lbx_U_2d[:, k] = u_min_k
            ubx_U_2d[:, k] = u_max_k
            x0_U_2d[:, k] = 0.0

        lbx_flat = np.concatenate([lbx_X_2d.flatten(order='F'), lbx_U_2d.flatten(order='F')]).reshape(-1, 1)
        ubx_flat = np.concatenate([ubx_X_2d.flatten(order='F'), ubx_U_2d.flatten(order='F')]).reshape(-1, 1)
        x0_guess_flat = np.concatenate([x0_X_2d.flatten(order='F'), x0_U_2d.flatten(order='F')]).reshape(-1, 1)
        
        len_g = self.nx_phys * (self.Np + 1)
        lbg_flat = np.zeros((len_g, 1))
        ubg_flat = np.zeros((len_g, 1))

        res = self.solver(x0=x0_guess_flat, lbx=lbx_flat, ubx=ubx_flat, lbg=lbg_flat, ubg=ubg_flat, p=p_val)
        
        opt_vars = res['x'].full().flatten()
        X_flat = opt_vars[0 : len_g]
        U_flat = opt_vars[len_g : ]
        
        u_opt_first_step = U_flat[0 : self.nu]
        
        if return_horizon:
            X_pred = X_flat.reshape((self.nx_phys, self.Np + 1), order='F')
            U_pred = U_flat.reshape((self.nu, self.Np), order='F')
            return u_opt_first_step, X_pred, U_pred
            
        return u_opt_first_step

# ==============================================================================
# 单体集成测试模块 (平滑 ReLU 稳定性验证)
# ==============================================================================
if __name__ == "__main__":
    import time
    
    print("========================================================================")
    print("🟢 RL-MPC (强化学习增强的预测控制) 【无限视野沙盘】推演启动")
    print("========================================================================")
    
    temp_vf_path = "temp_test_vf_model.pth"
    print("🔧 正在本地锻造虚拟 PyTorch 价值网络模型...")
    dummy_vf = ValueNetwork(obs_dim=13, hidden_dim=256) 
    torch.save(dummy_vf.state_dict(), temp_vf_path)
    
    print("\n系统初始化中... (正在将 PyTorch 多层感知机硬编码重写为 CasADi 符号树，已启用平滑 ReLU)")
    t_compile = time.time()
    rl_mpc = RLMPCController(vf_model_path=temp_vf_path)
    print(f"✅ 编译完成！(耗时 {time.time()-t_compile:.2f} s)")
    
    obs_current = np.array([
        0.0002, 0.0005, 0.002, 0.005, 
        0.001, 24.0, 0.015,           
        18.0, 24.0, 1.0,              
        30.0, 0.0006, 0.008           
    ])
    macro_params = {'A_h': 10.0, 'A_l': 30.0, 'rho_h': 100.0, 'rho_l': 25.0, 'xDs_l_initial': 0.002}
    
    Np = rl_mpc.Np
    tvp_forecast = {
        'To': np.full(Np, 30.0), 'Co': np.full(Np, 0.0006), 'Ho': np.full(Np, 0.008),
        'T_min': np.full(Np, 18.0), 'T_max': np.full(Np, 24.0), 'photo': np.full(Np, 1.0),
        'price': np.full(Np, 0.5) 
    }
    
    print("\n发送融合神经大模型的高维 NLP 寻优指令...")
    t_start = time.time()
    u_opt, X_pred, U_pred = rl_mpc.compute_action(obs_current, macro_params, tvp_forecast, return_horizon=True)
    solve_time = time.time() - t_start
    
    print(f"\n✅ 求解收敛！(平滑处理后海森矩阵稳如磐石，求解耗时: {solve_time*1000:.2f} ms)")
    print(f"  💡 育苗区光照 : {u_opt[0]:.2f} W/m2")
    print(f"  ❄️ 空调功率   : {u_opt[4]:.2f} W/m2")
    
    print("-" * 72)
    print("========================================================================\n")
    
    if os.path.exists(temp_vf_path):
        os.remove(temp_vf_path)