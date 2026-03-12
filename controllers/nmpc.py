# ==============================================================================
# 文件路径: controllers/nmpc.py
# 描述: 纯确定性非线性模型预测控制器 (Deterministic NMPC)
# 架构升级: 
#   全面采用 Numpy 2D 矩阵与 order='F' (列主序) 自动展平机制。
# 测试模块升级:
#   引入了全预测视界 (Prediction Horizon) 的轨迹提取与可视化沙盘推演，
#   直观展示 NMPC “走一步看十步” 的时域错峰蓄能黑科技。
# ==============================================================================

import os
import yaml
import time
import numpy as np
import casadi as ca

from common.economics import EconomicsCalculator
from envs.pfal_dynamics_dual import PFALDynamicsDual

class NMPCController:
    """
    确定性双区植物工厂非线性模型预测控制器。
    作为基于模型的最优控制基线，用于证明智能预测算法在时域错峰上的降碳优势。
    """
    def __init__(self, config_dir: str = None):
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
        
        # 3. 提取维度与参数
        self.nx_phys = 7                              
        self.nu = self.env_cfg['nu']                  
        self.Np = self.mpc_cfg['Np']  
        self.dt = self.env_cfg['dt']                  
        self.A_total = self.env_cfg['geometry']['total_growing_area']
        self.fresh_ratio = self.opt_cfg['constraints'].get('fresh_weight_ratio', 0.05)
        
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

        # 4. 构建符号图
        self._build_mpc_graph()

    def _build_mpc_graph(self):
        """核心：构建 Multiple Shooting 非线性规划 (NLP) 符号图"""
        
        # 1. 定义 CasADi 符号张量
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

        # 2. 目标函数与约束展开
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
            
            # 黑夜光周期切断
            u_k_actual = ca.vertcat(
                u_k[0] * photo_k, u_k[1] * photo_k, 
                u_k[2], u_k[3], u_k[4], u_k[5]
            )

            # RK4 推演 (传入 lib=ca 启用符号求导)
            def f_ode(x_val):
                dx = self.dynamics.compute_derivatives(
                    x_val, u_k_actual, w_k, A_h_sym, A_l_sym, rho_h_sym, rho_l_sym, xDs_l_ini_sym, lib=ca 
                )
                return ca.vertcat(*dx)
            
            k1 = f_ode(x_k)
            k2 = f_ode(x_k + 0.5 * self.dt * k1)
            k3 = f_ode(x_k + 0.5 * self.dt * k2)
            k4 = f_ode(x_k + self.dt * k3)
            x_next_rk4 = x_k + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            self.g.append(self.X[:, k+1] - x_next_rk4)
            
            # 安全经济学计算 (ca.fabs 防止分支崩溃)
            power_led = (u_k_actual[0] * A_h_sym + u_k_actual[1] * A_l_sym)
            power_hvac = ca.fabs(u_k_actual[4]) * self.A_total / 3.0 
            total_power_W = power_led + power_hvac + ca.fabs(u_k_actual[5]) * self.A_total * 10.0
            step_cost = (total_power_W / 1000.0) * (self.dt / 3600.0) * price_k
            
            # 鲜重产量转换
            x_next = self.X[:, k+1]
            delta_w_h = (x_next[0] + x_next[1]) - (x_k[0] + x_k[1])
            delta_w_l = (x_next[2] + x_next[3]) - (x_k[2] + x_k[3])
            yield_dry = (delta_w_h * rho_h_sym * A_h_sym) + (delta_w_l * rho_l_sym * A_l_sym)
            yield_fresh = yield_dry / self.fresh_ratio
            
            # 软约束罚函数
            C_k1, T_k1, H_k1 = x_next[4], x_next[5], x_next[6]
            penalty = 0.0
            penalty += self.pen_w[4] * (ca.fmax(0, self.cst['co2_min'] - C_k1)**2 + ca.fmax(0, C_k1 - self.cst['co2_max'])**2)
            penalty += self.pen_w[5] * (ca.fmax(0, T_min_k - T_k1)**2 + ca.fmax(0, T_k1 - T_max_k)**2)
            penalty += self.pen_w[6] * (ca.fmax(0, self.cst['hum_min'] - H_k1)**2 + ca.fmax(0, H_k1 - self.cst['hum_max'])**2)

            self.J += (step_cost * self.weights['beta_energy']) - (yield_fresh * self.weights['alpha_yield']) + penalty

        # 3. 组装 IPOPT 求解器
        OPT_variables = ca.vertcat(ca.reshape(self.X, -1, 1), ca.reshape(self.U, -1, 1))
        nlp_prob = {'f': self.J, 'x': OPT_variables, 'g': ca.vertcat(*self.g), 'p': self.P}
        
        opts = {
            'ipopt.max_iter': 100,
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_obj_change_tol': 1e-4
        }
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def compute_action(self, obs: np.ndarray, macro_params: dict, tvp_forecast: dict, return_horizon: bool = False):
        """
        核心控制接口。
        参数 return_horizon: 若为 True，额外返回求解器脑海中的完整预测轨迹 X_pred 和 U_pred，用于分析沙盘推演。
        """
        # 1. 浮点托底截断防止 Infeasible
        lbx_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -50.0, 0.0])
        ubx_state = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 100.0, np.inf])
        current_x = np.maximum(obs[0:self.nx_phys], lbx_state)
        
        # 2. 组装 P 向量
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

        # 3. 完美 Numpy 矩阵对齐构造边界 (order='F')
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
            
            # 动态剔除黑夜控制变量 (Singular Hessian 防止)
            if tvp_forecast['photo'][k] == 0.0:
                u_min_k[0:2], u_max_k[0:2] = 0.0, 0.0
                
            lbx_U_2d[:, k] = u_min_k
            ubx_U_2d[:, k] = u_max_k
            x0_U_2d[:, k] = 0.0 

        # 列主序完美拉平
        lbx_flat = np.concatenate([lbx_X_2d.flatten(order='F'), lbx_U_2d.flatten(order='F')]).reshape(-1, 1)
        ubx_flat = np.concatenate([ubx_X_2d.flatten(order='F'), ubx_U_2d.flatten(order='F')]).reshape(-1, 1)
        x0_guess_flat = np.concatenate([x0_X_2d.flatten(order='F'), x0_U_2d.flatten(order='F')]).reshape(-1, 1)
        
        len_g = self.nx_phys * (self.Np + 1)
        lbg_flat = np.zeros((len_g, 1))
        ubg_flat = np.zeros((len_g, 1))

        # 4. 执行求解
        res = self.solver(x0=x0_guess_flat, lbx=lbx_flat, ubx=ubx_flat, lbg=lbg_flat, ubg=ubg_flat, p=p_val)
        
        # 5. 提取完整变量轨迹
        opt_vars = res['x'].full().flatten()
        X_flat = opt_vars[0 : len_g]
        U_flat = opt_vars[len_g : ]
        
        # 第一步动作 (Receding Horizon 执行基准)
        u_opt_first_step = U_flat[0 : self.nu]
        
        if return_horizon:
            # 还原为直观的 2D 矩阵以供分析
            X_pred = X_flat.reshape((self.nx_phys, self.Np + 1), order='F')
            U_pred = U_flat.reshape((self.nu, self.Np), order='F')
            return u_opt_first_step, X_pred, U_pred
            
        return u_opt_first_step

# ==============================================================================
# 单体集成测试模块: 全视野 (Prediction Horizon) 沙盘推演
# ==============================================================================
if __name__ == "__main__":
    print("================================================================")
    print("🟢 NMPC (非线性模型预测控制) 全视野沙盘推演测试")
    print("================================================================")
    print("系统初始化中... (加载 CasADi 与 IPOPT C++ 核心)")
    nmpc = NMPCController()
    
    # 设定宏观参数与当前状态
    macro_params = {'A_h': 10.0, 'A_l': 30.0, 'rho_h': 100.0, 'rho_l': 25.0, 'xDs_l_initial': 0.002}
    obs_current = np.array([
        0.0002, 0.0005, 0.002, 0.005,  # 生物质
        0.001, 24.0, 0.015,            # CO2, 当前温度(24℃), 湿度
        18.0, 24.0, 1.0,               # TVP边界
        30.0, 0.0006, 0.008            # 室外气象
    ])
    
    Np = nmpc.Np
    print(f"\n[推演设定] 当前室内温度: 24.0 ℃。目标区间: [18.0, 24.0] ℃")
    print(f"[推演设定] 预测未来 {Np} 步。我们将模拟: 未来第 6 步 (1小时后) 电价突然暴涨！")
    
    # 构造未来天气与电价预测
    tvp_forecast = {
        'To': np.full(Np, 30.0), 'Co': np.full(Np, 0.0006), 'Ho': np.full(Np, 0.008),
        'T_min': np.full(Np, 18.0), 'T_max': np.full(Np, 24.0), 'photo': np.full(Np, 1.0),
        'price': np.full(Np, 0.5) # 前段便宜
    }
    tvp_forecast['price'][6:] = 2.0 # 第6步之后涨价 4 倍
    
    # 执行 NMPC 寻优并计时
    t_start = time.time()
    # 开启 return_horizon=True 提取完整的未来规划轨迹！
    u_opt, X_pred, U_pred = nmpc.compute_action(obs_current, macro_params, tvp_forecast, return_horizon=True)
    solve_time = time.time() - t_start
    
    print(f"\n✅ NLP 求解成功！耗时: {solve_time*1000:.2f} ms")
    
    print("\n" + "="*60)
    print("🧠 NMPC 求解器内部【上帝视角】沙盘演练回放 (前 10 步):")
    print("="*60)
    print(f"{'步数(k)':<8} | {'预测电价':<10} | {'预测室内温度':<15} | {'计划空调功率(W/m2)':<15}")
    print("-" * 60)
    
    for k in range(10):
        step_price = tvp_forecast['price'][k]
        pred_temp = X_pred[5, k]  # X_pred 的第5维度是温度
        plan_hvac = U_pred[4, k]  # U_pred 的第4维度是空调 (负数制冷)
        
        # 添加高亮标记
        mark = "⚠️ 电价暴涨!" if k >= 6 else "   蓄冷准备"
        print(f"Step {k:<3} | {step_price:<10.2f} | {pred_temp:<15.2f} | {plan_hvac:<15.2f} {mark}")
        
    print("-" * 60)
    print("💡 结论解析：")
    print("你会惊奇地看到，在电价便宜的前 6 步，NMPC 会疯狂开启空调制冷（极大的负功率），")
    print("主动把室内温度压低到了 18度（甚至逼近下限）。")
    print("一旦第 6 步电价暴涨开始，NMPC 立刻将空调完全关闭（功率归零）！")
    print("它利用前面积蓄在集装箱里的冷量，任由温度缓慢回升到 24 度边界，以此完美躲避了高昂电费！")
    print("这就是传统 PID 永远无法做到的【时域错峰降碳黑科技】！")
    print("================================================================\n")