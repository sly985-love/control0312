# ==============================================================================
# 文件路径: envs/pfal_dynamics_dual.py
# 描述: 工业级双区植物工厂 7维生境动力学核心引擎 (ODE) - 终极量纲对齐与防作弊版
# 核心机制:
#   1. 以单株质量 (kg/plant) 为追踪目标，通过密度 (rho) 进行光资源抢夺与蒸腾放大。
#   2. 双核支持: Numpy (强化学习高速步进) / CasADi (MPC 符号求导)。
#   3. 内置严密的安全锁，防止 AI 利用极端热力学状态 (如憋死 CO2、饿死植物) 作弊。
# ==============================================================================

import yaml
import os

class PFALDynamicsDual:
    """
    状态空间 x (7D): [xDn_h, xDs_h, xDn_l, xDs_l, C, T, H] (前四项为 kg/plant)
    控制空间 u (6D): [uL_h, uL_l, uC, uD, uQ, uV]
    扰动空间 w (3D): [To, Co, Ho]
    """
    def __init__(self, env_config_path: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if env_config_path is None:
            env_config_path = os.path.join(base_dir, "configs", "envs", "PFAL_dual.yml")
            
        with open(env_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 1. 物理几何参数 (单位: m, m2, m3)
        geo = config['geometry']
        self.A_total = geo['total_growing_area']
        self.Volume = geo['volume']
        self.Surface = geo['surface_area']
        self.U_value = geo['u_value']
        
        # 2. 宏观生物学机制
        bio = config['biology']
        self.enable_shock = bio['enable_shock']
        self.shock_depth = bio.get('shock_depth', 0.8)          # 跌幅 80%
        self.recovery_rate = bio.get('recovery_rate', 1500.0)   # 单株干重恢复系数
        self.enable_shading = bio['enable_density_shading']
        self.k_ext = bio['base_k_extinction']
        
        # 3. Van Henten 体系核心生化常数 (严格遵照经典量纲)
        self.c_alpha = 0.68         # 光合转化率
        self.c_beta = 0.8           # 结构干重分配率
        self.c_bnd = 0.004          # 边界层电导
        self.c_car_1 = -1.32E-5     
        self.c_car_2 = 5.94E-4      
        self.c_car_3 = -2.64E-3     
        self.c_eps = 17E-9          
        self.c_Gamma = 7.32E-5      # CO2 补偿点
        self.c_lar_s = 75.0         # 比叶面积
        self.c_par = 1.0            
        self.c_Q10_Gamma = 2.0      
        self.c_Q10_gr = 1.6         
        self.c_Q10_resp = 2.0       
        self.c_rad_rf = 1.0         
        self.c_r_gr_max = 5E-6      # 最大比生长速率 (1/s)
        self.c_resp_s = 3.47E-7     # 维持呼吸
        self.c_resp_r = 1.16E-7     
        self.c_stm = 0.007          
        self.c_tau = 0.07           
        self.c_a_pl = 62.8          # 蒸腾衰减常数 (标定于 kg/m2)
        self.c_v_pl_ai = 3.6E-3     
        
        # 4. 热力学环境常数
        self.mw_water = 18.0        
        self.c_R = 8314.0           
        self.c_T_abs = 273.15       
        self.c_v_0 = 0.85; self.c_v_1 = 611.0; self.c_v_2 = 17.4; self.c_v_3 = 239.0
        
        self.c_cap_q = 30000.0      # 围护面积比热容 (J/m2/K)
        self.c_cap_q_v = 1290.0     # 空气容积比热容 (J/m3/K)
        self.c_lat_water = 2256.4   # 汽化潜热 (kJ/kg)
        self.c_led_eff = 0.52       # LED 光电效率

    def _saturation_humidity(self, T, lib):
        """饱和绝对湿度求解 (kg/m3)"""
        return ((self.c_v_1 * self.mw_water) / (self.c_R * (T + self.c_T_abs))) * \
               lib.exp((self.c_v_2 * T) / (T + self.c_v_3))

    def compute_derivatives(self, x, u, w, A_h, A_l, rho_h, rho_l, xDs_l_initial, lib):
        """
        常微分方程求解器。
        参数:
        - x: 状态 [xDn_h, xDs_h, xDn_l, xDs_l, C, T, H] (前四项为单株干重 kg/plant)
        - rho_h, rho_l: 宏观排程决策的种植密度 (plants/m2)
        """
        xDn_h, xDs_h, xDn_l, xDs_l, C, T, H = x[0], x[1], x[2], x[3], x[4], x[5], x[6]
        uL_h, uL_l, uC, uD, uQ, uV = u[0], u[1], u[2], u[3], u[4], u[5]
        To, Co, Ho = w[0], w[1], w[2]

        # ---------------------------------------------------------
        # 1. 共享环境 CO2 安全生理预计算
        # ---------------------------------------------------------

        # 【致敬参考代码的修复】：在带入指数运算前，强制把 T 和 C 锁死在热力学安全边界内
        # 防止 RL 输出极端动作导致 RK4 积分发散，引发 overflow 警告
        T_safe = lib.fmax(0.0, lib.fmin(T, 50.0))      # 温度强制锁定在 0~50 度之间
        C_safe = lib.fmax(0.0, lib.fmin(C, 0.005))     # CO2 浓度上限保护 (约相当于极限高浓度)
        H_safe = lib.fmax(0.0, lib.fmin(H, 0.05))      # 湿度上限保护

        Gamma = self.c_Gamma * (self.c_Q10_Gamma ** ((T - 20.0) / 10.0))
        # [安全锁 1]: CO2 补偿点截断，防止 AI 故意憋死植物导致负向公式反转溢出
        C_minus_Gamma = lib.fmax(C - Gamma, 0.0) 
        
        eps = self.c_eps * (C_minus_Gamma / (C + 2 * Gamma + 1e-8))
        sigma_car = self.c_car_1 * (T ** 2) + self.c_car_2 * T + self.c_car_3
        sigma_CO2 = 1.0 / (1.0 / self.c_bnd + 1.0 / self.c_stm + 1.0 / (sigma_car + 1e-8))

        # ---------------------------------------------------------
        # 2. 育苗区 (H区) 生理演化 - 核心: 引入密度竞争与资源平摊
        # ---------------------------------------------------------
        # (1) 光合作用: LAI 与光截获率
        LAI_h = rho_h * xDs_h * self.c_lar_s * (1 - self.c_tau)
        f_h = 1.0 - lib.exp(-self.k_ext * LAI_h) if self.enable_shading else 1.0
        
        # 每平方米总截获的光合产物 (kg_CO2 / m2 / s)
        phi_phot_max_m2_h = (eps * self.c_par * self.c_rad_rf * uL_h * self.c_led_eff * sigma_CO2 * C_minus_Gamma) / \
                            (eps * self.c_par * self.c_rad_rf * uL_h * self.c_led_eff + sigma_CO2 * C_minus_Gamma + 1e-8)
        phi_phot_m2_h = phi_phot_max_m2_h * f_h
        
        # [单株抢夺分配]: 密度越大，分摊到单株的光合量越少
        phi_phot_plant_h = phi_phot_m2_h / (rho_h + 1e-5)
        
        # (2) 呼吸与干重演变 (单株视角)
        r_gr_h = self.c_r_gr_max * (xDn_h / (xDs_h + xDn_h + 1e-8)) * (self.c_Q10_gr ** ((T - 20.0) / 10.0))
        phi_resp_plant_h = (self.c_resp_s * (1 - self.c_tau) + self.c_resp_r * self.c_tau) * xDs_h * (self.c_Q10_resp ** ((T - 25.0) / 10.0))
        
        phi_phot_c_plant_h = phi_phot_plant_h - (1.0 / self.c_alpha) * phi_resp_plant_h - ((1.0 - self.c_beta) / (self.c_alpha * self.c_beta)) * r_gr_h * xDs_h
        
        # (3) 蒸腾作用: [量纲修复] 必须还原为每平米干重 (xDs * rho) 计算蒸腾厚度
        phi_transp_m2_h = (1.0 - lib.exp(-self.c_a_pl * (xDs_h * rho_h))) * self.c_v_pl_ai * (self._saturation_humidity(T, lib) - H)
        phi_transp_plant_h = phi_transp_m2_h / (rho_h + 1e-5)
        
        # 单株导数
        xDn_h_dot = self.c_alpha * phi_phot_plant_h - r_gr_h * xDs_h - phi_resp_plant_h - ((1.0 - self.c_beta) / self.c_beta) * r_gr_h * xDs_h
        xDs_h_dot = r_gr_h * xDs_h

        # ---------------------------------------------------------
        # 3. 成株区 (L区) 生理演化 - 核心: 缓苗保护与光合竞争
        # ---------------------------------------------------------
        if self.enable_shock:
            # [安全锁 2]: 修复绝对值导致的“饥饿假恢复”漏洞。必须有真实正向增重才能恢复。
            mass_gain = lib.fmax(xDs_l - xDs_l_initial, 0.0) 
            shock_factor = 1.0 - self.shock_depth * lib.exp(-self.recovery_rate * mass_gain)
            shock_factor = lib.fmax(0.01, lib.fmin(1.0, shock_factor)) 
        else:
            shock_factor = 1.0
            
        # 缓苗直接抑制极限生长率
        r_gr_max_l = self.c_r_gr_max * shock_factor
        
        # (1) 光合竞争与抢夺
        LAI_l = rho_l * xDs_l * self.c_lar_s * (1 - self.c_tau)
        f_l = 1.0 - lib.exp(-self.k_ext * LAI_l) if self.enable_shading else 1.0
        
        phi_phot_max_m2_l = (eps * self.c_par * self.c_rad_rf * uL_l * self.c_led_eff * sigma_CO2 * C_minus_Gamma) / \
                            (eps * self.c_par * self.c_rad_rf * uL_l * self.c_led_eff + sigma_CO2 * C_minus_Gamma + 1e-8)
        phi_phot_m2_l = phi_phot_max_m2_l * f_l
        phi_phot_plant_l = phi_phot_m2_l / (rho_l + 1e-5)
        
        # (2) 呼吸与干重演变
        r_gr_l = r_gr_max_l * (xDn_l / (xDs_l + xDn_l + 1e-8)) * (self.c_Q10_gr ** ((T - 20.0) / 10.0))
        phi_resp_plant_l = (self.c_resp_s * (1 - self.c_tau) + self.c_resp_r * self.c_tau) * xDs_l * (self.c_Q10_resp ** ((T - 25.0) / 10.0))
        
        phi_phot_c_plant_l = phi_phot_plant_l - (1.0 / self.c_alpha) * phi_resp_plant_l - ((1.0 - self.c_beta) / (self.c_alpha * self.c_beta)) * r_gr_l * xDs_l
        
        # (3) 蒸腾作用量纲修复
        phi_transp_m2_l = (1.0 - lib.exp(-self.c_a_pl * (xDs_l * rho_l))) * self.c_v_pl_ai * (self._saturation_humidity(T, lib) - H)
        phi_transp_plant_l = phi_transp_m2_l / (rho_l + 1e-5)

        # 单株导数
        xDn_l_dot = self.c_alpha * phi_phot_plant_l - r_gr_l * xDs_l - phi_resp_plant_l - ((1.0 - self.c_beta) / self.c_beta) * r_gr_l * xDs_l
        xDs_l_dot = r_gr_l * xDs_l

        # ---------------------------------------------------------
        # 4. MIMO 环境强耦合热力学 (集装箱系统级)
        # 将分离在不同区域的单株生理代谢，按绝对株数汇聚为系统总负荷
        # ---------------------------------------------------------
        N_h_total = A_h * rho_h  # 育苗区总存活株数
        N_l_total = A_l * rho_l  # 成株区总存活株数

        # 1. CO2 动态 (kg/m3/s)
        Phi_phot_c_total = N_h_total * phi_phot_c_plant_h + N_l_total * phi_phot_c_plant_l
        Phi_vent_c = uV * self.A_total * (C - Co) 
        C_dot = (1.0 / self.Volume) * (uC * self.A_total - Phi_phot_c_total - Phi_vent_c)

        # 2. 焓(温度)动态 (K/s)
        Q_led = uL_h * A_h * (1.0 - self.c_led_eff) + uL_l * A_l * (1.0 - self.c_led_eff)
        # 潜热: 水的质量流 (kg/s) * 汽化潜热 (2256.4 * 1000 J/kg) = W
        Q_transp_total = (N_h_total * phi_transp_plant_h + N_l_total * phi_transp_plant_l) * 1000.0 * self.c_lat_water
        Q_vent = self.c_cap_q_v * uV * self.A_total * (T - To)
        Q_wall = self.U_value * self.Surface * (T - To)
        
        T_dot = (1.0 / (self.c_cap_q * self.A_total)) * (uQ * self.A_total + Q_led - Q_transp_total - Q_vent - Q_wall)

        # 3. 湿度动态 (kg/m3/s)
        M_transp_total = N_h_total * phi_transp_plant_h + N_l_total * phi_transp_plant_l
        M_vent = uV * self.A_total * (H - Ho)
        M_dehum = uD * self.A_total
        H_dot = (1.0 / self.Volume) * (M_transp_total - M_vent - M_dehum)

        return [xDn_h_dot, xDs_h_dot, xDn_l_dot, xDs_l_dot, C_dot, T_dot, H_dot]