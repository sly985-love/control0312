# ==============================================================================
# 文件路径: common/wandb_logger.py
# 描述: 集成 Weights & Biases (Wandb) 的实验监控与日志收集器
# 作用: 实时追踪 BO 寻优过程中的排程参数探索轨迹、EPI 变化及约束违规情况。
# ==============================================================================

import wandb
from typing import Dict, Any

class BOLogger:
    def __init__(self, project_name: str = "BO-RLSMPC-PFAL", experiment_name: str = "BO_Optimization"):
        """
        初始化 Wandb 实验追踪器
        注意: 运行前需要在终端执行 `wandb login` 登录你的账号。
        """
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.run = None

    def start_run(self, config_dict: Dict[str, Any]) -> None:
        """开启一个记录会话，并上传全局超参数配置"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=config_dict,
            reinit=True
        )
        print(f"[Wandb] 实验记录已启动: {self.experiment_name}")

    def log_bo_iteration(self, iteration: int, params: Dict[str, float], 
                         epi: float, penalty: float, yield_kg: float) -> None:
        """
        记录每一次贝叶斯优化(BO)迭代的结果。
        
        参数:
        iteration: 当前迭代次数
        params: 动作空间字典 (t_h, t_l, rho_h, rho_l, photoperiod)
        epi: 综合能耗指标 (kWh/kg)
        penalty: 违规惩罚值
        yield_kg: 预估产能
        """
        if self.run is None:
            return

        log_data = {
            "Iteration": iteration,
            "Target/EPI": epi,
            "Target/Penalty": penalty,
            "Target/Estimated_Yield_kg": yield_kg,
            
            # 记录探索的排程动作
            "Actions/t_h": params.get('t_h', 0),
            "Actions/t_l": params.get('t_l', 0),
            "Actions/rho_h": params.get('rho_h', 0),
            "Actions/rho_l": params.get('rho_l', 0),
            "Actions/photoperiod": params.get('photoperiod', 0)
        }
        
        # 上传到 Wandb 云端实时画图
        wandb.log(log_data, step=iteration)

    def finish_run(self) -> None:
        """结束记录会话"""
        if self.run:
            self.run.finish()
            print("[Wandb] 实验记录已结束并同步至云端。")