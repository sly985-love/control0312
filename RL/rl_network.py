# ==============================================================================
# 文件路径: RL/rl_network.py
# 描述: 强化学习与 RL-MPC 共享的 PyTorch 神经网络架构定义库
# 终极修复: 补充了 numpy 导包，彻底修复初始化时的 NameError 崩溃漏洞。
# 核心作用:
#   作为“真理唯一来源”，集中管理状态价值网络 (Value Function) 的结构。
#   保证 SB3 训练抽取的模型与 CasADi 符号推理端在拓扑结构上绝对同源一致。
# ==============================================================================

import torch
import torch.nn as nn
import numpy as np  # 【修复1】: 必须导入 numpy 以支持正交初始化中的数学运算

class ValueNetwork(nn.Module):
    """
    终极状态价值网络 (State Value Function, V-Net)。
    
    输入维度 (obs_dim): 
      13 维全视野状态 (7维物理状态 + 3维设定边界 + 3维外部气象)。
      
    输出维度: 
      1 维标量 (代表在该状态下，未来无限步的预期折现总收益)。
      
    架构限制注意:
      为了配合 controllers/ 中的 CasADi 动态反射解析器，
      这里【必须】使用 nn.Sequential，且目前仅支持 nn.Linear 和 nn.ReLU 的交替。
    """
    def __init__(self, obs_dim: int = 13, hidden_dim: int = 256):
        super(ValueNetwork, self).__init__()
        
        # 标准的 2 层全连接隐藏层多层感知机 (MLP)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 最终输出标量 V(s)
        )
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """对网络层进行正交初始化 (Orthogonal Initialization) 以提升稳定性"""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # 这里依赖了 np.sqrt，必须确保 numpy 已正确导入
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch 前向传播接口"""
        return self.net(x)

# ==============================================================================
# 单体网络连通性测试
# ==============================================================================
if __name__ == "__main__":
    print("🟢 检查 ValueNetwork 架构连通性...")
    model = ValueNetwork(obs_dim=13, hidden_dim=256)
    
    dummy_obs = torch.tensor(np.random.randn(1, 13), dtype=torch.float32)
    
    with torch.no_grad():
        v_value = model(dummy_obs)
        
    print(f"✅ 网络实例化与初始化成功！")
    print(f"  输入形状: {dummy_obs.shape}")
    print(f"  输出形状: {v_value.shape}")
    print(f"  初始随机价值输出: {v_value.item():.4f}")