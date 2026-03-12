# ==============================================================================
# 文件路径: RL/extract_sb3_vf.py
# 描述: Stable-Baselines3 模型手术刀 (Critic -> Value Function 提取器)
# 终极修复:
#   1. 修复 DummyEnv 中 numpy 默认生成 float64 导致 SB3 崩溃的类型错位。
#   2. 强制指定 SB3 的激活函数为 ReLU，杜绝数学特征对齐错乱。
#   3. 【修复: 权重提取逻辑的脆弱性】废除硬编码键值，采用按张量形状与顺序的动态解析，彻底适配任意深度的网络层。
# 核心作用: 
#   将 SB3 训练出的大型黑盒模型 (.zip)，剥离出极其轻量的价值网络权重 (.pth)。
#   提取后的 .pth 文件将直接作为 rl_smpc.py 的“无限未来指路明灯”。
# ==============================================================================

import os
import torch
import warnings
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.utils import get_device

# 导入纯净网络结构
from RL.rl_network import ValueNetwork

def extract_vf_from_sb3(sb3_model_path: str, output_pth_path: str, algo: str = "PPO", obs_dim: int = 13, hidden_dim: int = 256):
    """从 SB3 的 ZIP 模型中提取状态价值函数 V(s) 的权重"""
    if not os.path.exists(sb3_model_path):
        raise FileNotFoundError(f"❌ 找不到 SB3 模型文件: {sb3_model_path}")

    print(f"🔍 正在挂载 SB3 {algo} 模型黑盒: {sb3_model_path}")
    
    device = get_device("cpu")
    custom_objects = {"device": device}
    
    # 实例化接收权重的干净网络
    clean_vf = ValueNetwork(obs_dim=obs_dim, hidden_dim=hidden_dim).to(device)
    vf_state_dict = clean_vf.state_dict()

    # 动态收集源模型的权重和偏置
    source_weights = []
    source_biases = []

    if algo.upper() == "PPO":
        model = PPO.load(sb3_model_path, custom_objects=custom_objects)
        
        # 依次遍历特征提取层和最终输出头的参数
        for param in model.policy.mlp_extractor.value_net.parameters():
            if len(param.shape) > 1: source_weights.append(param.data.clone())
            else: source_biases.append(param.data.clone())
            
        for param in model.policy.value_net.parameters():
            if len(param.shape) > 1: source_weights.append(param.data.clone())
            else: source_biases.append(param.data.clone())

    elif algo.upper() == "SAC":
        model = SAC.load(sb3_model_path, custom_objects=custom_objects)
        warnings.warn("注意: SAC 算法原生无独立 V(s) 网络。将提取 Actor 的特征提取层替代！输出层将保留随机初始化。")
        
        for param in model.actor.latent_pi.parameters():
            if len(param.shape) > 1: source_weights.append(param.data.clone())
            else: source_biases.append(param.data.clone())
    else:
        raise ValueError(f"不支持的算法: {algo}")

    # 动态按序分配至目标网络，彻底告别硬编码键值
    w_idx, b_idx = 0, 0
    try:
        for key in vf_state_dict.keys():
            if 'weight' in key and w_idx < len(source_weights):
                # 严格校验形状，防止隐式赋值导致的张量错位
                if vf_state_dict[key].shape == source_weights[w_idx].shape:
                    vf_state_dict[key] = source_weights[w_idx]
                    w_idx += 1
                else:
                    raise RuntimeError(f"权重形状不匹配: 目标 {key} 需 {vf_state_dict[key].shape}, 源提供 {source_weights[w_idx].shape}")
            
            elif 'bias' in key and b_idx < len(source_biases):
                if vf_state_dict[key].shape == source_biases[b_idx].shape:
                    vf_state_dict[key] = source_biases[b_idx]
                    b_idx += 1
                else:
                    raise RuntimeError(f"偏置形状不匹配: 目标 {key} 需 {vf_state_dict[key].shape}, 源提供 {source_biases[b_idx].shape}")
        
        print(f"✅ {algo} 价值网络 V(s) 完美解剖与移植成功！(共移植 {w_idx} 层权重, {b_idx} 层偏置)")
    except Exception as e:
        raise RuntimeError(f"❌ 动态权重映射失败！详细: {e}")

    os.makedirs(os.path.dirname(output_pth_path), exist_ok=True)
    torch.save(vf_state_dict, output_pth_path)
    print(f"🎯 纯净价值网络已保存至: {output_pth_path}")


# ==============================================================================
# 单体测试与验证
# ==============================================================================
if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium import spaces
    import torch.nn as nn
    
    print("================================================================")
    print("🟢 SB3 模型剥离手术刀集成测试")
    print("================================================================")
    
    test_zip_path = "models/test_ppo_model.zip"
    test_pth_path = "models/extracted_ppo_vf.pth"
    os.makedirs("models", exist_ok=True)
    
    print("🔧 1. 正在本地紧急锻造虚拟 PPO 模型黑盒...")
    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        # 显式声明 dtype=np.float32，防止 SB3 发生 float64 数据断言崩溃
        def step(self, a): return np.zeros(13, dtype=np.float32), 0.0, False, False, {}
        def reset(self, seed=None, options=None): return np.zeros(13, dtype=np.float32), {}
        
    # 强制指定 SB3 的激活函数为 ReLU (默认是 Tanh)，确保数学特征提取逻辑绝对对齐
    policy_kwargs = dict(
        activation_fn=nn.ReLU, 
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )
    dummy_model = PPO("MlpPolicy", DummyEnv(), policy_kwargs=policy_kwargs, n_steps=8, device="cpu")
    dummy_model.save(test_zip_path)
    
    print("\n🔪 2. 启动手术刀剥离程序...")
    extract_vf_from_sb3(sb3_model_path=test_zip_path, output_pth_path=test_pth_path, algo="PPO", obs_dim=13, hidden_dim=256)
    
    print("\n🔬 3. 验证移植后的 .pth 文件...")
    recovered_net = ValueNetwork(obs_dim=13, hidden_dim=256)
    recovered_net.load_state_dict(torch.load(test_pth_path))
    
    dummy_obs = torch.zeros(1, 13)
    val = recovered_net(dummy_obs)
    print(f"✅ 提取后的网络推理成功！输出标量打分 V: {val.item():.6f}")
    
    if os.path.exists(test_zip_path): os.remove(test_zip_path)
    if os.path.exists(test_pth_path): os.remove(test_pth_path)