# ==============================================================================
# 文件路径: experiments/train_rl.py
# 描述: 强化学习全自动炼丹流水线 (稳定版)
# 修正说明:
#   1. 强制使用 PPO 算法，以确保能提取到数学意义上最纯正的 V(s) 终端价值函数。
#   2. 【修复3: 归一化冲突】彻底移除 VecNormalize，完全依赖底层 ObservationScaler 的静态 Min-Max 缩放。
#      避免训练时动态缩放奖励导致提取出的价值网络与 MPC 物理目标函数的量纲撕裂。
#   3. 修复 CasADi 句柄在多进程下的序列化冲突。
# ==============================================================================

import os
import yaml
import time
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 导入底层基建
from envs.pfal_env_dual import PFALEnvDual
from RL.extract_sb3_vf import extract_vf_from_sb3

def train_agent():
    # 1. 路径初始化
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    log_dir = os.path.join(base_dir, "logs", "ppo_training")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. 实例化环境
    # 【修复】：彻底移除 VecNormalize。底层 env.rl_mode=True 时，
    # ObservationScaler 会接管观测归一化，而奖励将保持原汁原味的经济学量纲。
    def make_env():
        env = PFALEnvDual(config_dir=base_dir, rl_mode=True)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([make_env])

    # 3. 配置 PPO 算法 (PPO 拥有最适合 RL-SMPC 的 V-net 结构)
    # 隐藏层必须与 RL/rl_network.py 保持一致
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]) 
    )

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"
    )

    # 4. 设定评估回调
    eval_callback = EvalCallback(
        venv, 
        best_model_save_path=model_dir,
        log_path=log_dir, 
        eval_freq=5000,
        deterministic=True
    )

    # 5. 启动训练
    print("🔥 [炼丹开始] 正在训练 PPO 指挥官...")
    total_steps = 300000 # 初始测试可用 30w，正式论文建议 100w
    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    # 6. 保存最终模型
    final_model_path = os.path.join(model_dir, "ppo_final_model.zip")
    model.save(final_model_path)
    # (已移除 vec_normalize.pkl 的保存逻辑)

    # 7. 【自动手术】剥离 V-Net 权重
    print("\n🔪 正在剥离 V(s) 终端价值网络权重...")
    best_zip = os.path.join(model_dir, "best_model.zip") # EvalCallback 自动保存的文件
    output_pth = os.path.join(model_dir, "value_network_weights.pth")
    
    # 确保提取器能找到我们定义的 256 隐藏层
    try:
        extract_vf_from_sb3(
            sb3_model_path=best_zip if os.path.exists(best_zip) else final_model_path,
            output_pth_path=output_pth,
            algo="PPO",
            obs_dim=13,
            hidden_dim=256
        )
        print(f"🎯 价值网络已提取并就绪: {output_pth}")
    except Exception as e:
        print(f"⚠️ 提取失败: {e}。这通常是因为训练未达到保存点的步数。")

if __name__ == "__main__":
    train_agent()