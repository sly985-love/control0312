# ==============================================================================
# 文件路径: experiments/train_rl_wandb.py
# 描述: 强化学习训练流水线 + 实时云端监控 (WandB 版)
# 修正说明:
#   1. 集成 Weights & Biases (WandB)，实现远程实验跟踪。
#   2. 【修复3: 归一化冲突】彻底移除 VecNormalize 以防量纲撕裂。
# ==============================================================================

import os
import time
import torch as th
import torch.nn as nn
import wandb
from wandb.integration.sb3 import WandbCallback  # SB3 专属插件

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# 方法2：如果相对导入报错，用sys.path临时添加根目录（兼容方案）
import sys
import os
# 获取当前文件所在目录的上级目录（即GA_RLSMPC_PFAL根目录）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from envs.pfal_env_dual import PFALEnvDual
from RL.extract_sb3_vf import extract_vf_from_sb3

def train_agent():
    # 1. 初始化 WandB
    # 注意：运行前请先在终端运行 'wandb login'
    run = wandb.init(
        project="PFAL-RLSMPC-Research", # 项目名
        config={
            "algo": "PPO",
            "total_timesteps": 300000,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "net_arch": [256, 256],
            "batch_size": 64
        },
        sync_tensorboard=True,  # 关键：自动转发 Tensorboard 数据
        monitor_gym=True,
        save_code=True,         # 保存当前训练时的代码版本
    )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # 2. 构造物理基准环境 (不使用 VecNormalize)
    def make_env():
        env = PFALEnvDual(config_dir=base_dir, rl_mode=True)
        env = Monitor(env)
        return env

    venv = DummyVecEnv([make_env])

    # 3. 算法架构定义 (需与 RL/rl_network.py 的 MLP 拓扑严格对齐)
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[256, 256], vf=[256, 256]) 
    )

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=run.config["learning_rate"],
        batch_size=run.config["batch_size"],
        gamma=run.config["gamma"],
        verbose=1,
        tensorboard_log=f"runs/{run.id}", # 对应 WandB 同步目录
        device="auto"
    )

    # 4. 设置混合回调：EvalCallback (保存本地) + WandbCallback (上传云端)
    eval_callback = EvalCallback(
        venv, 
        best_model_save_path=model_dir,
        eval_freq=5000,
        deterministic=True
    )
    
    # 自动保存模型权重和性能指标到 WandB
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=os.path.join(model_dir, f"wandb_{run.id}"),
        verbose=2,
    )

    # 5. 开启训练
    print(f"🔥 [WandB] 实验室已开启！实验 ID: {run.id}")
    model.learn(
        total_timesteps=run.config["total_timesteps"],
        callback=[eval_callback, wandb_callback],
        progress_bar=True
    )

    # 6. 保存最终模型
    model.save(os.path.join(model_dir, "ppo_final_model.zip"))

    # 7. 自动执行“模型手术”提取 V-Net
    print("\n🔪 正在剥离用于 RL-SMPC 的价值网络权重...")
    best_zip = os.path.join(model_dir, "best_model.zip")
    output_pth = os.path.join(model_dir, "value_network_weights.pth")
    
    try:
        extract_vf_from_sb3(
            sb3_model_path=best_zip if os.path.exists(best_zip) else os.path.join(model_dir, "ppo_final_model.zip"),
            output_pth_path=output_pth,
            algo="PPO",
            obs_dim=13,
            hidden_dim=256
        )
        # 上传最终的 .pth 文件到 WandB云端，防止丢失
        wandb.save(output_pth)
        print(f"🎯 价值网络提取完成并已同步至云端。")
    except Exception as e:
        print(f"⚠️ 提取失败: {e}")

    # 8. 结束实验
    run.finish()

if __name__ == "__main__":
    train_agent()