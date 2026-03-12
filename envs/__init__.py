# ==============================================================================
# 文件路径: envs/__init__.py
# 描述: Gymnasium 环境注册表 (Registry)
# 作用: 允许使用 gym.make("PFAL_Dual-v0") 标准接口实例化环境，完美兼容 SB3 并行训练。
# ==============================================================================

from gymnasium.envs.registration import register

# 导入模块以便外部可以通过 from envs import PFALEnvDual 直接访问
from .pfal_env_dual import PFALEnvDual
from .pfal_dynamics_dual import PFALDynamicsDual
from .observations_dual import ObservationScaler

# 将我们的双区植物工厂环境注册到全局 Gymnasium 字典中
register(
    id='PFAL_Dual-v0',                                # 标准环境调用 ID
    entry_point='envs.pfal_env_dual:PFALEnvDual',     # 指向我们刚写的环境类
    # 注意: 由于我们的 max_episode_steps 是由上层 BO 动态决定的 (T_h + T_l)
    # 所以我们在这里不写死 max_episode_steps，而是在 env 的 step() 里通过 truncated 自行截断。
)

print("[Envs] PFAL_Dual-v0 环境已成功注册至 Gymnasium！")