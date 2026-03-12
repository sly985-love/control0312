# ==============================================================================
# 文件路径: common/utils.py
# 描述: 全局通用工具箱 (配置读取、随机种子固定、目录管理)
# ==============================================================================

import os
import yaml
import random
import numpy as np
import torch
from typing import Dict, Any

def load_config(file_path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"配置文件未找到: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_global_seed(seed: int = 42) -> None:
    """
    固定所有计算后端的随机种子。
    这是顶级期刊复现实验 (Reproducibility) 的核心要求！
    """
    # 1. Python 内置随机库
    random.seed(seed)
    # 2. Numpy 随机库
    np.random.seed(seed)
    # 3. PyTorch 随机库
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 强制 CuDNN 使用确定性算法，牺牲一点点速度换取绝对可复现
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 设定 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[Utils] 全局随机种子已固定为: {seed}")

def make_dirs(path: str) -> None:
    """安全创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[Utils] 创建新目录: {path}")