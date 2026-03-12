#!/bin/bash

# ==============================================================================
# 文件路径: run_pipeline.sh
# 描述: PFAL-RLSMPC 项目一键自动化实验流水线 (架构闭环版)
# 修正: 宏观排程已正式由贝叶斯优化(BO)升级为完美适配物理离散批次的遗传算法(GA)。
# ==============================================================================

# 定义颜色常量
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' 

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}🌟 PFAL-RLSMPC 数字孪生系统自动化流水线启动${NC}"
echo -e "${BLUE}================================================================${NC}"

# 0. 准备阶段
echo -e "${YELLOW}[Step 0/4] 正在检查目录与环境配置...${NC}"
mkdir -p models results logs plots configs/macro_best
sleep 1

# 1. 强化学习炼丹
echo -e "${YELLOW}[Step 1/4] 启动 RL 炼丹炉：正在训练价值网络 (Value Function)...${NC}"
# 注意：若需云端监控可替换为 train_rl_wandb
python -m experiments.train_rl
if [ $? -ne 0 ]; then echo "❌ Step 1 失败，流水线中断"; exit 1; fi

# 2. 宏观寻优 (已升级为 GA)
echo -e "${YELLOW}[Step 2/4] 启动宏观战略家：正在通过遗传算法 (GA) 寻找最佳离散物理排程...${NC}"
python -m macro_optimizers.ga_scheduler
if [ $? -ne 0 ]; then echo "❌ Step 2 失败，流水线中断"; exit 1; fi

# 3. 多算法测评
echo -e "${YELLOW}[Step 3/4] 启动角斗场：正在对 BASELINE, NMPC, RL_SMPC 进行性能大乱斗...${NC}"
for algo in BASELINE NMPC RL_SMPC
do
    echo -e "   ➡️ 正在评测算法: ${algo}..."
    python -m experiments.evaluate_benchmark --algo $algo
    if [ $? -ne 0 ]; then echo "❌ 算法 ${algo} 评测崩溃，流水线中断"; exit 1; fi
done

# 4. 可视化出图
echo -e "${YELLOW}[Step 4/4] 启动画师：正在生成高清论文配图...${NC}"
python -m experiments.visualize
if [ $? -ne 0 ]; then echo "⚠️ Step 4 出图部分出现异常，请检查绘图脚本"; fi

echo -e "${GREEN}================================================================${NC}"
echo -e "${GREEN}✅ 流水线任务圆满完成！PFAL-RLSMPC 系统验证闭环！${NC}"
echo -e "${GREEN}📂 最终价值网络与策略位于: /models${NC}"
echo -e "${GREEN}📂 最佳排程参数位于: /configs/macro_best/ga_best_schedule.yml${NC}"
echo -e "${GREEN}📂 测评指标数据位于: /results${NC}"
echo -e "${GREEN}📂 论文级对比图表位于: /plots${NC}"
echo -e "${GREEN}================================================================${NC}"