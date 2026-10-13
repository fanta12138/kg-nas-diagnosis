#!/bin/bash
# 一键运行所有实验

echo "========================================="
echo "无监督NAS for 木材缺陷检测"
echo "========================================="
echo ""

# 检查数据集
if [ ! -d "../datasets/MVTec-AD/wood" ]; then
    echo "错误: 未找到MVTec-AD wood数据集"
    echo "请确保数据集位于: ../datasets/MVTec-AD/wood"
    exit 1
fi

echo "步骤 1/3: 训练教师模型（对比学习）"
echo "-----------------------------------------"
python scripts/stage1_train_teacher.py \
    --config configs/stage1_contrastive.yaml \
    --output-dir experiments/stage1

if [ $? -ne 0 ]; then
    echo "阶段1失败！"
    exit 1
fi

echo ""
echo "步骤 2/3: NAS架构搜索"
echo "-----------------------------------------"
python scripts/stage2_nas_search.py \
    --config configs/stage2_nas_search.yaml \
    --teacher-model experiments/stage1/best_teacher.pth \
    --output-dir experiments/stage2

if [ $? -ne 0 ]; then
    echo "阶段2失败！"
    exit 1
fi

echo ""
echo "步骤 3/3: 最终训练"
echo "-----------------------------------------"
python scripts/stage3_final_training.py \
    --config configs/final_training.yaml \
    --output-dir experiments/final

if [ $? -ne 0 ]; then
    echo "阶段3失败！"
    exit 1
fi

echo ""
echo "========================================="
echo "所有实验完成！"
echo "========================================="
echo ""
echo "结果保存在: experiments/"
echo "最佳模型: experiments/final/best_model.pth"
echo ""
echo "评估模型:"
echo "python scripts/evaluate.py --model-path experiments/final/best_model.pth"
