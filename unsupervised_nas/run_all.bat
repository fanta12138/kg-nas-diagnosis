@echo off
REM 一键运行所有实验 (Windows版本)

echo =========================================
echo 无监督NAS for 木材缺陷检测
echo =========================================
echo.

REM 检查数据集
if not exist "..\datasets\MVTec-AD\wood" (
    echo 错误: 未找到MVTec-AD wood数据集
    echo 请确保数据集位于: ..\datasets\MVTec-AD\wood
    pause
    exit /b 1
)

echo 步骤 1/3: 训练教师模型（对比学习）
echo -----------------------------------------
python scripts\stage1_train_teacher.py ^
    --config configs\stage1_contrastive.yaml ^
    --output-dir experiments\stage1

if %errorlevel% neq 0 (
    echo 阶段1失败！
    pause
    exit /b 1
)

echo.
echo 步骤 2/3: NAS架构搜索
echo -----------------------------------------
python scripts\stage2_nas_search.py ^
    --config configs\stage2_nas_search.yaml ^
    --teacher-model experiments\stage1\best_teacher.pth ^
    --output-dir experiments\stage2

if %errorlevel% neq 0 (
    echo 阶段2失败！
    pause
    exit /b 1
)

echo.
echo 步骤 3/3: 最终训练
echo -----------------------------------------
python scripts\stage3_final_training.py ^
    --config configs\final_training.yaml ^
    --output-dir experiments\final

if %errorlevel% neq 0 (
    echo 阶段3失败！
    pause
    exit /b 1
)

echo.
echo =========================================
echo 所有实验完成！
echo =========================================
echo.
echo 结果保存在: experiments\
echo 最佳模型: experiments\final\best_model.pth
echo.
echo 评估模型:
echo python scripts\evaluate.py --model-path experiments\final\best_model.pth
echo.
pause
