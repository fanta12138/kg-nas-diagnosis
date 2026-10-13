#!/usr/bin/env python3
"""
在RTX 3090上重新运行完整的NAS流程（Stage 1 → 2 → 3）

优化配置：
- Stage 1: 增大batch_size到64（利用24GB显存）
- Stage 2: 完成全部30次trial搜索
- Stage 3: 修复评估问题，使用正确的数据集路径
"""

import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"命令: {' '.join(cmd)}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ 完成! 耗时: {elapsed/60:.1f}分钟")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ 失败! 耗时: {elapsed/60:.1f}分钟")
        print(f"错误代码: {e.returncode}")
        return False


def main():
    print("="*70)
    print("🎯 RTX 3090 完整NAS流程重跑")
    print("="*70)
    print("\n硬件信息:")
    print("  GPU: NVIDIA RTX 3090 (24GB)")
    print("  优化策略:")
    print("    - Stage 1: batch_size=64 (原32)")
    print("    - Stage 2: num_trials=30 (完整搜索)")
    print("    - Stage 3: 修复评估bug")
    print("="*70)
    
    # 确认开始
    response = input("\n⚠️  确认开始？这将覆盖现有结果 (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("已取消")
        sys.exit(0)
    
    total_start = time.time()
    
    # ==================== Stage 1: 教师模型训练 ====================
    stage1_success = run_command(
        [
            sys.executable, 'scripts/stage1_train_teacher.py',
            '--config', 'configs/stage1_contrastive.yaml',
            '--data-dir', '../datasets/mvtec_anomaly_detection/wood',
            '--output-dir', 'experiments/stage1_rtx3090',
            '--batch-size', '64',  # RTX 3090优化
            '--epochs', '100'
        ],
        "Stage 1: 对比学习预训练教师模型"
    )
    
    if not stage1_success:
        print("\n❌ Stage 1失败，终止流程")
        sys.exit(1)
    
    # ==================== Stage 2: NAS架构搜索 ====================
    stage2_success = run_command(
        [
            sys.executable, 'scripts/stage2_nas_search.py',
            '--config', 'configs/stage2_nas_search.yaml',
            '--teacher-model', 'experiments/stage1_rtx3090_convnextv2/best_teacher.pth',  # 更新为ConvNeXtV2
            '--data-dir', '../datasets/mvtec_anomaly_detection/wood',
            '--output-dir', 'experiments/stage2_rtx3090',
            '--num-trials', '30'  # 完整搜索
        ],
        "Stage 2: NAS架构搜索（30次试验）"
    )
    
    if not stage2_success:
        print("\n❌ Stage 2失败，终止流程")
        sys.exit(1)
    
    # ==================== Stage 3: 最终训练 ====================
    stage3_success = run_command(
        [
            sys.executable, 'scripts/stage3_final_training.py',
            '--config', 'configs/final_training.yaml',
            '--arch-config', 'experiments/stage2_rtx3090/best_architecture.yaml',
            '--data-dir', '../datasets/mvtec_anomaly_detection/wood',
            '--output-dir', 'experiments/final_rtx3090'
        ],
        "Stage 3: 使用最佳架构进行最终训练"
    )
    
    if not stage3_success:
        print("\n❌ Stage 3失败")
        sys.exit(1)
    
    # ==================== 完成总结 ====================
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"✅ 完整NAS流程完成！")
    print(f"{'='*70}")
    print(f"总耗时: {total_elapsed/60:.1f}分钟 ({total_elapsed/3600:.2f}小时)")
    print(f"\n结果位置:")
    print(f"  Stage 1: experiments/stage1_rtx3090/")
    print(f"  Stage 2: experiments/stage2_rtx3090/")
    print(f"  Stage 3: experiments/final_rtx3090/")
    print(f"\n下一步:")
    print(f"  1. 检查Stage 3的AUROC是否合理（应该>0.8）")
    print(f"  2. 如果性能不佳，分析原因并调整")
    print(f"  3. 与基线方法对比")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
