#!/usr/bin/env python3
"""
整理Stage 2 NAS搜索结果，生成Origin画图所需的结构化数据

输出:
1. Figure 6原始数据 (CNN vs Transformer对比)
2. 搜索统计摘要
3. Backbone性能分布
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np


def organize_nas_results(stage2_dir: str = 'experiments/stage2_topjournal_16backbones'):
    """整理NAS搜索结果"""
    
    stage2_path = Path(stage2_dir)
    results_file = stage2_path / 'search_results.json'
    
    if not results_file.exists():
        print(f"❌ 未找到搜索结果文件: {results_file}")
        print(f"   请等待Stage 2完成后再运行此脚本")
        return
    
    # 加载搜索结果
    print(f"📊 加载搜索结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = data.get('all_results', [])
    best_arch = data.get('best_architecture', {})
    
    print(f"   总trials: {len(results)}/100")
    print(f"   最佳架构: {best_arch}")
    
    if len(results) == 0:
        print("❌ 没有成功的trials")
        return
    
    # 定义架构分类
    cnn_backbones = [
        'convnextv2_tiny', 'convnextv2_base', 'efficientnetv2_s',
        'mobileone_s0', 'convnext_tiny', 'convnext_small',
        'efficientnet_b0', 'efficientnet_b1', 'mobilenetv3_small_100',
        'resnet18', 'resnet34', 'resnet50'
    ]
    
    transformer_backbones = [
        'vit_tiny_patch16_224', 'deit_tiny_patch16_224',
        'swin_tiny_patch4_window7_224'
    ]
    
    all_backbones = cnn_backbones + transformer_backbones
    
    # 统计每个backbone的表现
    backbone_stats = defaultdict(list)
    backbone_dims = defaultdict(lambda: defaultdict(list))
    
    for trial in results:
        bb = trial['architecture']['backbone']
        dim = trial['architecture']['feature_dim']
        score = trial['score']
        
        backbone_stats[bb].append(score)
        backbone_dims[bb][dim].append(score)
    
    # 创建输出目录
    output_dir = Path('experiments/results_for_paper/origin_plot_data')
    main_results_dir = Path('experiments/results_for_paper/main_results')
    ablation_dir = Path('experiments/results_for_paper/ablation_studies')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    main_results_dir.mkdir(parents=True, exist_ok=True)
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 生成结果文件...")
    
    # ========================================
    # 1. Figure 6: CNN vs Transformer 原始数据
    # ========================================
    figure6_file = output_dir / 'figure_6_cnn_vs_transformer_raw.csv'
    
    with open(figure6_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Architecture_Type', 'Backbone', 'Avg_Score', 'Std_Score', 
                        'Min_Score', 'Max_Score', 'Num_Trials'])
        
        for bb in all_backbones:
            if bb not in backbone_stats or len(backbone_stats[bb]) == 0:
                continue
            
            scores = backbone_stats[bb]
            avg_score = np.mean(scores)
            std_score = np.std(scores) if len(scores) > 1 else 0.0
            
            arch_type = 'CNN' if bb in cnn_backbones else 'Transformer'
            
            writer.writerow([
                arch_type,
                bb,
                f'{avg_score:.6f}',
                f'{std_score:.6f}',
                f'{min(scores):.6f}',
                f'{max(scores):.6f}',
                len(scores)
            ])
    
    print(f"   ✓ Figure 6数据: {figure6_file}")
    
    # ========================================
    # 2. Backbone性能分布详细数据
    # ========================================
    backbone_detail_file = output_dir / 'backbone_performance_detail.csv'
    
    with open(backbone_detail_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Backbone', 'Feature_Dim', 'Score', 'Trial_Index'])
        
        trial_idx = 0
        for trial in results:
            trial_idx += 1
            writer.writerow([
                trial['architecture']['backbone'],
                trial['architecture']['feature_dim'],
                f"{trial['score']:.6f}",
                trial_idx
            ])
    
    print(f"   ✓ Backbone详细数据: {backbone_detail_file}")
    
    # ========================================
    # 3. 搜索统计摘要
    # ========================================
    summary_file = main_results_dir / 'nas_search_summary.json'
    
    # 按backbone分组的统计
    backbone_summary = {}
    for bb in all_backbones:
        if bb in backbone_stats and len(backbone_stats[bb]) > 0:
            scores = backbone_stats[bb]
            backbone_summary[bb] = {
                'num_trials': len(scores),
                'avg_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'min_score': float(min(scores)),
                'max_score': float(max(scores)),
                'by_dimension': {}
            }
            
            # 按feature dimension分组
            for dim in [128, 256, 512]:
                if dim in backbone_dims[bb] and len(backbone_dims[bb][dim]) > 0:
                    dim_scores = backbone_dims[bb][dim]
                    backbone_summary[bb]['by_dimension'][str(dim)] = {
                        'num_trials': len(dim_scores),
                        'avg_score': float(np.mean(dim_scores)),
                        'std_score': float(np.std(dim_scores)) if len(dim_scores) > 1 else 0.0
                    }
    
    # CNN vs Transformer汇总
    cnn_scores = []
    transformer_scores = []
    
    for bb, scores in backbone_stats.items():
        if bb in cnn_backbones:
            cnn_scores.extend(scores)
        elif bb in transformer_backbones:
            transformer_scores.extend(scores)
    
    architecture_comparison = {
        'CNN': {
            'total_trials': len(cnn_scores),
            'avg_score': float(np.mean(cnn_scores)) if cnn_scores else 0.0,
            'std_score': float(np.std(cnn_scores)) if len(cnn_scores) > 1 else 0.0
        },
        'Transformer': {
            'total_trials': len(transformer_scores),
            'avg_score': float(np.mean(transformer_scores)) if transformer_scores else 0.0,
            'std_score': float(np.std(transformer_scores)) if len(transformer_scores) > 1 else 0.0
        }
    }
    
    summary_data = {
        'total_trials': len(results),
        'success_rate': len(results) / 100.0,
        'best_architecture': best_arch,
        'backbone_statistics': backbone_summary,
        'architecture_type_comparison': architecture_comparison
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ 搜索统计摘要: {summary_file}")
    
    # ========================================
    # 4. 打印关键发现
    # ========================================
    print(f"\n{'='*70}")
    print(f"📊 NAS搜索关键发现")
    print(f"{'='*70}")
    print(f"成功率: {len(results)/100*100:.1f}% ({len(results)}/100)")
    print(f"最佳架构: {best_arch.get('backbone', 'N/A')} + dim={best_arch.get('feature_dim', 'N/A')}")
    print(f"\nCNN vs Transformer:")
    print(f"  CNN trials: {architecture_comparison['CNN']['total_trials']}")
    print(f"  CNN avg score: {architecture_comparison['CNN']['avg_score']:.6f}")
    print(f"  Transformer trials: {architecture_comparison['Transformer']['total_trials']}")
    print(f"  Transformer avg score: {architecture_comparison['Transformer']['avg_score']:.6f}")
    
    print(f"\nTop 5 Backbones by Average Score (lower is better):")
    sorted_backbones = sorted(
        [(bb, stats['avg_score'], stats['num_trials']) 
         for bb, stats in backbone_summary.items()],
        key=lambda x: x[1]
    )[:5]
    
    for i, (bb, avg_score, num_trials) in enumerate(sorted_backbones, 1):
        print(f"  {i}. {bb}: {avg_score:.6f} ({num_trials} trials)")
    
    print(f"\n{'='*70}")
    print(f"✅ 结果整理完成！")
    print(f"{'='*70}")
    print(f"\n下一步:")
    print(f"  1. 检查生成的CSV文件是否符合Origin导入要求")
    print(f"  2. 等待Stage 3完成以获取最终AUROC指标")
    print(f"  3. 运行基线方法对比实验")
    print(f"  4. 执行消融实验")


if __name__ == '__main__':
    import sys
    
    # 允许命令行指定stage2目录
    stage2_dir = sys.argv[1] if len(sys.argv) > 1 else 'experiments/stage2_topjournal_16backbones'
    
    organize_nas_results(stage2_dir)
