#!/usr/bin/env python3
"""
收集并分析P0消融实验的所有结果
- Ablation 3: 损失函数组件
- Ablation 4: 动态维度对齐
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


# MVTec-AD全部15个类别
ALL_CATEGORIES = [
    'bottle', 'cable', 'capsule', 'carpet', 'grid',
    'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
    'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
]


def collect_ablation_3_results():
    """收集Ablation 3（损失函数组件）的结果"""
    print("\n" + "="*80)
    print("📊 收集 Ablation 3: 损失函数组件结果")
    print("="*80)
    
    base_dir = Path("experiments/ablation_3_loss_functions")
    results = {}
    
    for category in ALL_CATEGORIES:
        summary_file = base_dir / category / "ablation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results[category] = data
                print(f"✅ {category:15s} - kd_only: {data['kd_only']['auroc']:.4f}, "
                      f"kd_consistency: {data['kd_consistency']['auroc']:.4f}, "
                      f"full: {data['full']['auroc']:.4f}")
        else:
            print(f"❌ {category:15s} - 文件不存在")
    
    return results


def collect_ablation_4_results():
    """收集Ablation 4（动态维度对齐）的结果"""
    print("\n" + "="*80)
    print("📊 收集 Ablation 4: 动态维度对齐结果")
    print("="*80)
    
    base_dir = Path("experiments/ablation_4_dimension_alignment")
    results = {}
    
    for category in ALL_CATEGORIES:
        summary_file = base_dir / category / "ablation_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)
                results[category] = data
                
                # 计算平均提升
                improvements = []
                backbones = ['resnet18', 'efficientnet_b0', 'vit_tiny_patch16_224', 'convnextv2_tiny']
                
                for backbone in backbones:
                    if backbone in data['fixed_alignment'] and backbone in data['dynamic_alignment']:
                        fixed = data['fixed_alignment'][backbone]['auroc']
                        dynamic = data['dynamic_alignment'][backbone]['auroc']
                        imp = dynamic - fixed
                        improvements.append(imp)
                
                avg_improvement = np.mean(improvements) if improvements else 0
                resnet_imp = (data['dynamic_alignment']['resnet18']['auroc'] - 
                             data['fixed_alignment']['resnet18']['auroc']) if 'resnet18' in data['fixed_alignment'] else 0
                
                print(f"✅ {category:15s} - 平均提升: {avg_improvement:+.4f} "
                      f"(ResNet: {resnet_imp:+.4f})")
        else:
            print(f"❌ {category:15s} - 文件不存在")
    
    return results


def analyze_ablation_3(ablation_3_results):
    """分析Ablation 3结果"""
    print("\n" + "="*80)
    print("📈 Ablation 3 统计分析")
    print("="*80)
    
    # 统计每种损失函数的平均AUROC
    kd_only_aurocs = []
    kd_consistency_aurocs = []
    full_aurocs = []
    
    best_configs = {'kd_only': 0, 'kd_consistency': 0, 'full': 0}
    
    for category, data in ablation_3_results.items():
        kd_only_auroc = data['kd_only']['auroc']
        kd_consistency_auroc = data['kd_consistency']['auroc']
        full_auroc = data['full']['auroc']
        
        kd_only_aurocs.append(kd_only_auroc)
        kd_consistency_aurocs.append(kd_consistency_auroc)
        full_aurocs.append(full_auroc)
        
        # 统计最佳配置
        max_auroc = max(kd_only_auroc, kd_consistency_auroc, full_auroc)
        if kd_only_auroc == max_auroc:
            best_configs['kd_only'] += 1
        elif kd_consistency_auroc == max_auroc:
            best_configs['kd_consistency'] += 1
        else:
            best_configs['full'] += 1
    
    print(f"\n平均AUROC:")
    print(f"  kd_only:          {np.mean(kd_only_aurocs):.4f} ± {np.std(kd_only_aurocs):.4f}")
    print(f"  kd_consistency:   {np.mean(kd_consistency_aurocs):.4f} ± {np.std(kd_consistency_aurocs):.4f}")
    print(f"  full (InfoNCE):   {np.mean(full_aurocs):.4f} ± {np.std(full_aurocs):.4f}")
    
    print(f"\n最佳配置统计:")
    print(f"  kd_only:          {best_configs['kd_only']}/15 类别")
    print(f"  kd_consistency:   {best_configs['kd_consistency']}/15 类别")
    print(f"  full (InfoNCE):   {best_configs['full']}/15 类别")
    
    return {
        'avg_kd_only': np.mean(kd_only_aurocs),
        'avg_kd_consistency': np.mean(kd_consistency_aurocs),
        'avg_full': np.mean(full_aurocs),
        'best_configs': best_configs
    }


def analyze_ablation_4(ablation_4_results):
    """分析Ablation 4结果"""
    print("\n" + "="*80)
    print("📈 Ablation 4 统计分析")
    print("="*80)
    
    backbones = ['resnet18', 'efficientnet_b0', 'vit_tiny_patch16_224', 'convnextv2_tiny']
    
    # 统计每个backbone的提升
    backbone_improvements = {bb: [] for bb in backbones}
    category_avg_improvements = []
    
    positive_categories = 0
    negative_categories = 0
    
    for category, data in ablation_4_results.items():
        improvements = []
        
        for backbone in backbones:
            if backbone in data['fixed_alignment'] and backbone in data['dynamic_alignment']:
                fixed = data['fixed_alignment'][backbone]['auroc']
                dynamic = data['dynamic_alignment'][backbone]['auroc']
                imp = dynamic - fixed
                backbone_improvements[backbone].append(imp)
                improvements.append(imp)
        
        if improvements:
            avg_imp = np.mean(improvements)
            category_avg_improvements.append(avg_imp)
            
            if avg_imp > 0:
                positive_categories += 1
            else:
                negative_categories += 1
    
    print(f"\n各Backbone平均提升:")
    for backbone in backbones:
        imps = backbone_improvements[backbone]
        if imps:
            print(f"  {backbone:25s}: {np.mean(imps):+.4f} ± {np.std(imps):.4f} "
                  f"(范围: {min(imps):+.4f} ~ {max(imps):+.4f})")
    
    if category_avg_improvements:
        print(f"\n类别级别统计:")
        print(f"  平均提升:           {np.mean(category_avg_improvements):+.4f} ± {np.std(category_avg_improvements):.4f}")
        print(f"  正提升类别:         {positive_categories}/15")
        print(f"  负提升类别:         {negative_categories}/15")
        print(f"  最大提升:           {max(category_avg_improvements):+.4f}")
        print(f"  最大下降:           {min(category_avg_improvements):+.4f}")
    else:
        print(f"\n⚠️  警告: 没有有效的数据")
    
    return {
        'backbone_avg_improvements': {bb: np.mean(imps) for bb, imps in backbone_improvements.items() if imps},
        'category_avg_improvement': np.mean(category_avg_improvements) if category_avg_improvements else 0,
        'positive_categories': positive_categories,
        'negative_categories': negative_categories
    }


def generate_summary_tables(ablation_3_stats, ablation_4_stats):
    """生成论文用的汇总表格"""
    print("\n" + "="*80)
    print("📋 论文表格数据")
    print("="*80)
    
    # Table: Ablation 3 Summary
    print("\n【Table X】Ablation Study on Loss Function Components")
    print("-"*80)
    print(f"{'Loss Configuration':25s} {'Avg AUROC':>12s} {'Std':>8s} {'Best Count':>12s}")
    print("-"*80)
    print(f"{'kd_only':25s} {ablation_3_stats['avg_kd_only']:12.4f} {'-':>8s} "
          f"{ablation_3_stats['best_configs']['kd_only']:12d}")
    print(f"{'kd_consistency':25s} {ablation_3_stats['avg_kd_consistency']:12.4f} {'-':>8s} "
          f"{ablation_3_stats['best_configs']['kd_consistency']:12d}")
    print(f"{'full (InfoNCE, Ours)':25s} {ablation_3_stats['avg_full']:12.4f} {'-':>8s} "
          f"{ablation_3_stats['best_configs']['full']:12d}")
    
    # Table: Ablation 4 Summary
    print("\n【Table Y】Ablation Study on Dynamic Dimension Alignment")
    print("-"*80)
    print(f"{'Backbone':25s} {'Avg Improvement':>16s} {'Positive':>10s} {'Negative':>10s}")
    print("-"*80)
    
    for backbone, avg_imp in ablation_4_stats['backbone_avg_improvements'].items():
        # 简化的正负统计（实际需要遍历所有类别）
        print(f"{backbone:25s} {avg_imp:+16.4f} {'-':>10s} {'-':>10s}")
    
    print("-"*80)
    print(f"{'Overall Average':25s} {ablation_4_stats['category_avg_improvement']:+16.4f}")
    print(f"{'Positive Categories':25s} {ablation_4_stats['positive_categories']:16d}/15")
    print(f"{'Negative Categories':25s} {ablation_4_stats['negative_categories']:16d}/15")


def save_results_to_csv(ablation_3_results, ablation_4_results):
    """保存结果到CSV文件"""
    print("\n" + "="*80)
    print("💾 保存结果到CSV")
    print("="*80)
    
    # Ablation 3 CSV
    ablation_3_data = []
    for category, data in ablation_3_results.items():
        ablation_3_data.append({
            'Category': category,
            'kd_only_AUROC': data['kd_only']['auroc'],
            'kd_only_AP': data['kd_only']['ap'],
            'kd_consistency_AUROC': data['kd_consistency']['auroc'],
            'kd_consistency_AP': data['kd_consistency']['ap'],
            'full_AUROC': data['full']['auroc'],
            'full_AP': data['full']['ap']
        })
    
    df_ablation_3 = pd.DataFrame(ablation_3_data)
    csv_path_3 = Path("experiments/P0_ablation_results.csv")
    df_ablation_3.to_csv(csv_path_3, index=False)
    print(f"✅ Ablation 3结果已保存到: {csv_path_3}")
    
    # Ablation 4 CSV
    ablation_4_data = []
    backbones = ['resnet18', 'efficientnet_b0', 'vit_tiny_patch16_224', 'convnextv2_tiny']
    
    for category, data in ablation_4_results.items():
        row = {'Category': category}
        for backbone in backbones:
            if backbone in data['fixed_alignment'] and backbone in data['dynamic_alignment']:
                fixed = data['fixed_alignment'][backbone]['auroc']
                dynamic = data['dynamic_alignment'][backbone]['auroc']
                row[f'{backbone}_Fixed'] = fixed
                row[f'{backbone}_Dynamic'] = dynamic
                row[f'{backbone}_Improvement'] = dynamic - fixed
        ablation_4_data.append(row)
    
    df_ablation_4 = pd.DataFrame(ablation_4_data)
    csv_path_4 = Path("experiments/P0_ablation_4_results.csv")
    df_ablation_4.to_csv(csv_path_4, index=False)
    print(f"✅ Ablation 4结果已保存到: {csv_path_4}")


def main():
    print("\n" + "="*80)
    print("🚀 P0消融实验结果收集与分析")
    print("="*80)
    
    # 收集结果
    ablation_3_results = collect_ablation_3_results()
    ablation_4_results = collect_ablation_4_results()
    
    # 分析结果
    ablation_3_stats = analyze_ablation_3(ablation_3_results)
    ablation_4_stats = analyze_ablation_4(ablation_4_results)
    
    # 生成表格
    generate_summary_tables(ablation_3_stats, ablation_4_stats)
    
    # 保存到CSV
    save_results_to_csv(ablation_3_results, ablation_4_results)
    
    print("\n" + "="*80)
    print("🎉 分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()
