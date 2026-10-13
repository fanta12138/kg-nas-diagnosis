#!/usr/bin/env python3
"""
实验结果汇总分析
"""

import json
from pathlib import Path
import pandas as pd


def collect_results():
    """收集所有实验结果"""
    results_dir = Path('experiments/baselines_full')
    all_results = []
    
    for category_dir in sorted(results_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        
        for method_dir in sorted(category_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            
            method = method_dir.name
            results_file = method_dir / 'results.json'
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    metrics = json.load(f)
                
                all_results.append({
                    'Category': category,
                    'Method': method,
                    'AUROC': metrics.get('auroc', 0),
                    'AP': metrics.get('ap', 0),
                    'Inference_Time': metrics.get('inference_time', 0)
                })
    
    return pd.DataFrame(all_results)


def print_summary(df):
    """打印汇总统计"""
    print("="*80)
    print("📊 MVTec-AD 基线实验结果汇总")
    print("="*80)
    print(f"\n总实验数: {len(df)}")
    print(f"类别数: {df['Category'].nunique()}")
    print(f"方法数: {df['Method'].nunique()}")
    
    # 按方法统计平均性能
    print("\n" + "="*80)
    print("🏆 各方法平均性能 (AUROC)")
    print("="*80)
    
    method_stats = df.groupby('Method')['AUROC'].agg(['mean', 'std', 'min', 'max'])
    method_stats = method_stats.sort_values('mean', ascending=False)
    
    print(f"\n{'Method':<25} {'Mean AUROC':>12} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-"*80)
    
    for method, stats in method_stats.iterrows():
        print(f"{method:<25} {stats['mean']:>12.4f} {stats['std']:>8.4f} {stats['min']:>8.4f} {stats['max']:>8.4f}")
    
    # 按类别统计
    print("\n" + "="*80)
    print("📦 各类别最佳方法")
    print("="*80)
    
    print(f"\n{'Category':<15} {'Best Method':<25} {'AUROC':>8}")
    print("-"*80)
    
    for category in sorted(df['Category'].unique()):
        cat_df = df[df['Category'] == category]
        best_idx = cat_df['AUROC'].idxmax()
        best_row = cat_df.loc[best_idx]
        print(f"{category:<15} {best_row['Method']:<25} {best_row['AUROC']:>8.4f}")
    
    # Top 5 最佳组合
    print("\n" + "="*80)
    print("🥇 Top 5 最佳实验组合")
    print("="*80)
    
    top5 = df.nlargest(5, 'AUROC')
    print(f"\n{'Rank':<6} {'Category':<15} {'Method':<25} {'AUROC':>8} {'AP':>8}")
    print("-"*80)
    
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"{rank:<6} {row['Category']:<15} {row['Method']:<25} {row['AUROC']:>8.4f} {row['AP']:>8.4f}")
    
    # 保存完整表格
    output_file = Path('experiments/baselines_full/RESULTS_SUMMARY.csv')
    df.to_csv(output_file, index=False)
    print(f"\n✅ 完整结果已保存到: {output_file}")
    
    return method_stats


if __name__ == '__main__':
    df = collect_results()
    
    if len(df) == 0:
        print("❌ 未找到任何实验结果")
    else:
        stats = print_summary(df)
