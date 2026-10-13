#!/usr/bin/env python3
"""
收集所有15个类别的训练结果并生成统计报告
修复之前脚本的bug
"""

import json
import csv
from pathlib import Path
from datetime import datetime


def collect_all_results():
    """收集所有类别的结果"""
    
    base_dir = Path(__file__).parent.parent
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    print(f"{'='*80}")
    print(f"📊 收集所有15个类别的训练结果")
    print(f"{'='*80}\n")
    
    results = []
    
    for category in categories:
        # Wood类别的特殊路径
        if category == 'wood':
            eval_file = base_dir / 'experiments' / 'final_topjournal' / 'final_evaluation.json'
        else:
            eval_file = base_dir / 'experiments' / f'final_topjournal_{category}' / 'final_evaluation.json'
        
        if eval_file.exists():
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
            
            result = {
                'category': category,
                'status': 'success',
                'auroc': eval_data.get('auroc') or eval_data.get('AUROC'),
                'ap': eval_data.get('ap') or eval_data.get('AP'),
                'f1_max': eval_data.get('best_f1') or eval_data.get('F1_Max'),
                'params_m': 6.0,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            auroc_str = f"{result['auroc']:.4f}" if result['auroc'] else "N/A"
            ap_str = f"{result['ap']:.4f}" if result['ap'] else "N/A"
            print(f"✅ {category:<15} AUROC: {auroc_str:<10} AP: {ap_str}")
        else:
            print(f"❌ {category:<15} 评估文件不存在")
            results.append({
                'category': category,
                'status': 'missing_file',
                'auroc': None,
                'ap': None,
                'f1_max': None,
                'params_m': 6.0,
                'timestamp': datetime.now().isoformat()
            })
    
    # 计算统计信息
    print(f"\n{'='*80}")
    print(f"📈 统计摘要")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success' and r['auroc'] is not None]
    
    if successful:
        aurocs = [r['auroc'] for r in successful]
        aps = [r['ap'] for r in successful if r['ap'] is not None]
        
        avg_auroc = sum(aurocs) / len(aurocs)
        median_auroc = sorted(aurocs)[len(aurocs)//2]
        min_auroc = min(aurocs)
        max_auroc = max(aurocs)
        
        avg_ap = sum(aps) / len(aps) if aps else 0
        
        print(f"\n成功类别: {len(successful)}/{len(categories)}")
        print(f"\nAUROC统计:")
        print(f"  平均值: {avg_auroc:.4f}")
        print(f"  中位数: {median_auroc:.4f}")
        print(f"  最小值: {min_auroc:.4f}")
        print(f"  最大值: {max_auroc:.4f}")
        print(f"  标准差: {(sum((x - avg_auroc)**2 for x in aurocs) / len(aurocs))**0.5:.4f}")
        
        print(f"\nAP统计:")
        print(f"  平均值: {avg_ap:.4f}")
        
        # 按性能排序
        print(f"\n各类别详细结果（按AUROC排序）:")
        print(f"{'Rank':<5} {'Category':<15} {'AUROC':<10} {'AP':<10} {'Status'}")
        print(f"{'-'*55}")
        
        sorted_results = sorted(successful, key=lambda x: x['auroc'], reverse=True)
        for i, r in enumerate(sorted_results, 1):
            auroc_str = f"{r['auroc']:.4f}" if r['auroc'] else "N/A"
            ap_str = f"{r['ap']:.4f}" if r['ap'] else "N/A"
            print(f"{i:<5} {r['category']:<15} {auroc_str:<10} {ap_str:<10} {r['status']}")
    
    # 保存JSON结果
    output_json = base_dir / 'experiments' / 'all_categories_results.json'
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            'total_categories': len(categories),
            'successful': len(successful),
            'average_auroc': avg_auroc if successful else None,
            'median_auroc': median_auroc if successful else None,
            'average_ap': avg_ap if successful else None,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ JSON结果已保存到: {output_json}")
    
    # 生成CSV表格（用于论文Table 3）
    csv_dir = base_dir / 'experiments' / 'results_for_paper' / 'main_results'
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_file = csv_dir / 'table_3_per_category.csv'
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'AUROC', 'AP', 'F1_Max', 'Params_M', 'Status'])
        
        # 按字母顺序排序
        for r in sorted(results, key=lambda x: x['category']):
            writer.writerow([
                r['category'],
                f"{r['auroc']:.4f}" if r['auroc'] else '',
                f"{r['ap']:.4f}" if r['ap'] else '',
                f"{r['f1_max']:.4f}" if r['f1_max'] else '',
                r['params_m'],
                r['status']
            ])
    
    print(f"✅ CSV表格已保存到: {csv_file}")
    
    # 生成Table 2所需的数据（平均结果）
    if successful:
        table2_data = {
            'method': 'KG-NAS (Ours)',
            'avg_auroc': f"{avg_auroc:.4f}",
            'avg_ap': f"{avg_ap:.4f}",
            'params_m': 6.0,
            'year': 2024
        }
        
        table2_file = csv_dir / 'kg_nas_average_result.json'
        with open(table2_file, 'w', encoding='utf-8') as f:
            json.dump(table2_data, f, indent=2)
        
        print(f"✅ Table 2数据已保存到: {table2_file}")
    
    print(f"\n{'='*80}")
    print(f"🎉 结果收集完成！")
    print(f"{'='*80}")
    
    return results


if __name__ == '__main__':
    collect_all_results()
