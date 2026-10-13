#!/usr/bin/env python3
"""
在MVTec-AD所有15个类别上运行Stage 3最终训练
使用NAS搜索的最佳架构（EfficientNet-B0 + 256维）

输出:
- 每个类别的AUROC/AP/F1指标
- 平均性能统计
- 可用于Table 2和Table 3的数据
"""

import subprocess
import json
import csv
from pathlib import Path
from datetime import datetime


def run_stage3_on_all_categories():
    """在所有15个MVTec-AD类别上运行Stage 3"""
    
    # MVTec-AD所有类别
    categories = [
        'bottle', 'cable', 'capsule', 'carpet', 'grid',
        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
        'tile', 'toothbrush', 'transistor', 'wood', 'zipper'
    ]
    
    # 已完成的类别
    completed_categories = ['wood']
    
    # 待运行的类别
    remaining_categories = [cat for cat in categories if cat not in completed_categories]
    
    print(f"{'='*80}")
    print(f"🚀 在MVTec-AD所有类别上运行Stage 3")
    print(f"{'='*80}")
    print(f"总类别数: {len(categories)}")
    print(f"已完成: {len(completed_categories)} ({', '.join(completed_categories)})")
    print(f"待运行: {len(remaining_categories)}")
    print(f"\n使用架构: EfficientNet-B0 + Feature Dim 256")
    print(f"{'='*80}\n")
    
    # 结果收集
    results = []
    
    # Wood类别的结果（已完成）
    wood_result = {
        'category': 'wood',
        'status': 'completed',
        'auroc': 0.7535,
        'ap': 0.9194,
        'f1_max': None,  # 待提取
        'params_m': 6.0,
        'timestamp': datetime.now().isoformat()
    }
    results.append(wood_result)
    
    # 遍历剩余类别
    base_dir = Path(__file__).parent.parent
    data_base_dir = base_dir.parent / 'datasets' / 'mvtec_anomaly_detection'
    
    for i, category in enumerate(remaining_categories, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(remaining_categories)}] 处理类别: {category.upper()}")
        print(f"{'='*80}")
        
        # 构建命令
        output_dir = base_dir / 'experiments' / f'final_topjournal_{category}'
        
        cmd = [
            'python', str(base_dir / 'scripts' / 'stage3_final_training.py'),
            '--config', str(base_dir / 'configs' / 'final_training.yaml'),
            '--arch-config', str(base_dir / 'experiments' / 'stage2_topjournal_15backbones_fixed' / 'best_architecture.yaml'),
            '--data-dir', str(data_base_dir / category),
            '--output-dir', str(output_dir)
        ]
        
        print(f"命令: {' '.join(cmd)}")
        print(f"数据目录: {data_base_dir / category}")
        print(f"输出目录: {output_dir}")
        
        try:
            # 运行Stage 3
            result = subprocess.run(
                cmd,
                cwd=str(base_dir),
                capture_output=False,  # 显示实时输出
                timeout=3600  # 1小时超时
            )
            
            if result.returncode == 0:
                # 读取结果
                eval_file = output_dir / 'final_evaluation.json'
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    category_result = {
                        'category': category,
                        'status': 'success',
                        'auroc': eval_data.get('AUROC', None),
                        'ap': eval_data.get('AP', None),
                        'f1_max': eval_data.get('F1_Max', None),
                        'params_m': 6.0,
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(category_result)
                    
                    print(f"\n✅ {category} 完成!")
                    print(f"   AUROC: {category_result['auroc']:.4f}")
                    print(f"   AP: {category_result['ap']:.4f}")
                else:
                    print(f"\n⚠️  {category} 完成但未找到评估结果文件")
                    results.append({
                        'category': category,
                        'status': 'no_eval_file',
                        'auroc': None,
                        'ap': None,
                        'f1_max': None,
                        'params_m': 6.0,
                        'timestamp': datetime.now().isoformat()
                    })
            else:
                print(f"\n❌ {category} 失败 (返回码: {result.returncode})")
                results.append({
                    'category': category,
                    'status': 'failed',
                    'auroc': None,
                    'ap': None,
                    'f1_max': None,
                    'params_m': 6.0,
                    'error_code': result.returncode,
                    'timestamp': datetime.now().isoformat()
                })
        
        except subprocess.TimeoutExpired:
            print(f"\n❌ {category} 超时 (>1小时)")
            results.append({
                'category': category,
                'status': 'timeout',
                'auroc': None,
                'ap': None,
                'f1_max': None,
                'params_m': 6.0,
                'timestamp': datetime.now().isoformat()
            })
        
        except Exception as e:
            print(f"\n❌ {category} 异常: {e}")
            results.append({
                'category': category,
                'status': 'error',
                'auroc': None,
                'ap': None,
                'f1_max': None,
                'params_m': 6.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # 计算统计信息
    print(f"\n\n{'='*80}")
    print(f"📊 所有类别运行完成 - 统计摘要")
    print(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    
    if successful:
        avg_auroc = sum(r['auroc'] for r in successful if r['auroc']) / len([r for r in successful if r['auroc']])
        avg_ap = sum(r['ap'] for r in successful if r['ap']) / len([r for r in successful if r['ap']])
        
        print(f"\n成功类别: {len(successful)}/{len(categories)}")
        print(f"平均 AUROC: {avg_auroc:.4f}")
        print(f"平均 AP: {avg_ap:.4f}")
        
        print(f"\n各类别详细结果:")
        print(f"{'Category':<15} {'AUROC':<10} {'AP':<10} {'Status'}")
        print(f"{'-'*50}")
        for r in results:
            auroc_str = f"{r['auroc']:.4f}" if r['auroc'] else "N/A"
            ap_str = f"{r['ap']:.4f}" if r['ap'] else "N/A"
            print(f"{r['category']:<15} {auroc_str:<10} {ap_str:<10} {r['status']}")
    
    # 保存结果
    output_file = base_dir / 'experiments' / 'all_categories_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_categories': len(categories),
            'successful': len(successful),
            'average_auroc': avg_auroc if successful else None,
            'average_ap': avg_ap if successful else None,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存到: {output_file}")
    
    # 生成CSV格式（用于Origin和论文表格）
    csv_file = base_dir / 'experiments' / 'results_for_paper' / 'main_results' / 'table_3_per_category.csv'
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Category', 'AUROC', 'AP', 'F1_Max', 'Params_M', 'Status'])
        
        for r in results:
            writer.writerow([
                r['category'],
                r['auroc'] if r['auroc'] else '',
                r['ap'] if r['ap'] else '',
                r['f1_max'] if r['f1_max'] else '',
                r['params_m'],
                r['status']
            ])
    
    print(f"✅ CSV表格已保存到: {csv_file}")
    print(f"\n{'='*80}")
    print(f"🎉 全部完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    run_stage3_on_all_categories()
