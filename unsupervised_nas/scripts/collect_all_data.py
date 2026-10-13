#!/usr/bin/env python3
"""
收集所有论文所需的数据

功能:
1. 从experiments/目录读取所有实验结果
2. 计算Params和FLOPs
3. 测量Inference Time
4. 生成ROC/PR曲线数据
5. 汇总到paper_data/目录
"""

import json
import sys
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))


def collect_baseline_results():
    """收集所有基线的结果"""
    print("="*60)
    print("Step 1: 收集基线实验结果")
    print("="*60)
    
    baselines_dir = Path("experiments/baselines")
    results = []
    
    if not baselines_dir.exists():
        print("⚠️  警告: experiments/baselines目录不存在")
        return results
    
    for method_dir in sorted(baselines_dir.iterdir()):
        if method_dir.is_dir():
            results_file = method_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    result["source"] = str(results_file)
                    results.append(result)
                    print(f"  ✓ {result.get('method', 'Unknown')}: AUROC={result.get('auroc', 'N/A')}")
            else:
                print(f"  ⚠️  {method_dir.name}: 未找到results.json")
    
    print(f"\n共收集 {len(results)} 个基线结果\n")
    return results


def collect_final_result():
    """收集最终训练结果"""
    print("="*60)
    print("Step 2: 收集最终训练结果")
    print("="*60)
    
    final_result_file = Path("experiments/final/evaluation_results.json")
    
    if final_result_file.exists():
        with open(final_result_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            result["name"] = "Ours (NAS)"
            result["highlight"] = True
            result["source"] = str(final_result_file)
            print(f"  ✓ Ours (NAS): AUROC={result.get('auroc', 'N/A')}")
            return result
    else:
        print("  ⚠️  未找到最终训练结果")
        return None


def calculate_efficiency_metrics(model_path, config):
    """计算参数量和FLOPs"""
    print("="*60)
    print("Step 3: 计算效率指标")
    print("="*60)
    
    try:
        from thop import profile
        
        # 这里需要根据实际模型加载逻辑实现
        # 暂时返回placeholder
        print("  ⚠️  Params/FLOPs计算需要模型加载逻辑")
        print("  提示: 安装thop库: pip install thop")
        
        return {
            "params_m": None,
            "flops_g": None
        }
    
    except ImportError:
        print("  ⚠️  thop库未安装，跳过Params/FLOPs计算")
        print("  安装命令: pip install thop")
        return {
            "params_m": None,
            "flops_g": None
        }


def measure_inference_time(model, test_loader, num_runs=100):
    """测量推理时间"""
    print("="*60)
    print("Step 4: 测量推理时间")
    print("="*60)
    
    model.eval()
    device = next(model.parameters()).device
    
    # 预热
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            _ = model.compute_anomaly_scores(images[:1])
            break
    
    # 测量
    times = []
    with torch.no_grad():
        for images, _, _ in test_loader:
            images = images.to(device)
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model.compute_anomaly_scores(images[:1])
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            
            if len(times) >= num_runs:
                break
    
    avg_time = np.mean(times)
    print(f"  平均推理时间: {avg_time:.2f} ms/image")
    
    return avg_time


def generate_main_comparison_table(baseline_results, final_result):
    """生成主对比表格"""
    print("="*60)
    print("Step 5: 生成主对比表格")
    print("="*60)
    
    all_methods = baseline_results.copy()
    if final_result:
        all_methods.append(final_result)
    
    comparison_data = {
        "dataset": "MVTec-AD Wood",
        "metrics": ["auroc", "ap", "params_m", "flops_g", "inference_ms", "training_min"],
        "methods": []
    }
    
    for method in all_methods:
        method_data = {
            "name": method.get("method", method.get("name", "Unknown")),
            "year": method.get("year", None),
            "venue": method.get("conference", method.get("venue", "-")),
            "auroc": method.get("auroc", None),
            "ap": method.get("ap", None),
            "params_m": method.get("params_m", None),
            "flops_g": method.get("flops_g", None),
            "inference_ms": method.get("inference_ms", None),
            "training_min": method.get("training_min", None),
            "source": method.get("source", ""),
            "highlight": method.get("highlight", False)
        }
        comparison_data["methods"].append(method_data)
    
    # 保存
    output_path = Path("paper_data/main_comparison_table.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ 主对比表格已保存到: {output_path}")
    print(f"  共 {len(comparison_data['methods'])} 个方法")
    
    return comparison_data


def collect_all_data():
    """主函数：收集所有数据"""
    print("\n" + "="*60)
    print("📊 开始收集论文数据")
    print("="*60 + "\n")
    
    # Step 1: 收集基线结果
    baseline_results = collect_baseline_results()
    
    # Step 2: 收集最终结果
    final_result = collect_final_result()
    
    # Step 3: 计算效率指标（可选）
    # efficiency = calculate_efficiency_metrics(...)
    
    # Step 4: 生成主对比表格
    comparison_table = generate_main_comparison_table(baseline_results, final_result)
    
    # Step 5: 生成其他数据（ROC曲线、消融实验等）
    # 这些可以在后续实验中逐步补充
    
    print("\n" + "="*60)
    print("✅ 数据收集完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - paper_data/main_comparison_table.json")
    print("  - paper_data/dataset_stats.json (已有)")
    print("  - paper_data/implementation_details.json (已有)")
    print("\n待补充:")
    print("  - paper_data/ablation_study.json (运行消融实验后)")
    print("  - paper_data/roc_pr_curves.json (需要详细分数)")
    print("  - paper_data/visualizations/ (需要生成图像)")


if __name__ == "__main__":
    collect_all_data()
