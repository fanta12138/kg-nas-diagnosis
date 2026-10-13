#!/usr/bin/env python3
"""
P1 实验补强 - 主 runner（阶段1：Stage3 类实验）

三项实验串行执行（输出到 experiments/p1_*/）：
  A. Fixed-KD 基线对照：3 个固定架构 x 15 类别，200 epochs
     - resnet18+512d, swin_tiny+512d, mobilenetv3_small+512d（与主实验 512d 对齐）
     - 与 NAS 最佳（efficientnet_b0+512d）对比，证明搜索价值
  B. Seed 多跑：NAS 最佳架构 x 3 seeds x 15 类别，200 epochs
     - 报告 mean±std，回应统计批评
  C. 500 epochs 性能提升：NAS 最佳架构 x 15 类别
     - 验证更长训练是否提升 AUROC

用法: python scripts/p1_run_experiments.py [--only fixed_kd|seed|long]
"""
import subprocess
import json
import csv
import sys
import time
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_BASE = BASE_DIR.parent / 'datasets' / 'mvtec_anomaly_detection'
CONFIG = BASE_DIR / 'configs' / 'final_training.yaml'
NAS_ARCH = BASE_DIR / 'experiments' / 'stage2_topjournal_15backbones_fixed' / 'best_architecture.yaml'

CATEGORIES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

# 注意：真实主实验（final_topjournal_*）用的架构是 efficientnet_b0+512d
# （best_architecture.yaml 与 config.yaml 一致），论文中写的 256d 需在 P2 修正。
# Fixed-KD 对照统一用 512d 与主实验对齐，确保对比公平。
FIXED_KD_ARCHS = {
    'resnet18_512d': {'backbone': 'resnet18', 'feature_dim': 512},
    'swin_tiny_512d': {'backbone': 'swin_tiny_patch4_window7_224', 'feature_dim': 512},
    'mobilenetv3_small_512d': {'backbone': 'mobilenetv3_small_100', 'feature_dim': 512},
}

SEEDS = [42, 123, 2024]


def write_arch_yaml(path, backbone, feature_dim):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"best_architecture:\n  backbone: {backbone}\n  feature_dim: {feature_dim}\n")


def run_stage3(category, arch_yaml, output_dir, seed=42, epochs=200):
    # 若该类别已有完整结果则跳过（支持断点续跑）
    if (output_dir / 'final_evaluation.json').exists():
        with open(output_dir / 'final_evaluation.json') as f:
            d = json.load(f)
        print(f"  [skip] {category}: 已存在结果 AUROC={d['auroc']}")
        return {'category': category, 'rc': 0, 'auroc': d['auroc'], 'ap': d['ap'], 'minutes': 0, 'skipped': True}
    cmd = [
        sys.executable, str(BASE_DIR / 'scripts' / 'stage3_final_training.py'),
        '--config', str(CONFIG),
        '--arch-config', str(arch_yaml),
        '--data-dir', str(DATA_BASE / category),
        '--output-dir', str(output_dir),
        '--seed', str(seed),
        '--epochs', str(epochs),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR), capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    elapsed = (time.time() - t0) / 60
    auroc = ap = None
    eval_file = output_dir / 'final_evaluation.json'
    if eval_file.exists():
        with open(eval_file) as f:
            d = json.load(f)
        auroc, ap = d['auroc'], d['ap']
    elif result.returncode != 0:
        # 记录失败信息（截取 stderr 尾部）
        log_path = output_dir / 'error.log'
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f'rc={result.returncode}\n')
            f.write((result.stderr or '')[-2000:])
    return {'category': category, 'rc': result.returncode, 'auroc': auroc, 'ap': ap, 'minutes': round(elapsed, 1)}


def save_summary(results, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['category', 'auroc', 'ap', 'minutes', 'rc'])
        for r in results:
            w.writerow([r['category'], r['auroc'], r['ap'], r['minutes'], r['rc']])
    print(f"汇总已保存: {csv_path}")


def run_fixed_kd():
    print("=" * 70)
    print("[P1-A] Fixed-KD 基线对照: 3 架构 x 15 类别")
    print("=" * 70)
    all_rows = []
    for arch_name, arch in FIXED_KD_ARCHS.items():
        arch_yaml = BASE_DIR / 'experiments' / 'p1_archs' / f'{arch_name}.yaml'
        write_arch_yaml(arch_yaml, arch['backbone'], arch['feature_dim'])
        rows = []
        for i, cat in enumerate(CATEGORIES, 1):
            out = BASE_DIR / 'experiments' / 'p1_fixed_kd' / arch_name / cat
            r = run_stage3(cat, arch_yaml, out, seed=42, epochs=200)
            rows.append(r)
            all_rows.append({**r, 'arch': arch_name})
            print(f"  [{arch_name}][{i}/{len(CATEGORIES)}] {cat}: AUROC={r['auroc']} ({r['minutes']}min)")
        save_summary(rows, BASE_DIR / 'experiments' / 'p1_fixed_kd' / f'{arch_name}_summary.csv')
    with open(BASE_DIR / 'experiments' / 'p1_fixed_kd' / 'all_summary.json', 'w') as f:
        json.dump(all_rows, f, indent=2)


def run_seed_multi():
    print("=" * 70)
    print("[P1-B] Seed 多跑: NAS 最佳架构 x 3 seeds x 15 类别")
    print("=" * 70)
    all_rows = []
    for seed in SEEDS:
        rows = []
        for i, cat in enumerate(CATEGORIES, 1):
            out = BASE_DIR / 'experiments' / 'p1_seed' / f'seed{seed}' / cat
            r = run_stage3(cat, NAS_ARCH, out, seed=seed, epochs=200)
            rows.append(r)
            all_rows.append({**r, 'seed': seed})
            print(f"  [seed{seed}][{i}/{len(CATEGORIES)}] {cat}: AUROC={r['auroc']} ({r['minutes']}min)")
        save_summary(rows, BASE_DIR / 'experiments' / 'p1_seed' / f'seed{seed}_summary.csv')
    with open(BASE_DIR / 'experiments' / 'p1_seed' / 'all_summary.json', 'w') as f:
        json.dump(all_rows, f, indent=2)


def run_long_training():
    print("=" * 70)
    print("[P1-C] 500 epochs 性能提升: NAS 最佳架构 x 15 类别")
    print("=" * 70)
    rows = []
    for i, cat in enumerate(CATEGORIES, 1):
        out = BASE_DIR / 'experiments' / 'p1_long500' / cat
        r = run_stage3(cat, NAS_ARCH, out, seed=42, epochs=500)
        rows.append(r)
        print(f"  [{i}/{len(CATEGORIES)}] {cat}: AUROC={r['auroc']} ({r['minutes']}min)")
    save_summary(rows, BASE_DIR / 'experiments' / 'p1_long500' / 'summary.csv')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--only', choices=['fixed_kd', 'seed', 'long'], default=None,
                        help='只运行指定实验')
    args = parser.parse_args()

    t_start = datetime.now()
    print(f"P1 实验启动: {t_start.isoformat()}")
    print(f"数据目录: {DATA_BASE}")

    if args.only in (None, 'fixed_kd'):
        run_fixed_kd()
    if args.only in (None, 'seed'):
        run_seed_multi()
    if args.only in (None, 'long'):
        run_long_training()

    print(f"\nP1 全部完成。开始: {t_start.isoformat()}  结束: {datetime.now().isoformat()}")
