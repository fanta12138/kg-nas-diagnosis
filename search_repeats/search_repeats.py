# -*- coding: utf-8 -*-
"""Does the architecture search select better backbones than chance?

The manuscript's search draws T configurations uniformly at random from the
42-cell space (`NASSearchEngine.sample_architecture` uses `random.choice`) and
returns the argmin of a 3-epoch knowledge-distillation proxy. This script runs
that exact procedure K times with independent seeds and records what it picks.

Because the searched `feature_dim` cannot reach the deployed detector
(StudentEncoder hard-codes a 256-d bottleneck), the deployed model is determined
solely by the selected backbone. We therefore look each selected backbone's TRUE
mean AUROC up in proxy_diag/phaseB_truth.json and compare the search's outcome
against uniform random selection and against the per-backbone oracle.

Usage:
    python search_repeats.py --runs 12 --trials 100        # ~10 h on RTX 3090
    python search_repeats.py --analyse-only                # no GPU
"""
# --- repository bootstrap (added for the public release) ---
import os as _os, sys as _sys
_KGD_P = _os.path.abspath(__file__)
while _KGD_P != _os.path.dirname(_KGD_P) and not _os.path.exists(
        _os.path.join(_KGD_P, 'kgd_paths.py')):
    _KGD_P = _os.path.dirname(_KGD_P)
if _KGD_P not in _sys.path:
    _sys.path.insert(0, _KGD_P)
from kgd_paths import (REPO_ROOT as _KGD_ROOT, DATASETS_ROOT as _KGD_DATA,
                       PAPER_ROOT as _KGD_PAPER)
# --- end bootstrap ------------------------------------------------------------

import argparse, collections, io, json, os, random, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

OUT = _os.path.join(_KGD_ROOT, 'search_repeats')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'search_repeats.json')
SUMFILE = os.path.join(OUT, 'search_repeats_summary.json')
TRUTH = _os.path.join(_KGD_ROOT, 'proxy_diag', 'phaseB_truth.json')

BACKBONES = ['convnextv2_tiny', 'convnextv2_base', 'mobileone_s0', 'convnext_tiny',
             'convnext_small', 'vit_tiny_patch16_224', 'deit_tiny_patch16_224',
             'swin_tiny_patch4_window7_224', 'efficientnet_b0', 'efficientnet_b1',
             'mobilenetv3_small_100', 'resnet18', 'resnet34', 'resnet50']
DIMS = [128, 256, 512]
SEARCH_CAT = 'wood'
QUICK_EPOCHS = 3


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def jload(p, d):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else d


def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_setup():
    from datasets import MVTecWoodDataset
    from models import ContrastiveModel
    device = torch.device('cuda')
    ck = torch.load('experiments/stage1/best_teacher.pth', map_location=device)
    teacher = ContrastiveModel(backbone_name=ck['config']['model']['backbone'],
                               projection_dim=ck['config']['model']['projection_dim'],
                               pretrained=False).to(device)
    teacher.load_state_dict(ck['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
    root = '../datasets/mvtec_anomaly_detection/' + SEARCH_CAT
    tr = MVTecWoodDataset(root=root, split='train', defect_type='good',
                          transform=tf, return_pair=False)
    va = MVTecWoodDataset(root=root, split='test', defect_type='good',
                          transform=tf, return_pair=False)
    return (device, teacher,
            DataLoader(tr, batch_size=16, shuffle=True, num_workers=0, pin_memory=True),
            DataLoader(va, batch_size=16, shuffle=False, num_workers=0, pin_memory=True))


def run_searches(runs, trials):
    from methods.nas_engine import NASSearchEngine
    device, teacher, tl, vl = build_setup()
    res = jload(RESFILE, {})
    for k in range(runs):
        key = 'run%02d' % k
        if key in res and res[key].get('done'):
            continue
        seed = 5000 + k
        set_seed(seed)
        t0 = time.time()
        eng = NASSearchEngine(search_space={'backbones': BACKBONES, 'feature_dims': DIMS},
                              teacher_model=teacher, device=device,
                              alpha=0.5, beta=0.3, gamma=0.2)
        best = eng.search(tl, vl, num_trials=trials, quick_epochs=QUICK_EPOCHS)
        rows = eng.get_all_results()
        res[key] = {'done': True, 'seed': seed, 'trials': trials,
                    'recorded': len(rows),
                    'selected': best, 'minutes': round((time.time() - t0) / 60, 1),
                    'all_results': rows}
        jdump(res, RESFILE)
        log('run%02d seed=%d -> %s@%d  (%d/%d trials recorded, %.1f min)'
            % (k, seed, best['backbone'], best['feature_dim'], len(rows), trials,
               res[key]['minutes']))
    return res


def analyse(res):
    truth_raw = jload(TRUTH, {})
    truth = collections.defaultdict(list)
    for v in truth_raw.values():
        a = v.get('auroc', float('nan'))
        if np.isfinite(a):
            truth[v['backbone']].append(a)
    truth = {k: float(np.mean(v)) for k, v in truth.items()}
    if not truth:
        log('no truth matrix yet (%s); run proxy_diag_v2.py --phase B first' % TRUTH)
        return

    done = [v for v in res.values() if v.get('done')]
    sel = [v['selected'] for v in done]
    sel_bb = [s['backbone'] for s in sel]
    sel_dim = [s['feature_dim'] for s in sel]

    known = [b for b in sel_bb if b in truth]
    sel_auroc = [truth[b] for b in known]
    rng = np.random.default_rng(0)
    pool = [truth[b] for b in BACKBONES if b in truth]
    boot = np.array([np.mean(rng.choice(pool, size=len(known), replace=True))
                     for _ in range(20000)]) if known else np.array([])
    best_bb = max(truth, key=lambda b: truth[b])

    out = {
        'n_runs': len(done),
        'trials_per_run': done[0]['trials'] if done else None,
        'selected_backbones': sel_bb,
        'selected_dims': sel_dim,
        'backbone_counts': dict(collections.Counter(sel_bb)),
        'dim_counts': dict(collections.Counter(sel_dim)),
        'unique_backbones_selected': len(set(sel_bb)),
        'unique_configs_selected': len(set((s['backbone'], s['feature_dim']) for s in sel)),
        'search_mean_true_auroc': float(np.mean(sel_auroc)) if sel_auroc else None,
        'random_mean_true_auroc': float(np.mean(pool)),
        'random_bootstrap_sd': float(boot.std(ddof=0)) if boot.size else None,
        'random_bootstrap_p95': float(np.percentile(boot, 95)) if boot.size else None,
        'oracle_backbone': best_bb,
        'oracle_true_auroc': float(truth[best_bb]),
        'p_search_leq_random_mean': float((boot <= np.mean(sel_auroc)).mean())
        if boot.size and sel_auroc else None,
        'truth_per_backbone': truth,
    }
    jdump(out, SUMFILE)
    log('')
    log('=' * 70)
    log('search runs                 : %d  (%d trials each)' % (len(done), out['trials_per_run']))
    log('distinct backbones selected : %d / %d' % (out['unique_backbones_selected'], len(BACKBONES)))
    log('dimension counts            : %s' % out['dim_counts'])
    log('TRUE mean AUROC of the search output : %.4f%%' % (100 * out['search_mean_true_auroc']))
    log('TRUE mean AUROC of uniform random    : %.4f%%  (bootstrap sd %.4f pp)'
        % (100 * out['random_mean_true_auroc'], 100 * out['random_bootstrap_sd']))
    log('oracle backbone in the space         : %s  %.4f%%' % (best_bb, 100 * out['oracle_true_auroc']))
    log('P(random mean <= search mean)        : %.3f' % out['p_search_leq_random_mean'])
    log('=' * 70)
    log('saved -> %s' % SUMFILE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=12)
    ap.add_argument('--trials', type=int, default=100)
    ap.add_argument('--analyse-only', action='store_true')
    a = ap.parse_args()
    res = jload(RESFILE, {})
    if not a.analyse_only:
        res = run_searches(a.runs, a.trials)
    analyse(res)


if __name__ == '__main__':
    main()
