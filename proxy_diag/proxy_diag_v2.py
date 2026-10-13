# -*- coding: utf-8 -*-
"""Proxy reliability diagnostic for the KG-NAS architecture search (v2).

Why v2: v1 replicated the wrong signal.  It used the RD reconstruction loss as
the "proxy", but the search actually ranks architectures with

    NASSearchEngine.evaluate_architecture  ->  score = kd_loss + 0.3 * consistency_loss

computed with a *ContrastiveModel* student on MVTec wood normal data only.
v2 imports and calls that exact method so the proxy is byte-faithful.

Structure of the study
----------------------
Phase A  proxy repeatability
         For each of the 42 = 14 backbones x 3 feature_dims configurations in the
         manuscript's search space, run the real proxy R times with distinct
         seeds.  This makes the within-configuration noise observable.

Phase B  ground truth
         For each of the 14 backbones run the manuscript's own final training
         protocol (p2_rd_train.train_category, 500 epochs, bs 64, seed 42) on all
         15 MVTec categories.  Gives the TRUE image-level AUROC.

Phase C  analysis
         * within-configuration CV of the proxy
         * Spearman rho between proxy and truth (42 configs, and 14 backbones)
         * variance decomposition of the proxy into backbone / dim / residual
         * selection stability under bootstrap resampling of search trials
         * whether the search-selected backbone is actually the best backbone
         * structural probe: the searched feature_dim is inert in deployment

Usage
-----
    python proxy_diag_v2.py --phase A          # ~2 h
    python proxy_diag_v2.py --phase B          # ~20 h
    python proxy_diag_v2.py --phase C          # instant
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
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np
import torch
import importlib.util
from torch.utils.data import DataLoader
from torchvision import transforms

OUT = _os.path.join(_KGD_ROOT, 'proxy_diag')
os.makedirs(OUT, exist_ok=True)
FILE_A = os.path.join(OUT, 'phaseA_proxy.json')
FILE_B = os.path.join(OUT, 'phaseB_truth.json')
FILE_C = os.path.join(OUT, 'phaseC_analysis.json')

# ---------------------------------------------------------------- search space
# Exactly the configuration of experiments/stage2_topjournal_15backbones_fixed
# (the run whose best_architecture the manuscript reports).
BACKBONES = ['convnextv2_tiny', 'convnextv2_base', 'mobileone_s0', 'convnext_tiny',
             'convnext_small', 'vit_tiny_patch16_224', 'deit_tiny_patch16_224',
             'swin_tiny_patch4_window7_224', 'efficientnet_b0', 'efficientnet_b1',
             'mobilenetv3_small_100', 'resnet18', 'resnet34', 'resnet50']
DIMS = [128, 256, 512]
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
SEARCH_CAT = 'wood'          # the search ran on wood only
QUICK_EPOCHS = 3             # configs/stage2_nas_search*.yaml: quick_train_epochs
SEARCH_BS = 16               # configs/stage2_nas_search.yaml: batch_size
FINAL_EPOCHS = 500           # manuscript protocol
FINAL_SEED = 42


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jload(p, d):
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else d


def jdump(o, p):
    json.dump(o, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)


# --------------------------------------------------------------------- loaders
def build_loaders(data_dir, img_size=224, bs=SEARCH_BS):
    """Byte-faithful copy of stage2_nas_search.build_dataloaders (workers=0 for
    determinism -- the wood split is tiny)."""
    from datasets import MVTecWoodDataset
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tr = MVTecWoodDataset(root=data_dir, split='train', defect_type='good',
                          transform=tf, return_pair=False)
    va = MVTecWoodDataset(root=data_dir, split='test', defect_type='good',
                          transform=tf, return_pair=False)
    return (DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True),
            DataLoader(va, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True))


def load_teacher(device):
    from models import ContrastiveModel
    ck = torch.load('experiments/stage1/best_teacher.pth', map_location=device)
    m = ContrastiveModel(backbone_name=ck['config']['model']['backbone'],
                         projection_dim=ck['config']['model']['projection_dim'],
                         pretrained=False).to(device)
    m.load_state_dict(ck['model_state_dict'])
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


# ------------------------------------------------------------------ phase A
def phase_a(repeats):
    from methods.nas_engine import NASSearchEngine
    device = torch.device('cuda')
    log('phase A: loading teacher + wood loaders')
    teacher = load_teacher(device)
    tl, vl = build_loaders('../datasets/mvtec_anomaly_detection/wood')
    eng = NASSearchEngine(search_space={'backbones': BACKBONES, 'feature_dims': DIMS},
                          teacher_model=teacher, device=device,
                          alpha=0.5, beta=0.3, gamma=0.2)
    res = jload(FILE_A, {})
    n = len(BACKBONES) * len(DIMS)
    i = 0
    for bb in BACKBONES:
        for dim in DIMS:
            i += 1
            key = '%s|%d' % (bb, dim)
            have = res.get(key, {}).get('scores', [])
            if len(have) >= repeats:
                continue
            scores = list(have)
            for r in range(len(scores), repeats):
                seed = 7000 + 17 * r
                set_seed(seed)
                t0 = time.time()
                try:
                    s = eng.evaluate_architecture(
                        {'backbone': bb, 'feature_dim': int(dim)}, tl, vl,
                        quick_epochs=QUICK_EPOCHS)
                    scores.append(float(s))
                except Exception as e:
                    log('   FAIL %s: %s' % (key, str(e)[:70]))
                    scores.append(float('nan'))
                res[key] = {'backbone': bb, 'dim': int(dim), 'scores': scores,
                            'seconds': round(time.time() - t0, 1)}
                jdump(res, FILE_A)
            a = np.array([x for x in scores if np.isfinite(x)])
            if len(a):
                log('  [%2d/%d] %-30s d=%3d  mean=%.6g  std=%.6g  CV=%.1f%%  (n=%d)'
                    % (i, n, bb, dim, a.mean(), a.std(ddof=0),
                       100 * a.std(ddof=0) / abs(a.mean()) if a.mean() else float('nan'),
                       len(a)))
    log('phase A done -> %s' % FILE_A)


# ------------------------------------------------------------------ phase B
def phase_b(cats, backbones, epochs, seeds=(FINAL_SEED,)):
    """Ground-truth AUROC per (category, backbone, seed).

    Uses train_faithful.train_category_safe rather than p2_rd_train.train_category:
    the manuscript's batch size of 64 exceeds device memory for the largest
    backbones in the search space, and on Windows WDDM the overflow is silently
    paged to host RAM instead of raising, which makes a single run ~80x slower.
    train_category_safe keeps the effective batch size at 64 via gradient
    accumulation and records the micro-batch it used.
    """
    _s = importlib.util.spec_from_file_location(
        'tf', os.path.join(OUT, 'train_faithful.py'))
    tf = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(tf)
    device = torch.device('cuda')
    res = jload(FILE_B, {})
    total = len(cats) * len(backbones) * len(seeds)
    i = 0
    for seed in seeds:
        for cat in cats:
            for bb in backbones:
                i += 1
                key = '%s|%s|%d' % (cat, bb, seed)
                if key in res and np.isfinite(res[key].get('auroc', float('nan'))):
                    continue
                outdir = os.path.join(OUT, 'true', 's%d' % seed, cat, bb)
                t0 = time.time()
                try:
                    rec = tf.train_category_safe(cat, bb, epochs, seed, outdir, device,
                                                 verbose=True, log_every=50)
                    res[key] = {'category': cat, 'backbone': bb, 'seed': seed,
                                'auroc': float(rec['auroc']), 'ap': float(rec['ap']),
                                'epochs': epochs, 'minutes': rec['minutes'],
                                'micro_batch': rec.get('micro_batch'),
                                'accum_steps': rec.get('accum_steps'),
                                'probe_peak_gb': rec.get('probe_peak_gb')}
                    log('  [%4d/%d] s%d %-11s %-30s AUROC=%.4f (%.1f min, mb=%s x%s, peak=%.1fGB)'
                        % (i, total, seed, cat, bb, rec['auroc'], rec['minutes'],
                           rec.get('micro_batch'), rec.get('accum_steps'),
                           rec.get('probe_peak_gb') or 0))
                except Exception as e:
                    log('  [%4d/%d] s%d %-11s %-30s FAIL: %s'
                        % (i, total, seed, cat, bb, str(e)[:70]))
                    res[key] = {'category': cat, 'backbone': bb, 'seed': seed,
                                'auroc': float('nan'), 'error': str(e)[:200]}
                    torch.cuda.empty_cache()
                jdump(res, FILE_B)
    log('phase B done -> %s' % FILE_B)


# ------------------------------------------------------------------ phase C
def structural_probe():
    """Show, without training, that the searched feature_dim cannot reach the
    deployed detector: StudentEncoder.forward hard-codes a 256-d bottleneck."""
    from methods.rd_detector import RDDetector
    device = torch.device('cuda')
    out = {}
    d = RDDetector(student_backbone='efficientnet_b0', teacher_arch='resnet18').to(device)
    d.eval()
    with torch.no_grad():
        d.encoder(torch.zeros(1, 3, 224, 224, device=device))
    out['encoder_bottleneck_out_channels'] = int(d.encoder.bottleneck[0].out_channels)
    out['decoder_first_conv_in_channels'] = int(d.decoder.dec3[0].in_channels)
    out['rddetector_signature_has_bottleneck_dim'] = 'bottleneck_dim' in \
        RDDetector.__init__.__code__.co_varnames
    # decoder DOES honour the argument -> encoder/decoder disagree for dim != 256
    d128 = RDDetector(student_backbone='efficientnet_b0', teacher_arch='resnet18',
                      bottleneck_dim=128).to(device)
    d128.eval()
    with torch.no_grad():
        d128.encoder(torch.zeros(1, 3, 224, 224, device=device))
    out['dim128_encoder_out_channels'] = int(d128.encoder.bottleneck[0].out_channels)
    out['dim128_decoder_in_channels'] = int(d128.decoder.dec3[0].in_channels)
    out['verdict'] = ('encoder bottleneck is hard-coded to 256 while the decoder '
                      'honours bottleneck_dim; p2_rd_train.py never passes the '
                      'argument, so all three searched dims map to the identical '
                      'deployed detector.')
    return out


def phase_c(repeats_expected):
    from scipy.stats import spearmanr, friedmanchisquare
    A = jload(FILE_A, {})
    B = jload(FILE_B, {})
    out = {'structural_probe': structural_probe()}

    # ---- per-config proxy statistics
    rows = []
    for bb in BACKBONES:
        for dim in DIMS:
            s = np.array([x for x in A.get('%s|%d' % (bb, dim), {}).get('scores', [])
                          if np.isfinite(x)], dtype=float)
            if len(s) == 0:
                continue
            rows.append({'backbone': bb, 'dim': dim, 'n': int(len(s)),
                         'mean': float(s.mean()), 'std': float(s.std(ddof=0)),
                         'min': float(s.min()), 'max': float(s.max()),
                         'cv': float(s.std(ddof=0) / abs(s.mean())) if s.mean() else None})
    out['per_config'] = rows

    ok = [r for r in rows if r['mean'] is not None and np.isfinite(r['mean'])]
    if ok:
        cvs = np.array([r['cv'] for r in ok if r['cv'] is not None])
        out['cv_summary'] = {'n_configs': len(ok), 'cv_mean': float(cvs.mean()),
                             'cv_median': float(np.median(cvs)),
                             'cv_min': float(cvs.min()), 'cv_max': float(cvs.max()),
                             'configs_with_cv_gt_50pct': int((cvs > 0.5).sum()),
                             'configs_with_cv_gt_100pct': int((cvs > 1.0).sum())}

    # ---- variance decomposition of the proxy: backbone vs dim vs residual
    means = {r['backbone'] + '|' + str(r['dim']): r['mean'] for r in ok}
    if len(means) == len(BACKBONES) * len(DIMS):
        M = np.array([[means[bb + '|' + str(d)] for d in DIMS] for bb in BACKBONES])
        gm = M.mean()
        ss_bb = len(DIMS) * ((M.mean(axis=1) - gm) ** 2).sum()
        ss_dim = len(BACKBONES) * ((M.mean(axis=0) - gm) ** 2).sum()
        ss_res = ((M - M.mean(axis=1, keepdims=True)
                   - M.mean(axis=0, keepdims=True) + gm) ** 2).sum()
        tot = ss_bb + ss_dim + ss_res
        out['variance_decomposition_proxy'] = {
            'eta2_backbone': float(ss_bb / tot), 'eta2_dim': float(ss_dim / tot),
            'eta2_residual': float(ss_res / tot),
            'note': 'computed on the mean log10 proxy of each configuration cell'}
        # same on log scale (scores span orders of magnitude)
        with np.errstate(divide='ignore'):
            L = np.log10(np.clip(M, 1e-12, None))
        gm = L.mean()
        ss_bb = len(DIMS) * ((L.mean(axis=1) - gm) ** 2).sum()
        ss_dim = len(BACKBONES) * ((L.mean(axis=0) - gm) ** 2).sum()
        ss_res = ((L - L.mean(axis=1, keepdims=True)
                   - L.mean(axis=0, keepdims=True) + gm) ** 2).sum()
        tot = ss_bb + ss_dim + ss_res
        out['variance_decomposition_proxy_log10'] = {
            'eta2_backbone': float(ss_bb / tot), 'eta2_dim': float(ss_dim / tot),
            'eta2_residual': float(ss_res / tot)}

    # ---- truth: aggregate over seeds -----------------------------------------
    if B:
        truth = {}
        for bb in BACKBONES:
            rows = [v for v in B.values()
                    if v.get('backbone') == bb and np.isfinite(v.get('auroc', np.nan))]
            if not rows:
                continue
            percat = collections.defaultdict(list)
            for r in rows:
                percat[r['category']].append(r['auroc'])
            truth[bb] = {
                'mean_auroc': float(np.mean([r['auroc'] for r in rows])),
                'n_obs': len(rows), 'n_cats': len(percat),
                'n_seeds': len({r['seed'] for r in rows}),
                'auroc_wood': float(np.mean(percat.get(SEARCH_CAT, [float('nan')]))),
                'per_category': {c: float(np.mean(v)) for c, v in sorted(percat.items())},
            }
        out['truth_per_backbone'] = truth

        if len(truth) >= 3:
            order = [bb for bb in BACKBONES if bb in truth]
            tmean = np.array([truth[bb]['mean_auroc'] for bb in order])
            twood = np.array([truth[bb]['auroc_wood'] for bb in order])
            if len(means) == len(BACKBONES) * len(DIMS):
                pmean = np.array([np.mean([means[bb + '|' + str(d)] for d in DIMS])
                                  for bb in order])
                out['rank_correlation'] = {
                    'n_backbones': len(order),
                    'spearman_proxy_vs_mean_auroc': [float(x) for x in spearmanr(pmean, tmean)],
                    'spearman_proxy_vs_wood_auroc': [float(x) for x in spearmanr(pmean, twood)]}
            if np.isfinite(twood).all():
                rho, p = spearmanr(twood, tmean)
            else:
                rho, p = float('nan'), float('nan')
            out['wood_vs_overall_transfer'] = {'spearman': float(rho), 'p': float(p)}
            best_true = order[int(np.argmax(tmean))]
            wood_best = order[int(np.argmax(twood))] if np.isfinite(twood).all() else None
            out['selection_consequence'] = {
                'best_backbone_true_mean': best_true,
                'best_backbone_true_mean_auroc': float(tmean.max()),
                'spread_across_backbones_pp': float(100 * (tmean.max() - tmean.min())),
                'best_backbone_on_wood': wood_best,
                'manuscript_selected': 'efficientnet_b0',
                'rank_of_manuscript_selected': int(
                    (tmean > tmean[order.index('efficientnet_b0')]).sum()) + 1
                if 'efficientnet_b0' in order else None,
                'gap_selected_to_best_pp': float(100 * (tmean.max()
                    - tmean[order.index('efficientnet_b0')]))
                if 'efficientnet_b0' in order else None}

            # ---- variance decomposition of the TRUE metric -------------------
            # Y[backbone, category], averaged over seeds.
            cells = collections.defaultdict(list)
            for v in B.values():
                a = v.get('auroc', float('nan'))
                if np.isfinite(a):
                    cells[(v['backbone'], v['category'])].append(a)
            bbs = sorted({k[0] for k in cells})
            ccs = sorted({k[1] for k in cells})
            if len(bbs) >= 3 and len(ccs) >= 3:
                Y = np.full((len(bbs), len(ccs)), np.nan)
                for (bb, cc), vals in cells.items():
                    Y[bbs.index(bb)][ccs.index(cc)] = float(np.mean(vals))
                full = np.isfinite(Y)
                row_ok = full.all(axis=1)
                if row_ok.sum() >= 3:
                    Z = Y[row_ok]
                    gm = Z.mean()
                    ss_bb = Z.shape[1] * ((Z.mean(axis=1) - gm) ** 2).sum()
                    ss_cc = Z.shape[0] * ((Z.mean(axis=0) - gm) ** 2).sum()
                    ss_r = ((Z - Z.mean(axis=1, keepdims=True)
                             - Z.mean(axis=0, keepdims=True) + gm) ** 2).sum()
                    tot = ss_bb + ss_cc + ss_r
                    # seed-level noise: spread within (backbone, category) cells
                    seedsp = [float(np.std(v, ddof=0)) for v in cells.values() if len(v) > 1]
                    out['variance_decomposition_truth'] = {
                        'n_backbones': int(Z.shape[0]), 'n_categories': int(Z.shape[1]),
                        'eta2_backbone': float(ss_bb / tot),
                        'eta2_category': float(ss_cc / tot),
                        'eta2_residual': float(ss_r / tot),
                        'mean_within_cell_seed_sd': float(np.mean(seedsp)) if seedsp else None,
                        'n_cells_with_multiple_seeds': len(seedsp),
                        'f_backbone': float((ss_bb / (Z.shape[0] - 1)) /
                                            (ss_r / ((Z.shape[0] - 1) * (Z.shape[1] - 1))))
                        if ss_r > 0 else None,
                    }

    # ---- selection stability under bootstrap resampling of search trials
    if len(means) == len(BACKBONES) * len(DIMS):
        rng = np.random.default_rng(0)
        draws = min(r['n'] for r in ok)
        cells = list(means)
        dist = {c: np.array([x for x in A[c.replace('|', '|')]['scores']
                             if np.isfinite(x)]) for c in cells}
        # cells keyed differently; rebuild
        dist = {}
        for bb in BACKBONES:
            for d in DIMS:
                k = '%s|%d' % (bb, d)
                v = np.array([x for x in A.get(k, {}).get('scores', []) if np.isfinite(x)])
                if len(v):
                    dist[k] = v
        cells = sorted(dist)
        n_trials = len(cells) * 2       # a 100-trial run over 42 cells samples ~2x
        wins, dims_won = {}, {}
        for _ in range(2000):
            pick = rng.choice(cells, size=n_trials, replace=True)
            best, bv = None, np.inf
            for c in pick:
                v = rng.choice(dist[c])
                if v < bv:
                    bv, best = v, c
            wins[best] = wins.get(best, 0) + 1
            dims_won[best.split('|')[1]] = dims_won.get(best.split('|')[1], 0) + 1
        top = sorted(wins.items(), key=lambda kv: -kv[1])[:5]
        out['selection_stability'] = {
            'bootstrap_runs': 2000, 'trials_per_run': n_trials,
            'most_frequent_winners': [{'config': k, 'share': v / 2000} for k, v in top],
            'unique_winners': len(wins),
            'dim_share_of_winners': {k: v / 2000 for k, v in sorted(dims_won.items())},
            'backbone_share_of_winners': {c.split('|')[0]: v / 2000
                                          for c, v in sorted(wins.items(),
                                                             key=lambda kv: -kv[1])[:6]},
        }
    jdump(out, FILE_C)
    log('phase C done -> %s' % FILE_C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='A', choices=['A', 'B', 'C', 'AB', 'ABC'])
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=FINAL_EPOCHS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[FINAL_SEED],
                    help='training seeds for phase B; each seed is a full re-train '
                         'of every (backbone, category) pair')
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--backbones', nargs='+', default=BACKBONES)
    a = ap.parse_args()
    if 'A' in a.phase:
        phase_a(a.repeats)
    if 'B' in a.phase:
        phase_b(a.categories, a.backbones, a.epochs, tuple(a.seeds))
    if 'C' in a.phase:
        phase_c(a.repeats)


if __name__ == '__main__':
    main()
