# -*- coding: utf-8 -*-
"""Faithful baselines via anomalib 0.7 (reference implementations).

The manuscript's own baseline suite is marked "简化版实现" in its source and
scores PatchCore 50.6 / STPM 51.1 / SimpleNet 50.5 on MVTec-AD — i.e. chance —
against published values of 99.1 / 95.6 / 99.6. This runner instead uses
anomalib's reference implementations, so the comparison in the paper is made
against methods that actually work.

Usage: python run_anomalib.py [--methods ...] [--categories ...]
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

import argparse, io, json, os, sys, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

ROOT_DATA = _os.path.join(_KGD_DATA, 'datasets', 'mvtec_anomaly_detection')
OUT = _os.path.join(_KGD_ROOT, 'anomalib_baselines')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'results.json')

CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
METHODS = ['patchcore', 'padim', 'cfa', 'fastflow', 'stfpm', 'reverse_distillation', 'efficientad']


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def build(method):
    """Standard MVTec-AD configurations for each anomalib model."""
    from anomalib.models import (Cfa, EfficientAd, Fastflow, Padim, Patchcore,
                                 ReverseDistillation, Stfpm)
    SIZE = (256, 256)
    if method == 'patchcore':
        return Patchcore(input_size=SIZE, backbone='resnet18',
                         layers=['layer2', 'layer3'], coreset_sampling_ratio=0.1)
    if method == 'padim':
        return Padim(layers=['layer1', 'layer2', 'layer3'],
                     input_size=SIZE, backbone='resnet18')
    if method == 'cfa':
        return Cfa(input_size=SIZE, backbone='resnet18')
    if method == 'fastflow':
        return Fastflow(input_size=SIZE, backbone='resnet18')
    if method == 'stfpm':
        return Stfpm(input_size=SIZE, backbone='resnet18',
                     layers=['layer1', 'layer2', 'layer3'])
    if method == 'reverse_distillation':
        from anomalib.models.reverse_distillation.anomaly_map import AnomalyMapGenerationMode
        return ReverseDistillation(
            input_size=SIZE, backbone='resnet18',
            layers=['layer1', 'layer2', 'layer3'],
            anomaly_map_mode=AnomalyMapGenerationMode.ADD,
            lr=0.005, beta1=0.5, beta2=0.999)
    if method == 'efficientad':
        from anomalib.models.efficient_ad import EfficientAdModelSize
        return EfficientAd(teacher_out_channels=384, image_size=SIZE,
                           model_size=EfficientAdModelSize.S, lr=0.0001)
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', nargs='+', default=METHODS)
    ap.add_argument('--categories', nargs='+', default=CATS)
    ap.add_argument('--max-epochs', type=int, default=1)
    a = ap.parse_args()

    import pytorch_lightning as pl
    from anomalib.data import MVTec

    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for method in a.methods:
        for cat in a.categories:
            key = '%s|%s' % (method, cat)
            if key in res and res[key].get('image_auroc') is not None:
                continue
            t0 = time.time()
            try:
                dm = MVTec(root=ROOT_DATA, category=cat, image_size=(256, 256),
                           train_batch_size=32, eval_batch_size=32, num_workers=4)
                model = build(method)
                trainer = pl.Trainer(max_epochs=a.max_epochs, devices=1, accelerator='gpu',
                                     logger=False, enable_progress_bar=False,
                                     enable_model_summary=False,
                                     default_root_dir=os.path.join(OUT, 'runs'))
                trainer.fit(model=model, datamodule=dm)
                out = trainer.test(model=model, datamodule=dm)
                m = out[0] if isinstance(out, (list, tuple)) and out else (out or {})
                keys = {k: v for k, v in m.items() if isinstance(v, (int, float))}
                img = next((float(v) for k, v in keys.items()
                            if 'image' in k.lower() and 'auroc' in k.lower()), None)
                res[key] = {'method': method, 'category': cat, 'image_auroc': img,
                            'all_metrics': {k: round(float(v), 5) for k, v in keys.items()},
                            'minutes': round((time.time() - t0) / 60, 2)}
                log('%-22s %-11s image-AUROC=%s  (%.1f min)'
                    % (method, cat, ('%.4f' % img) if img is not None else 'n/a',
                       res[key]['minutes']))
            except Exception as e:
                import traceback
                res[key] = {'method': method, 'category': cat, 'image_auroc': None,
                            'error': (str(e)[:300] + ' | ' + traceback.format_exc()[-400:]),
                            'minutes': round((time.time() - t0) / 60, 2)}
                log('%-22s %-11s FAIL: %s' % (method, cat, str(e)[:120]))
            json.dump(res, open(RESFILE, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    # aggregate
    log('')
    log('=' * 62)
    log('%-24s %12s %6s' % ('method', 'mean image AUROC', 'n'))
    log('-' * 62)
    for method in a.methods:
        vals = [v['image_auroc'] for k, v in res.items()
                if v.get('method') == method and v.get('image_auroc') is not None]
        if vals:
            log('%-24s %11.2f%% %6d' % (method, 100 * sum(vals) / len(vals), len(vals)))
    log('')
    log('wrote ' + RESFILE)


if __name__ == '__main__':
    main()
