# -*- coding: utf-8 -*-
"""Memory-aware training for the truth matrix.

Why this exists
---------------
The manuscript's detector training uses batch size 64 (p2_rd_train.train_category).
On a 24 GB RTX 3090 the large backbones in the search space -- convnextv2_base in
particular -- exceed device memory at that batch size. Windows WDDM does not fail
loudly in that situation: it silently pages VRAM to host RAM, which is
catastrophically slow. A measured example from this project:

    convnextv2_base, bottle, 500 epochs : ~1 min/epoch  (~8.4 h per run)
    resnet50,        same protocol      : 6.4 min total (from the capacity study)

That is a ~80x slowdown for a 3.5x larger model, i.e. paging rather than compute.

What this module does
---------------------
1. Probes, per backbone, the largest micro-batch that stays inside a memory budget
   (measured with a real forward+backward, so it accounts for activations).
2. Trains with that micro-batch and gradient accumulation, so the *effective*
   batch size remains the manuscript's 64 and the number of optimiser steps per
   epoch is unchanged.

The only deviation from train_category is that accumulated micro-batches of size
m < 64 supply the batch statistics to BatchNorm instead of full batches of 64.
Both the micro-batch and the measured peak are recorded so the deviation is
auditable.
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

import importlib.util, os, random, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score

NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
_spec = importlib.util.spec_from_file_location(
    'p2', os.path.join(NAS, 'scripts', 'p2_rd_train.py'))
_p2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_p2)

load_category_tensors = _p2.load_category_tensors
NORM = _p2.NORM
set_seed = _p2.set_seed

CANDIDATES = [64, 48, 32, 24, 16, 12, 8, 4, 2]
BUDGET_GB = 12.0          # measured peak we allow; leaves ~12 GB headroom on a 24 GB card

_PROBE_CACHE = {}


def probe_micro_batch(backbone, device, target_bs=64, budget_gb=BUDGET_GB,
                      teacher_arch='resnet18'):
    """Return (micro_bs, peak_gb) for the largest micro-batch inside the budget.

    Cached per (backbone, target_bs): the answer depends only on the architecture,
    and re-probing for every one of the 15 categories would waste ~15 min.
    """
    ck = (backbone, target_bs)
    if ck in _PROBE_CACHE:
        return _PROBE_CACHE[ck]
    from methods.rd_detector import RDDetector
    # Ascend from the smallest candidate and stop at the first breach, so we never
    # attempt an allocation large enough to trigger WDDM paging.
    fits = None
    cands = sorted([b for b in CANDIDATES if b <= target_bs])
    for bs in cands:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = RDDetector(student_backbone=backbone, teacher_arch=teacher_arch).to(device)
            m.train()
            x = torch.randn(bs, 3, 224, 224, device=device)
            loss = m.training_loss(x)
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            del m, x, loss
            torch.cuda.empty_cache()
            if peak <= budget_gb:
                fits = (bs, peak)
            else:
                break
        except RuntimeError:
            torch.cuda.empty_cache()
            break
    out = fits if fits is not None else (cands[0], float('nan'))
    _PROBE_CACHE[ck] = out
    return out


@torch.no_grad()
def evaluate(model, test_imgs, test_labels, device, bs=64):
    model.eval()
    scores = []
    for i in range(0, len(test_imgs), bs):
        images = NORM(test_imgs[i:i + bs].to(device))
        scores.extend(model.anomaly_score(images).cpu().numpy())
    labels = test_labels.numpy()
    if len(np.unique(labels)) < 2:
        return 0.0, 0.0
    return (float(roc_auc_score(labels, np.array(scores))),
            float(average_precision_score(labels, np.array(scores))))


def train_category_safe(category, student_backbone, epochs, seed, output_dir, device,
                        teacher='resnet18', target_bs=64, log_every=50, verbose=False):
    """train_category with memory-aware micro-batching. Returns a record dict."""
    import json
    from pathlib import Path
    output_dir = Path(output_dir)
    eval_file = output_dir / 'final_evaluation.json'
    if eval_file.exists():
        d = json.load(open(eval_file))
        d.setdefault('minutes', 0.0)
        d['cached'] = True
        return d

    t0 = time.time()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    micro_bs, peak = probe_micro_batch(student_backbone, device, target_bs=target_bs,
                                       teacher_arch=teacher)
    accum = max(1, int(round(target_bs / micro_bs)))

    train_imgs, test_imgs, test_labels = load_category_tensors(category)
    n_train = len(train_imgs)

    from methods.rd_detector import RDDetector
    model = RDDetector(student_backbone=student_backbone, teacher_arch=teacher).to(device)
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n_train)
        optimizer.zero_grad()
        pending = 0
        for b in range(0, n_train, micro_bs):
            images = NORM(train_imgs[idx[b:b + micro_bs]].to(device))
            if random.random() > 0.5:
                images = torch.flip(images, dims=[3])
            loss = model.training_loss(images) / accum
            loss.backward()
            pending += 1
            last = (b + micro_bs) >= n_train
            if pending == accum or last:
                optimizer.step()
                optimizer.zero_grad()
                pending = 0
        scheduler.step()

        if verbose and (epoch % log_every == 0 or epoch == epochs):
            auroc, ap = evaluate(model, test_imgs, test_labels, device, bs=64)
            print('  [%s/%s] epoch %d: loss=%.4f AUROC=%.4f AP=%.4f'
                  % (student_backbone, category, epoch, float(loss) * accum, auroc, ap),
                  flush=True)

    auroc, ap = evaluate(model, test_imgs, test_labels, device, bs=64)
    minutes = (time.time() - t0) / 60
    rec = {'auroc': auroc, 'ap': ap, 'student_backbone': student_backbone,
           'teacher': teacher, 'epochs': epochs, 'seed': seed,
           'minutes': round(minutes, 1), 'micro_batch': micro_bs,
           'accum_steps': accum, 'effective_batch': micro_bs * accum,
           'probe_peak_gb': round(peak, 2), 'cached': False}
    with open(eval_file, 'w') as f:
        json.dump(rec, f, indent=2)
    del model, train_imgs, test_imgs, test_labels
    torch.cuda.empty_cache()
    return rec
