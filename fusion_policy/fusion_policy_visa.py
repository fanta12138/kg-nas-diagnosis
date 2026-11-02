# -*- coding: utf-8 -*-
"""Scale-fusion policy sweep on VisA -- cross-dataset replication of §5.2.

Purpose
-------
The MVTec sweep (fusion_policy.py) shows that replacing the default mean
pooling with top-k raises mean AUROC by 7.4 points on already-trained
detectors. The manuscript states that this was measured on one dataset and one
student family. This script repeats the identical protocol on VisA (12
categories), so the claim can be stated across two datasets.

What is deliberately kept identical to the MVTec run
----------------------------------------------------
* student backbone   : efficientnet_b0
* teacher            : ImageNet ResNet-18 (RDDetector(teacher_arch='resnet18'))
* optimiser          : AdamW, lr 1e-3, wd 1e-4, cosine schedule
* epochs             : 500
* batch size         : 64
* seed               : 42
* augmentation       : random horizontal flip, p=0.5
* input              : Resize((224,224)), ImageNet normalisation
* policy grid        : the same 82 policies (9 single-scale + 3 uniform-sum
                       + 66 simplex weights at step 0.1 with top-k
                       + 4 per-image-normalised)

The only change is the data source: VisA's official 1cls split.

Usage
-----
    python fusion_policy_visa.py                     # all 12 categories
    python fusion_policy_visa.py --categories candle # one category
    python fusion_policy_visa.py --epochs 2 --limit 20   # smoke test
"""
import argparse, csv, io, json, os, random, sys, time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

NAS = os.path.join(os.environ.get('KGD_WORK_ROOT', r'D:/autopaper'), 'unsupervised_nas')
sys.path.insert(0, NAS)
os.chdir(NAS)

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics import roc_auc_score

from methods.rd_detector import RDDetector
from torch.optim import AdamW

ROOT = os.path.join(os.environ.get('KGD_DATASETS_ROOT', r'D:/autopaper'), 'datasets-2')                 # VisA, official 1cls split
OUT = os.path.join(os.environ.get('KGD_WORK_ROOT', r'D:/autopaper'), 'fusion_policy')
os.makedirs(OUT, exist_ok=True)
RESFILE = os.path.join(OUT, 'fusion_policy_visa.json')

VISA_CATS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
RESIZE = transforms.Resize((224, 224))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(m):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), m), flush=True)


def load_visa_tensors(category, root=ROOT, limit=None):
    """Return (train_imgs, test_imgs, test_labels) exactly as the MVTec path does.

    VisA images are much larger than MVTec's; they are resized to 224x224 on a
    single read and held in memory, as load_category_tensors does for MVTec.
    """
    csv_path = os.path.join(root, 'split_csv', '1cls.csv')
    train, test = [], []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['object'] != category:
                continue
            rec = (os.path.join(root, row['image']), row['label'])
            if row['split'] == 'train':
                train.append(rec)
            elif row['split'] == 'test':
                test.append(rec)
    if limit:
        train, test = train[:limit], test[:limit]

    def load(recs, with_label):
        imgs, labels = [], []
        for path, lab in recs:
            im = Image.open(path).convert('RGB')
            imgs.append(RESIZE(transforms.ToTensor()(im)))
            if with_label:
                labels.append(0 if lab == 'normal' else 1)
        return torch.stack(imgs), labels

    tr, _ = load(train, False)
    te, te_labels = load(test, True)
    log('  %s: %d train / %d test (%.1f%% anomalous)'
        % (category, len(tr), len(te), 100.0 * np.mean(te_labels)))
    return tr, te, torch.tensor(te_labels)


def train_student(fit_imgs, backbone, epochs, device, seed=42, bs=64):
    set_seed(seed)
    model = RDDetector(student_backbone=backbone, teacher_arch='resnet18').to(device)
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(fit_imgs)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            x = NORM(fit_imgs[perm[b:b + bs]].to(device))
            if torch.rand(1).item() > 0.5:
                x = torch.flip(x, dims=[3])
            opt.zero_grad()
            loss = model.training_loss(x)
            loss.backward()
            opt.step()
        sched.step()
        if epochs <= 5 or (ep + 1) % 100 == 0:
            log('    epoch %d/%d' % (ep + 1, epochs))
    return model


@torch.no_grad()
def scale_maps(model, imgs, device, bs=16, size=224):
    import torch.nn.functional as F
    model.eval()
    out = [[], [], []]
    for i in range(0, len(imgs), bs):
        x = NORM(imgs[i:i + bs].to(device))
        t1, t2, t3 = model.teacher(x)
        s1, s2, s3 = model.forward_student(x)
        for k, (s, t) in enumerate([(s1, t1), (s2, t2), (s3, t3)]):
            d = torch.norm(s - t, dim=1, keepdim=True)
            out[k].append(F.interpolate(d, size=(size, size), mode='bilinear',
                                        align_corners=False).squeeze(1).cpu())
    return [torch.cat(o, dim=0).numpy() for o in out]


def pool(maps, mode, k=0.01):
    flat = maps.reshape(maps.shape[0], -1)
    if mode == 'mean':
        return flat.mean(axis=1)
    if mode == 'max':
        return flat.max(axis=1)
    if mode == 'topk':
        n = max(1, int(flat.shape[1] * k))
        return np.sort(flat, axis=1)[:, -n:].mean(axis=1)
    raise ValueError(mode)


def norm01(m):
    lo = m.reshape(m.shape[0], -1).min(axis=1)[:, None, None]
    hi = m.reshape(m.shape[0], -1).max(axis=1)[:, None, None]
    return (m - lo) / np.maximum(hi - lo, 1e-8)


def auroc(y, s):
    return float(roc_auc_score(y, s))


def evaluate_policies(maps, y):
    """The same 82-policy grid used for MVTec."""
    rows = {}
    for k in range(3):                                    # 9 single-scale
        for pm in ('mean', 'topk', 'max'):
            rows['s%d_%s' % (k + 1, pm)] = auroc(y, pool(maps[k], pm))
    for pm in ('mean', 'topk', 'max'):                    # 3 uniform sum
        rows['sum_%s' % pm] = auroc(y, pool(maps[0] + maps[1] + maps[2], pm))
    best_w, best_auc = None, -1
    for w1 in np.arange(0, 1.01, 0.1):                    # 66 weight grid
        for w2 in np.arange(0, 1.01 - w1 + 1e-9, 0.1):
            w3 = round(1 - w1 - w2, 10)
            if w3 < -1e-9:
                continue
            s = pool(w1 * maps[0] + w2 * maps[1] + w3 * maps[2], 'topk')
            au = auroc(y, s)
            tag = 'w%.1f_%.1f_%.1f_topk' % (w1, w2, w3)
            rows[tag] = au
            if au > best_auc:
                best_auc, best_w = au, tag
    n0, n1, n2 = norm01(maps[0]), norm01(maps[1]), norm01(maps[2])   # 4 normalised
    for pm in ('mean', 'topk'):
        rows['norm_sum_%s' % pm] = auroc(y, pool(n0 + n1 + n2, pm))
        rows['norm_max_%s' % pm] = auroc(y, pool(np.maximum(np.maximum(n0, n1), n2), pm))
    return rows, best_w, best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--categories', nargs='+', default=VISA_CATS)
    ap.add_argument('--epochs', type=int, default=500)
    ap.add_argument('--backbone', default='efficientnet_b0')
    ap.add_argument('--data-root', default=ROOT)
    ap.add_argument('--limit', type=int, default=0, help='smoke test: cap images per split')
    a = ap.parse_args()

    device = torch.device('cuda')
    res = json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}

    for cat in a.categories:
        if cat in res and res[cat].get('done') and not a.limit:
            log('=== %s === already done, skipping' % cat)
            continue
        t0 = time.time()
        log('=== %s ===' % cat)
        train_imgs, test_imgs, y_t = load_visa_tensors(cat, a.data_root, a.limit or None)
        model = train_student(train_imgs, a.backbone, a.epochs, device)
        maps = scale_maps(model, test_imgs, device)
        y = np.asarray(y_t)
        rows, best_w, best_auc = evaluate_policies(maps, y)
        best_tag = max(rows, key=lambda k: rows[k])
        res[cat] = {
            'done': True, 'dataset': 'visa', 'backbone': a.backbone,
            'epochs': a.epochs, 'seed': 42,
            'n_train': int(len(train_imgs)), 'n_test': int(len(test_imgs)),
            'policies': rows,
            'current_default': rows['sum_mean'],
            'sum_topk': rows['sum_topk'],
            'delta_topk_minus_mean': rows['sum_topk'] - rows['sum_mean'],
            'best_policy': best_tag, 'best_auroc': rows[best_tag],
            'best_weight_policy': best_w, 'best_weight_auroc': best_auc,
            'minutes': round((time.time() - t0) / 60.0, 1),
        }
        with open(RESFILE, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        log('  default %.2f | top-k %.2f | delta %+.2f | %.1f min'
            % (rows['sum_mean'], rows['sum_topk'],
               rows['sum_topk'] - rows['sum_mean'], res[cat]['minutes']))

    done = [c for c in VISA_CATS if c in res and res[c].get('done')]
    if done:
        d = np.mean([res[c]['current_default'] for c in done])
        t = np.mean([res[c]['sum_topk'] for c in done])
        log('VisA so far (%d/%d categories): default %.2f | top-k %.2f | delta %+.2f'
            % (len(done), len(VISA_CATS), d, t, t - d))


if __name__ == '__main__':
    main()
