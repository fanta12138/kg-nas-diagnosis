# -*- coding: utf-8 -*-
"""Shared machinery for the AD benchmark harness.

Design goals: (1) no training required for the first probes, so results arrive in
minutes; (2) every metric reported per category and with several seeds, because
the whole point is to measure the *noise floor* of what the literature reports;
(3) deployment-faithful metrics (bound-FPR) alongside the usual AUROC.

Contents
  data        -- MVTec-AD loading with masks
  Backbone    -- frozen feature extractors (timm), multi-scale
  metrics     -- image AUROC, pixel AUROC, PRO, and bound-FPR (R@q%FPR, AP@FPR)
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

import io, json, os, sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = _os.path.join(_KGD_DATA, 'datasets', 'mvtec_anomaly_detection')
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------- data
def _load_image(path, size):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    return (a - IMAGENET_MEAN) / IMAGENET_STD


def _load_mask(path, size):
    m = Image.open(path).convert('L')
    m = m.resize((size, size), Image.NEAREST)
    return (np.asarray(m, dtype=np.float32) > 0.5).astype(np.uint8)


def _load_mask_dir(dirpath, size):
    """LOCO stores one *directory* per anomalous image, holding one PNG per
    annotated defect instance. Union them into a single binary mask."""
    acc = None
    for f in sorted(os.listdir(dirpath)):
        if not f.lower().endswith('.png'):
            continue
        m = Image.open(os.path.join(dirpath, f)).convert('L')
        m = m.resize((size, size), Image.NEAREST)
        a = (np.asarray(m, dtype=np.float32) > 0.5)
        acc = a if acc is None else (acc | a)
    if acc is None:
        return np.zeros((size, size), np.uint8)
    return acc.astype(np.uint8)


def load_category(cat, size=448, root=ROOT):
    """Return (train_normal, test, labels, masks). All arrays images-first."""
    base = os.path.join(root, cat)
    tr, te, lab, msk = [], [], [], []
    good = os.path.join(base, 'train', 'good')
    for f in sorted(os.listdir(good)):
        tr.append(_load_image(os.path.join(good, f), size))
    tdir = os.path.join(base, 'test')
    for defect in sorted(os.listdir(tdir)):
        d = os.path.join(tdir, defect)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            te.append(_load_image(os.path.join(d, f), size))
            lab.append(0 if defect == 'good' else 1)
            gp = os.path.join(base, 'ground_truth', defect, os.path.splitext(f)[0] + '_mask.png')
            msk.append(_load_mask(gp, size) if os.path.exists(gp)
                       else np.zeros((size, size), np.uint8))
    return (np.stack(tr), np.stack(te), np.array(lab, np.int64), np.stack(msk))


# ------------------------------------------------------------------- MVTec LOCO
# Logical-constraint benchmark (Bergmann et al., IJCV 2022). Same normal-only
# training protocol, but the test set is split into three parts -- good, logical
# and structural -- and the paper's metric is *s*PRO, a PRO whose integration
# limit is FPR <= 5% rather than 30%. Logical anomalies are the interesting ones:
# local appearance is normal and only the arrangement/count/composition is wrong.
LOCO_ROOT = _os.path.join(_KGD_DATA, 'mvtec_loco_anomaly_detection')
LOCO_CATS = ['breakfast_box', 'juice_bottle', 'pushpins', 'screw_bag',
             'splicing_connectors']


def _first_existing(*paths):
    for p in paths:
        if os.path.isdir(p):
            return p
    return None


def load_category_loco(cat, size=448, root=LOCO_ROOT):
    """Return (train, test, labels, masks, kinds).

    kinds[i] is one of 'good' / 'logical' / 'structural' and lets the caller
    report the two anomaly families separately, which is the whole point of LOCO.
    """
    base = os.path.join(root, cat)
    tr, te, lab, msk, kind = [], [], [], [], []

    for sub in ('train', 'validation'):
        d = _first_existing(os.path.join(base, sub, 'good'),
                            os.path.join(base, sub, 'normal'))
        if d:
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    tr.append(_load_image(os.path.join(d, f), size))

    gt = _first_existing(os.path.join(base, 'ground_truth'), base)
    for fam in ('good', 'logical_anomalies', 'structural_anomalies'):
        d = os.path.join(base, 'test', fam)
        if not os.path.isdir(d):
            continue
        label = 0 if fam == 'good' else 1
        k = 'good' if fam == 'good' else ('logical' if fam.startswith('logical') else 'structural')
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            te.append(_load_image(os.path.join(d, f), size))
            lab.append(label)
            kind.append(k)
            stem = os.path.splitext(f)[0]
            arr = np.zeros((size, size), np.uint8)
            if gt:
                gdir = os.path.join(gt, fam, stem)          # dir of instance masks
                gfile = _first_existing(os.path.join(gt, fam, stem + '.png'),
                                        os.path.join(gt, fam, stem + '_mask.png'))
                if os.path.isdir(gdir):
                    arr = _load_mask_dir(gdir, size)
                elif gfile:
                    arr = _load_mask(gfile, size)
            msk.append(arr)
    return (np.stack(tr), np.stack(te), np.array(lab, np.int64),
            np.stack(msk), np.array(kind))


def summarise_loco(labels, kinds, scores, masks=None, smaps=None, pro_limit=0.05):
    """Overall metrics plus the logical / structural split, using sPRO."""
    out = {'i_auroc': image_auroc(labels, scores)}
    r, ap = bound_fpr(labels, scores, 0.01)
    out['r_at_1fpr'], out['ap_at_1fpr'] = r, ap
    if smaps is not None and masks is not None:
        if np.asarray(smaps).shape[-2:] != np.asarray(masks).shape[-2:]:
            smaps = upsample_maps(smaps, np.asarray(masks).shape[-2:])
        masks = np.asarray(masks)
        smaps = np.asarray(smaps)
        out['p_auroc'] = pixel_auroc(masks, smaps)
        out['spro'] = pro(masks, smaps, fpr_limit=pro_limit)
        for fam in ('logical', 'structural'):
            sel = kinds == fam
            if sel.sum() == 0:
                continue
            sub_m = masks[sel]
            sub_s = smaps[sel]
            if sub_m.sum() == 0:
                continue
            # Within a single family every image is anomalous, so a family-only
            # AUROC is undefined; only the family-vs-good variants below are
            # meaningful. sPRO is well defined because it uses the normal-pixel
            # score distribution.
            out['spro_%s' % fam[:3]] = pro(sub_m, sub_s, fpr_limit=pro_limit)
        # the paper also reports a "logical only vs good" detection AUROC
        for fam in ('logical', 'structural'):
            sel = (kinds == fam) | (kinds == 'good')
            if (kinds == fam).sum() and (kinds == 'good').sum():
                out['i_auroc_%s_vs_good' % fam[:3]] = image_auroc(
                    labels[sel], np.asarray(scores)[sel])
    return out


# ------------------------------------------------------------------------ VisA
# Visual Anomaly detection benchmark (Zou et al., ECCV 2022). 12 objects. Unlike
# MVTec-AD the official protocol is defined by split_csv/1cls.csv, which splits
# the *normal* images into a train and a test part, so the split must be read from
# the CSV rather than assumed.
VISA_ROOT = _os.path.join(_KGD_DATA, 'datasets-2')
VISA_CATS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def load_category_visa(cat, size=448, root=VISA_ROOT, split='1cls'):
    """Return (train_normal, test, labels, masks) following the official split."""
    import csv
    csvp = os.path.join(root, 'split_csv', '%s.csv' % split)
    rows = [r for r in csv.DictReader(open(csvp, encoding='utf-8'))
            if r['object'] == cat]
    if not rows:
        raise RuntimeError('no rows for %s in %s' % (cat, csvp))
    tr, te, lab, msk = [], [], [], []
    for r in rows:
        img = _load_image(os.path.join(root, r['image'].strip()), size)
        if r['split'] == 'train':
            tr.append(img)
            continue
        te.append(img)
        is_anom = (r['label'].strip() == 'anomaly')
        lab.append(1 if is_anom else 0)
        mp = (r.get('mask') or '').strip()
        if is_anom and mp and os.path.exists(os.path.join(root, mp)):
            msk.append(_load_mask(os.path.join(root, mp), size))
        else:
            msk.append(np.zeros((size, size), np.uint8))
    return (np.stack(tr), np.stack(te), np.array(lab, np.int64), np.stack(msk))


# ---------------------------------------------------------------------- backbone
class Backbone:
    """Frozen multi-scale feature extractor.

    Two paths, both driven by the model's *own* forward pass so we never have to
    reimplement architecture-specific shape logic:

      CNN (features_only) -- the stage outputs are returned directly
      ViT/DeiT            -- forward hooks on the chosen blocks capture the token
                             tensors, which are then reshaped to feature maps

    Features are L2-normalised per location at every scale so that distances are
    comparable across scales.
    """

    VIT_HINTS = ('vit', 'deit', 'xcit', 'beit', 'flexivit')

    def __init__(self, name='resnet50.a1_in1k', device='cuda', layers=None, size=448):
        import timm
        self.name = name
        self.device = device
        self.size = size
        self.is_vit = any(k in name for k in self.VIT_HINTS)
        self._buf = {}
        self._blocks = None

        if self.is_vit:
            # dynamic_img_size lets timm interpolate the positional embeddings so a
            # 224-registered checkpoint accepts the higher resolution used here.
            self.model = timm.create_model(name, pretrained=True, num_classes=0,
                                           dynamic_img_size=True)
            n = len(self.model.blocks)
            self.layers = layers or [n // 4, n // 2, int(n * 0.75), n - 1]
            self._blocks = self.model.blocks
            self.n_prefix = int(getattr(self.model, 'num_prefix_tokens', 1) or 1)
            for idx in self.layers:
                self._blocks[idx].register_forward_hook(self._mk_hook(idx))
        else:
            self.model = timm.create_model(name, pretrained=True, features_only=True)
            self.layers = layers or [1, 2, 3]

        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _mk_hook(self, idx):
        def hook(module, inp, out):
            self._buf[idx] = out
        return hook

    @torch.no_grad()
    def features(self, batch_np, chunk=8, grid=None):
        """batch_np: (N,H,W,3) already normalised. Returns list of (N,C,h,w).

        If `grid` is given each scale is resized to grid x grid *inside* the loop,
        so full-resolution feature maps are never held for the whole batch. With a
        few hundred training images the un-gridded tensors otherwise exceed
        INT_MAX elements and the interpolate kernel refuses them.
        """
        out = None
        for i in range(0, len(batch_np), chunk):
            x = torch.from_numpy(np.ascontiguousarray(batch_np[i:i + chunk]))
            x = x.permute(0, 3, 1, 2).to(self.device)
            maps = self._forward(x)
            if grid is not None:
                maps = [F.interpolate(m, size=(grid, grid), mode='bilinear',
                                      align_corners=False) for m in maps]
            if out is None:
                out = [[] for _ in maps]
            for j, m in enumerate(maps):
                out[j].append(m.float().cpu())
        return [torch.cat(o, 0).numpy() for o in out]

    def _forward(self, x):
        if not self.is_vit:
            fs = self.model(x)
            return [F.normalize(fs[i], dim=1) for i in self.layers]

        self._buf.clear()
        _ = self.model.forward_features(x)          # hooks populate self._buf
        res = []
        for idx in self.layers:
            t = self._buf[idx]
            if t.dim() == 4:                         # (B,C,H,W) style output
                t = t.flatten(2).transpose(1, 2)     # -> (B,N,C)
            tok = t[:, self.n_prefix:]               # drop cls/register tokens
            b, n, c = tok.shape
            h = w = int(round(n ** 0.5))
            if h * w != n:
                raise RuntimeError('non-square token grid for %s: %d tokens' % (self.name, n))
            res.append(F.normalize(tok.transpose(1, 2).reshape(b, c, h, w), dim=1))
        return res


# ----------------------------------------------------------------------- metrics
def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def image_auroc(labels, scores):
    return _auroc(np.asarray(labels), np.asarray(scores))


def pixel_auroc(masks, smaps):
    y = np.asarray(masks).reshape(-1)
    s = np.asarray(smaps).reshape(-1)
    if len(y) > 2_000_000:
        idx = np.random.default_rng(0).choice(len(y), 2_000_000, replace=False)
        y, s = y[idx], s[idx]
    return _auroc(y, s)


def bound_fpr(labels, scores, q=0.01):
    """Recall at a fixed false-positive rate, and AP below that FPR.

    This is the deployment-faithful family: in production the operating point is
    set by the tolerable false-alarm rate, not by AUROC.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    neg, pos = scores[labels == 0], scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float('nan'), float('nan')
    thr = np.quantile(neg, 1.0 - q)            # FPR = q on normal images
    r = float((pos > thr).mean())
    # AP restricted to thresholds with FPR <= q
    order = np.argsort(-scores)
    l = labels[order]; s = scores[order]
    tp = np.cumsum(l); fp = np.cumsum(1 - l)
    fpr = fp / len(neg)
    keep = fpr <= q
    if keep.sum() == 0:
        return r, 0.0
    prec = tp[keep] / np.maximum(tp[keep] + fp[keep], 1)
    rec = tp[keep] / len(pos)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    return r, ap


def pro(masks, smaps, fpr_limit=0.3, n_thr=64):
    """Per-Region Overlap (Bergmann et al.).

    Each connected ground-truth region counts once, so a method cannot win by
    over-segmenting the defect. PRO(t) = mean over regions of the fraction of the
    region's pixels scoring above t; PRO = mean of PRO(t) over thresholds chosen
    to sweep the false-positive rate from 0 to `fpr_limit` on normal pixels.
    """
    from skimage.measure import label as cc_label

    masks = np.asarray(masks, dtype=np.uint8)
    smaps = np.asarray(smaps, dtype=np.float32)
    n = len(masks)

    regions = []                      # list of (image_index, flat boolean index)
    for i in range(n):
        if masks[i].max() == 0:
            continue
        lab, k = cc_label(masks[i], connectivity=2, return_num=True)
        for c in range(1, k + 1):
            idx = np.flatnonzero(lab.reshape(-1) == c)
            if idx.size:
                regions.append((i, idx))
    if not regions:
        return float('nan')

    flat_s = smaps.reshape(n, -1)
    normal_scores = np.concatenate([flat_s[i][masks[i].reshape(-1) == 0] for i in range(n)])
    if normal_scores.size == 0:
        return float('nan')

    # thresholds = quantiles of the NORMAL score distribution sweeping FPR 0..limit
    lv = 1.0 - np.linspace(0.0, fpr_limit, n_thr)
    thrs = np.unique(np.quantile(normal_scores, lv))
    if thrs.size == 0:
        return float('nan')

    ov = np.empty((thrs.size, len(regions)), dtype=np.float64)
    for ri, (i, idx) in enumerate(regions):
        v = flat_s[i][idx]
        v = np.sort(v)
        # fraction of the region's pixels strictly above each threshold
        above = (v.size - np.searchsorted(v, thrs, side='right')) / v.size
        ov[:, ri] = above
    return float(ov.mean())


def upsample_maps(smaps, hw):
    """Resample anomaly maps to the ground-truth mask resolution."""
    import torch
    import torch.nn.functional as Fn
    t = torch.as_tensor(np.asarray(smaps), dtype=torch.float32)[:, None]
    t = Fn.interpolate(t, size=hw, mode='bilinear', align_corners=False)
    return t[:, 0].numpy()


def summarise(labels, scores, masks=None, smaps=None, pro_limit=0.3):
    out = {'i_auroc': image_auroc(labels, scores)}
    if smaps is not None and masks is not None:
        if np.asarray(smaps).shape[-2:] != np.asarray(masks).shape[-2:]:
            smaps = upsample_maps(smaps, np.asarray(masks).shape[-2:])
    r, ap = bound_fpr(labels, scores, 0.01)
    out['r_at_1fpr'] = r
    out['ap_at_1fpr'] = ap
    r5, ap5 = bound_fpr(labels, scores, 0.05)
    out['r_at_5fpr'] = r5
    if smaps is not None and masks is not None:
        out['p_auroc'] = pixel_auroc(masks, smaps)
        out['pro'] = pro(masks, smaps, fpr_limit=pro_limit)
    return out
