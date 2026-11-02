# -*- coding: utf-8 -*-
"""Per-category defect-area statistics, used to test the pooling-gain mechanism.

For every anomalous test image with a ground-truth mask, the defect area fraction
is (#mask pixels > 0) / (image pixels). Fractions are scale invariant, so the
mask resolution does not matter and the 224x224 resize used for training does not
have to be repeated here.

MVTec-AD : <root>/<category>/ground_truth/<defect>/*_mask.png
VisA     : the 'mask' column of split_csv/1cls.csv, relative to the dataset root

Writes  fusion_policy/defect_area.json
"""
import csv, io, json, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import numpy as np
from PIL import Image

MVTEC_ROOT = os.path.join(os.environ.get('KGD_DATASETS_ROOT', r'D:/autopaper'), 'datasets/mvtec_anomaly_detection')
VISA_ROOT = os.path.join(os.environ.get('KGD_DATASETS_ROOT', r'D:/autopaper'), 'datasets-2')
DEST = os.path.join(os.environ.get('KGD_WORK_ROOT', r'D:/autopaper'), 'fusion_policy', 'defect_area.json')

MVTEC_CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
              'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
              'wood', 'zipper']
VISA_CATS = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
             'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']


def frac(path):
    a = np.asarray(Image.open(path).convert('L'))
    # MVTec masks are 0/255, VisA masks are 0/1 -- both are binary
    return float((a > 0).sum()) / a.size


def mvtec_stats(cat):
    base = os.path.join(MVTEC_ROOT, cat, 'ground_truth')
    if not os.path.isdir(base):
        return None
    fr = []
    for d in sorted(os.listdir(base)):
        sub = os.path.join(base, d)
        if not os.path.isdir(sub):
            continue
        for f in sorted(os.listdir(sub)):
            if f.lower().endswith(('.png', '.jpg', '.bmp')):
                fr.append(frac(os.path.join(sub, f)))
    return fr or None


def visa_stats(cat, root=VISA_ROOT):
    csv_path = os.path.join(root, 'split_csv', '1cls.csv')
    fr = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['object'] != cat or row['label'] != 'anomaly':
                continue
            if not row.get('mask'):
                continue
            p = os.path.join(root, row['mask'])
            if os.path.exists(p):
                fr.append(frac(p))
    return fr or None


def summarise(name, cats, getter):
    out = {}
    for c in cats:
        fr = getter(c)
        if not fr:
            print('  %-14s (no masks)' % c)
            continue
        a = np.array(fr)
        out[c] = {'n': int(len(a)), 'mean_frac': float(a.mean()),
                  'median_frac': float(np.median(a)),
                  'p90_frac': float(np.percentile(a, 90))}
        print('  %-14s n=%4d  mean %.4f  median %.4f  p90 %.4f'
              % (c, len(a), a.mean(), np.median(a), np.percentile(a, 90)))
    print('  -- %s: %d categories, mean of category means %.4f'
          % (name, len(out), np.mean([v['mean_frac'] for v in out.values()]) if out else float('nan')))
    return out


def main():
    print('=== MVTec-AD defect area ===')
    mv = summarise('MVTec-AD', MVTEC_CATS, mvtec_stats)
    print('\n=== VisA defect area ===')
    vi = summarise('VisA', VISA_CATS, visa_stats)
    json.dump({'MVTec-AD': mv, 'VisA': vi}, open(DEST, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print('\nwritten: %s' % DEST)


if __name__ == '__main__':
    main()
