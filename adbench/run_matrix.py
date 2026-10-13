# -*- coding: utf-8 -*-
"""Run the full method x locality x seed x category matrix.

Loop order is chosen so that only one category's images are ever resident: the
dataset is loaded per category, all method configurations are run against it, and
it is then released.

Resumable: results are appended to results/matrix.json and completed cells are
skipped. Every cell stores the full metric set, so later analyses (seed noise,
ranking consistency, per-category risk) are re-derivations over this one file.

Usage: python run_matrix.py [--backbones a,b] [--radii None,4,2,1,0] [--seeds 0,1,2]
"""
import argparse, io, json, os, sys, time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
import torch

import common as C
from methods import PatchCore, KNNGlobal

OUTDIR = os.path.join(HERE, 'results')
os.makedirs(OUTDIR, exist_ok=True)
RESFILE = os.path.join(OUTDIR, 'matrix.json')   # overridden by --tag
SIZE = 448


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def load():
    return json.load(open(RESFILE, encoding='utf-8')) if os.path.exists(RESFILE) else {}


def save(d):
    tmp = RESFILE + '.tmp'
    json.dump(d, open(tmp, 'w', encoding='utf-8'), indent=2)
    os.replace(tmp, RESFILE)


def parse_radii(v):
    out = []
    for t in v.split(','):
        out.append(None if t.strip().lower() in ('none', 'global', '') else int(t))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', default='resnet50.a1_in1k')
    ap.add_argument('--radii', default='None,4,2,1,0')
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--categories', default='')
    ap.add_argument('--grid', type=int, default=32)
    ap.add_argument('--bank', type=int, default=8192)
    ap.add_argument('--max-per-cell', type=int, default=64)
    ap.add_argument('--tag', default='')
    ap.add_argument('--dataset', default='mvtec', choices=['mvtec', 'loco', 'visa'])
    a = ap.parse_args()

    backbones = [b.strip() for b in a.backbones.split(',') if b.strip()]
    radii = parse_radii(a.radii)
    seeds = [int(s) for s in a.seeds.split(',')]
    if a.categories:
        cats = [c.strip() for c in a.categories.split(',') if c.strip()]
    else:
        cats = {'mvtec': C.CATS, 'loco': C.LOCO_CATS, 'visa': C.VISA_CATS}[a.dataset]
    log('dataset=%s  categories=%d' % (a.dataset, len(cats)))

    global RESFILE
    if a.tag:
        RESFILE = os.path.join(OUTDIR, 'matrix_%s.json' % a.tag)
    res = load()
    bbs = {}

    for cat in cats:
        t0 = time.time()
        if a.dataset == 'mvtec':
            tr, te, lab, msk = C.load_category(cat, size=SIZE)
            kinds = None
        elif a.dataset == 'visa':
            tr, te, lab, msk = C.load_category_visa(cat, size=SIZE)
            kinds = None
        else:
            tr, te, lab, msk, kinds = C.load_category_loco(cat, size=SIZE)
        log('%s: train=%d test=%d pos=%d%s'
            % (cat, len(tr), len(te), int(lab.sum()),
               '' if kinds is None else '  logical=%d structural=%d'
               % (int((kinds == 'logical').sum()), int((kinds == 'structural').sum()))))
        for bname in backbones:
            if bname not in bbs:
                log('loading backbone %s' % bname)
                bbs[bname] = C.Backbone(bname, 'cuda', size=SIZE)
            bb = bbs[bname]
            for rad in radii:
                for sd in seeds:
                    key = '%s|%s|rad%s|s%d' % (cat, bname, rad, sd)
                    if key in res and res[key].get('ok'):
                        continue
                    t1 = time.time()
                    try:
                        m = PatchCore(bb, locality_radius=rad, bank_size=a.bank,
                                      max_per_cell=a.max_per_cell, seed=sd,
                                      g=a.grid).fit(tr)
                        img, maps = m.score(te)
                        if kinds is None:
                            r = C.summarise(lab, img, msk, maps)
                        else:
                            r = C.summarise_loco(lab, kinds, img, msk, maps)
                        r.update({'ok': True, 'seconds': round(time.time() - t1, 1),
                                  'category': cat, 'backbone': bname,
                                  'radius': ('global' if rad is None else rad),
                                  'seed': sd, 'grid': a.grid, 'bank': a.bank,
                                  'max_per_cell': a.max_per_cell, 'tag': a.tag,
                                  'dataset': a.dataset,
                                  'n_train': int(len(tr)), 'n_test': int(len(te)),
                                  'n_pos': int(lab.sum())})
                    except Exception as e:
                        r = {'ok': False, 'error': str(e)[:200], 'category': cat,
                             'backbone': bname, 'radius': ('global' if rad is None else rad),
                             'seed': sd}
                        torch.cuda.empty_cache()
                    res[key] = r
                    save(res)
            # a global-kNN baseline, once per (cat, backbone, seed)
            for sd in seeds:
                key = '%s|%s|knn_global|s%d' % (cat, bname, sd)
                if key in res and res[key].get('ok'):
                    continue
                try:
                    g = KNNGlobal(bb)
                    g.fit(tr)
                    img, maps = g.score(te)
                    if kinds is None:
                        r = C.summarise(lab, img, msk, maps)
                    else:
                        r = C.summarise_loco(lab, kinds, img, msk, maps)
                    r.update({'ok': True, 'category': cat, 'backbone': bname,
                              'radius': 'knn_global', 'seed': sd})
                except Exception as e:
                    r = {'ok': False, 'error': str(e)[:200], 'category': cat,
                         'backbone': bname, 'radius': 'knn_global', 'seed': sd}
                res[key] = r
                save(res)
        log('%s done (%.0fs, %d cells total)' % (cat, time.time() - t0, len(res)))
        del tr, te, lab, msk, kinds
        torch.cuda.empty_cache()

    log('MATRIX DONE: %d cells' % len(res))


if __name__ == '__main__':
    main()
