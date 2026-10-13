# -*- coding: utf-8 -*-
"""End-to-end driver for the capacity experiment.

Stage B+C: wait for every budget's search to finish, pick the best in-budget
configuration, train it on all 15 MVTec categories at 500 epochs, then
aggregate the accuracy-parameter curve.

Launched as a managed background task so it survives the whole run.
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

import json, os, subprocess, sys, time

EXP = _os.path.join(_KGD_ROOT, 'capacity_experiment')
NAS = _os.path.join(_KGD_ROOT, 'unsupervised_nas')
TRACE = _os.path.join(_KGD_ROOT, 'audit', 'claim-audit_20261008')
LOGS = os.path.join(EXP, 'logs', 'train')
os.makedirs(LOGS, exist_ok=True)

BUDGETS = [3, 6, 8, 13, 30]
CATS = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper']
PY = os.environ.get('KGD_PYTHON', sys.executable)  # interpreter used for the child workers
PARALLEL = int(os.environ.get('CAP_PARALLEL', '6'))
SEARCH_TIMEOUT_MIN = 240


def log(msg):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), msg), flush=True)


def wait_for_searches():
    t0 = time.time()
    done = set()
    while len(done) < len(BUDGETS):
        if (time.time() - t0) / 60 > SEARCH_TIMEOUT_MIN:
            log('search wait timed out after %d min' % SEARCH_TIMEOUT_MIN)
            break
        for B in BUDGETS:
            if B in done:
                continue
            p = os.path.join(NAS, 'experiments', 'cap_budget_%dM' % B, 'search_results.json')
            if os.path.exists(p):
                done.add(B)
                log('search %2dM finished (%d/%d)' % (B, len(done), len(BUDGETS)))
        if len(done) < len(BUDGETS):
            time.sleep(30)
    return done


def collect(available):
    """Pick, per budget, the config with the lowest MEAN valid search score.

    The author's engine uses a single-trial argmin over all draws
    (methods/nas_engine.py:273), which lets a degenerate trial — one where no
    validation batch passes the numerical-validity filter and the score
    collapses to exactly 0 — win the search. The manuscript already discloses
    this in Limitations. For a capacity comparison we instead apply the rule
    the paper STATES ("the architecture with the lowest average distillation
    loss"), after discarding degenerate (<=0) and divergent (>1e6) draws.
    """
    import statistics
    params = json.load(open(os.path.join(TRACE, 'params_by_backbone.json'), encoding='utf-8'))
    sel, diag = {}, {}
    for B in BUDGETS:
        if B not in available:
            continue
        sr = os.path.join(NAS, 'experiments', 'cap_budget_%dM' % B, 'search_results.json')
        d = json.load(open(sr, encoding='utf-8'))
        groups, dropped = {}, 0
        for t in (d.get('all_results') or []):
            a = t.get('architecture') or {}
            bb, dim, sc = a.get('backbone'), a.get('feature_dim'), t.get('score')
            if bb is None or dim is None or sc is None:
                continue
            sc = float(sc)
            if sc <= 0.0 or sc > 1e6:
                dropped += 1
                continue
            info = params.get(bb, {}).get(str(dim))
            if not info or 'total_m' not in info:
                continue
            groups.setdefault((bb, int(dim)), []).append(sc)

        ranked = []
        for (bb, dim), scs in groups.items():
            p = params[bb][str(dim)]['total_m']
            if p <= B:
                ranked.append({'backbone': bb, 'dim': dim, 'params_m': p,
                               'mean_score': statistics.mean(scs),
                               'std_score': statistics.pstdev(scs) if len(scs) > 1 else 0.0,
                               'n_valid': len(scs)})
        if not ranked:
            log('budget %2dM: no in-budget config with a valid score' % B)
            continue
        best = min(ranked, key=lambda r: r['mean_score'])
        best['score'] = best['mean_score']
        sel[str(B)] = best
        diag[str(B)] = {'n_valid_trials': sum(len(v) for v in groups.values()),
                        'n_dropped_degenerate': dropped,
                        'n_configs_ranked': len(ranked),
                        'runner_up': sorted(ranked, key=lambda r: r['mean_score'])[1:3]}
        log('budget %2dM -> %s + %dd (%.2fM, mean score %.4f, n=%d) | dropped %d degenerate | %d configs ranked'
            % (B, best['backbone'], best['dim'], best['params_m'], best['mean_score'],
               best['n_valid'], dropped, len(ranked)))
    json.dump({'selected': sel, 'diagnostics': diag},
              open(os.path.join(EXP, 'results', 'selected_configs.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    return sel


def run_trainings(sel):
    jobs = []
    for b, info in sorted(sel.items(), key=lambda kv: int(kv[0])):
        for c in CATS:
            ev = os.path.join(EXP, 'results', 'budget_%sM' % b, c, 'final_evaluation.json')
            if not os.path.exists(ev):
                jobs.append((b, info['backbone'], c, info['dim']))
    log('training jobs pending: %d (parallel=%d)' % (len(jobs), PARALLEL))

    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONUNBUFFERED='1')
    running, ok, bad = [], 0, []
    t0 = time.time()
    while jobs or running:
        while jobs and len(running) < PARALLEL:
            b, bb, c, info_dim = jobs.pop(0)
            fh = open(os.path.join(LOGS, '%sM_%s.log' % (b, c)), 'w', encoding='utf-8')
            p = subprocess.Popen([PY, '-u', os.path.join(EXP, 'code', 'train_worker.py'),
                                  b, bb, str(info_dim), c],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=NAS)
            running.append((p, b, c, fh, time.time()))
            log('  start %sM/%-11s %s' % (b, c, bb))
        time.sleep(15)
        still = []
        for p, b, c, fh, ts in running:
            if p.poll() is None:
                still.append((p, b, c, fh, ts))
                continue
            fh.close()
            if p.returncode == 0:
                ok += 1
                log('  ok    %sM/%-11s %.1f min  (done=%d left=%d)'
                    % (b, c, (time.time() - ts) / 60, ok, len(jobs) + len(still)))
            else:
                bad.append((b, c))
                log('  FAIL  %sM/%-11s rc=%d' % (b, c, p.returncode))
        running = still
    log('training finished in %.1f min | ok=%d failed=%d' % ((time.time() - t0) / 60, ok, len(bad)))
    return bad


def aggregate(sel):
    params = json.load(open(os.path.join(TRACE, 'params_by_backbone.json'), encoding='utf-8'))
    rows = []
    for B in BUDGETS:
        k = str(B)
        if k not in sel:
            continue
        info = sel[k]
        base = os.path.join(EXP, 'results', 'budget_%sM' % B)
        au, ap = [], []
        for c in CATS:
            f = os.path.join(base, c, 'final_evaluation.json')
            if os.path.exists(f):
                d = json.load(open(f, encoding='utf-8'))
                au.append(d['auroc'])
                ap.append(d['ap'])
        if not au:
            continue
        rows.append({'budget_m': B, 'backbone': info['backbone'], 'dim': info['dim'],
                     'params_m': info['params_m'], 'n_categories': len(au),
                     'mean_auroc': round(sum(au) / len(au), 4),
                     'mean_ap': round(sum(ap) / len(ap), 4),
                     'per_category': {c: None for c in CATS}})
        # fill per-category
        for c in CATS:
            f = os.path.join(base, c, 'final_evaluation.json')
            if os.path.exists(f):
                rows[-1]['per_category'][c] = round(
                    json.load(open(f, encoding='utf-8'))['auroc'], 4)
    out = os.path.join(EXP, 'results', 'capacity_curve.json')
    json.dump({'curve': rows}, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    log('=== CAPACITY CURVE ===')
    for r in rows:
        log('  %5.1fM (%-30s %3dd)  AUROC %.4f  AP %.4f  n=%d'
            % (r['params_m'], r['backbone'], r['dim'], r['mean_auroc'], r['mean_ap'],
               r['n_categories']))
    log('wrote ' + out)
    return rows


if __name__ == '__main__':
    log('pipeline start — waiting for searches')
    avail = wait_for_searches()
    sel = collect(avail)
    bad = run_trainings(sel)
    rows = aggregate(sel)
    log('PIPELINE DONE (failed pairs: %d)' % len(bad))
