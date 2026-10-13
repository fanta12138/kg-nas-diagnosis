"""Parameter-efficiency analysis for KG-NAS against all compared methods."""
import json

# (method, I-AUROC %, params M, source)
M = [
    ('Dinomaly',             99.6,  86.0, 'tab:sota_comparison'),
    ('SimpleNet',            98.0,  12.0, 'tab:sota_comparison'),
    ('EfficientAD',          97.5,   8.0, 'tab:sota_comparison'),
    ('CFA',                  89.75, 12.0, 'tab:efficiency_comparison'),
    ('KG-NAS (ours)',        82.40,  5.3, 'measured 5.33-5.41'),
    ('Reverse Distillation', 80.49, 15.0, 'tab:efficiency_comparison'),
    ('PaDiM',                77.92, 12.0, 'tab:efficiency_comparison'),
    ('STPM',                 51.10, 15.0, 'tab:efficiency_comparison'),
    ('PatchCore',            50.59, 12.0, 'tab:efficiency_comparison'),
]
BASE = 82.40
BASE_P = 5.3

print('=' * 88)
print('%-22s %8s %9s %12s %14s %s' % ('method', 'AUROC', 'params M', 'AUROC/M', 'vs ours (dP)', 'verdict'))
print('=' * 88)
rows = []
for name, acc, p, src in sorted(M, key=lambda x: x[2]):
    eff = acc / p
    dp = p / BASE_P
    if name.startswith('KG-NAS'):
        verdict = '<-- OURS (smallest)'
    else:
        d_acc = acc - BASE
        if p <= BASE_P and d_acc <= 0:
            verdict = 'dominated by us'
        elif p >= BASE_P and d_acc >= 0:
            verdict = 'dominates us (+%.1fpp for %.1fx size)' % (d_acc, dp)
        else:
            verdict = 'trade-off (%.1fpp, %.1fx size)' % (d_acc, dp)
    rows.append((name, acc, p, eff, dp, verdict))
    print('%-22s %8.2f %9.1f %12.2f %13.2fx  %s' % (name, acc, p, eff, dp, verdict))

print()
print('=' * 88)
print('PARETO FRONTIER (no other method is both smaller AND more accurate)')
print('=' * 88)
front = []
best = -1
for name, acc, p, eff, dp, v in rows:          # already sorted by params asc
    if acc > best:
        front.append((name, acc, p))
        best = acc
for n, a, p in front:
    print('   %-22s %6.2f%%  %6.1f M' % (n, a, p))

print()
print('=' * 88)
print('HEAD-TO-HEAD: what does each parameter buy?')
print('=' * 88)
print('  ours            : 5.3 M -> 82.40%%')
for name, acc, p, *_ in sorted(M, key=lambda x: x[2]):
    if name.startswith('KG-NAS'):
        continue
    if acc > BASE:
        gain = acc - BASE
        cost = p - BASE_P
        print('  %-15s : %5.1f M -> %.2f%%   (+%.1fpp for +%.1fM = %.2f pp/M)'
              % (name, p, acc, gain, cost, gain / cost))
    else:
        print('  %-15s : %5.1f M -> %.2f%%   (loses %.1fpp while being %.1fx larger)'
              % (name, p, acc, BASE - acc, p / BASE_P))

print()
print('=' * 88)
print('SUMMARY')
print('=' * 88)
ours_eff = BASE / BASE_P
runner = max((a / p, n) for n, a, p, *_ in rows if not n.startswith('KG-NAS'))
print('  best AUROC per parameter : ours %.2f  vs next best %s %.2f  (+%.0f%%)'
      % (ours_eff, runner[1], runner[0], 100 * (ours_eff / runner[0] - 1)))
print('  strictly Pareto-dominates: %s' %
      ', '.join(n for n, a, p, _, _, v in rows if v == 'dominated by us'))
