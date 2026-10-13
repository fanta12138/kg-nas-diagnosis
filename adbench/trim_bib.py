# -*- coding: utf-8 -*-
"""Remove bibliography entries that no longer receive a \cite in the manuscript."""
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

import glob
import os
import re

ROOT = _KGD_PAPER
BIB = os.path.join(ROOT, 'references.bib')

cited = set()
for f in glob.glob(os.path.join(ROOT, 'sections', '_draft_v2', '*.tex')) + \
        [os.path.join(ROOT, 'main_v2.tex')]:
    txt = open(f, encoding='utf-8').read()
    for grp in re.findall(r'\\cite\{([^}]*)\}', txt):
        for k in grp.split(','):
            cited.add(k.strip())

txt = open(BIB, encoding='utf-8').read()
keys = [k.strip() for k in re.findall(r'@[a-zA-Z]+\{([^,]+),', txt)]
print('entries:', len(keys), 'cited:', len(cited))

drop = [k for k in keys if k not in cited]
for k in drop:
    pat = re.compile(r'\n?@[a-zA-Z]+\{' + re.escape(k) + r',.*?\n\}\n', re.S)
    txt, n = pat.subn('\n', txt)
    print('  %-22s removed=%d' % (k, n))

open(BIB, 'w', encoding='utf-8').write(txt)
left = re.findall(r'@[a-zA-Z]+\{([^,]+),', txt)
print('entries now:', len(left))
missing = [c for c in cited if c.strip() not in [x.strip() for x in left]]
print('cited-but-missing:', missing)
