# -*- coding: utf-8 -*-
"""Report abstract / keyword / title statistics for the manuscript."""
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

import re

t = open(_os.path.join(_KGD_PAPER, 'main_v2.tex'), encoding='utf-8').read()

ab = re.search(r'\\abstract\{(.*?)\}\s*\n', t, re.S)
if ab:
    body = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?', ' ', ab.group(1))
    body = body.replace('$', ' ').replace('{', ' ').replace('}', ' ')
    words = [w for w in body.split() if any(c.isalnum() for c in w)]
    print('abstract: %d words, %d chars' % (len(words), len(ab.group(1))))
else:
    print('abstract not matched')

kw = re.search(r'\\keywords\{(.*?)\n?\}', t, re.S)
print('keywords:', ' '.join(kw.group(1).split()) if kw else 'not matched')

ti = re.search(r'\\title\[[^\]]*\]\{(.*?)\n?\}', t, re.S)
print('title:', ' '.join(ti.group(1).split()) if ti else 'not matched')
