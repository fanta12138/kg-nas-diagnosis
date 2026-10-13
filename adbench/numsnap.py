# -*- coding: utf-8 -*-
"""Extract every numeric token from the draft sections, so a compression pass can
be checked for silently dropped numbers.

usage:
  python numsnap.py save  <dir> <out.json>
  python numsnap.py diff  <before.json> <dir>
"""
import glob
import json
import os
import re
import sys

NUM = re.compile(r'\d+(?:\.\d+)?')
TEX = re.compile(r'(?<!\\)%.*')          # strip comments
NEG_LOOK = ('00',)                       # ignore pure page/format noise like 100, 200


def strip_tex(text):
    out = []
    for line in text.splitlines():
        line = TEX.sub('', line) if not line.startswith('%') else ''
        out.append(line)
    return '\n'.join(out)


def collect(d):
    res = {}
    for f in sorted(glob.glob(os.path.join(d, '*.tex'))):
        t = strip_tex(open(f, encoding='utf-8').read())
        res[os.path.basename(f)] = NUM.findall(t)
    return res


def main():
    mode = sys.argv[1]
    if mode == 'save':
        d, out = sys.argv[2], sys.argv[3]
        json.dump(collect(d), open(out, 'w'), indent=1)
        print('saved', {k: len(v) for k, v in collect(d).items()})
    elif mode == 'diff':
        before = json.load(open(sys.argv[2], encoding='utf-8'))
        after = collect(sys.argv[3])
        bad = False
        for f in sorted(set(before) | set(after)):
            b = before.get(f, [])
            a = after.get(f, [])
            missing = []
            for x in sorted(set(b), key=len, reverse=True):
                if b.count(x) > a.count(x):
                    missing.append('%s x%d' % (x, b.count(x) - a.count(x)))
            if missing:
                bad = True
                print('LOST in %s: %s' % (f, ', '.join(missing[:40])))
            print('%-24s before %4d  after %4d' % (f, len(b), len(a)))
        print('\nRESULT:', 'NUMBERS LOST — review the lines above' if bad
              else 'no numeric token lost')
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
