# -*- coding: utf-8 -*-
"""Citation audit: verify every bibliography entry against Crossref.

For each entry that carries a DOI we resolve it through the Crossref API and
compare the registered title, first author and year with what the .bib claims.
Entries without a DOI are listed separately and classified by type, because a
conference paper with no DOI is acceptable while a journal article with no DOI is
not (for this project's rules).

Nothing is written to the .bib: this only reports.

Usage: python audit_citations.py [path/to/references.bib]
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

import io, json, os, re, sys, time, unicodedata, urllib.parse, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
BIB = sys.argv[1] if len(sys.argv) > 1 else _os.path.join(_KGD_PAPER, 'references.bib')
MAILTO = 'audit@example.org'
OUT = os.path.join(HERE, 'citation_audit.json')


def norm(s):
    """Normalise for comparison: strip LaTeX, punctuation, case, accents."""
    if not s:
        return ''
    s = re.sub(r'\\[a-zA-Z]+\s*', ' ', s)          # LaTeX commands
    s = re.sub(r'[{}$\\]', '', s)
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def parse_bib(text):
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        kind, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)

        def field(name):
            mm = re.search(r'\b%s\s*=\s*[{"]?(.*?)[}"]?\s*,\s*\n' % name, body, re.S)
            if not mm:
                mm = re.search(r'\b%s\s*=\s*[{"]?(.*?)[}"]?\s*$' % name, body, re.S)
            return mm.group(1).strip() if mm else None

        entries.append(dict(kind=kind, key=key,
                            title=field('title'), author=field('author'),
                            year=field('year'), doi=field('doi'),
                            venue=field('journal') or field('booktitle'),
                            note=field('note'), eprint=field('eprint')))
    return entries


def crossref(doi):
    url = 'https://api.crossref.org/works/%s?mailto=%s' % (
        urllib.parse.quote(doi), MAILTO)
    req = urllib.request.Request(url, headers={'User-Agent': 'citation-audit/1.0'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)['message']


def main():
    text = io.open(BIB, encoding='utf-8').read()
    entries = parse_bib(text)
    print('bib entries: %d\n' % len(entries))

    report = []
    for i, e in enumerate(entries, 1):
        rec = dict(e)
        doi = (e['doi'] or '').strip()
        if not doi:
            rec['status'] = 'NO_DOI'
            rec['detail'] = '%s, %s' % (e['kind'], e['year'])
            print('%-34s NO DOI   (%s, %s)' % (e['key'], e['kind'], e['year']))
        else:
            try:
                msg = crossref(doi)
                ct = norm((msg.get('title') or [''])[0])
                bt = norm(e['title'])
                # tolerate subtitle/punctuation differences
                overlap = 0.0
                if ct and bt:
                    a, b = set(ct.split()), set(bt.split())
                    overlap = len(a & b) / max(1, min(len(a), len(b)))
                cy = None
                for k in ('published-print', 'published-online', 'issued', 'created'):
                    if k in msg and msg[k].get('date-parts'):
                        cy = msg[k]['date-parts'][0][0]
                        break
                ok_title = overlap >= 0.75
                # A one-year difference is normal (online-first vs printed issue);
                # only flag two years or more.
                try:
                    dy = abs(int(cy) - int(e['year'])) if cy else 0
                except Exception:
                    dy = 0
                ok_year = dy <= 1
                rec['year_delta'] = dy
                rec.update(crossref_title=(msg.get('title') or [''])[0],
                           crossref_year=cy, title_overlap=round(overlap, 3),
                           crossref_type=msg.get('type'))
                if ok_title and ok_year:
                    rec['status'] = 'OK'
                    print('%-34s OK        %s (%s)  title overlap %.2f'
                          % (e['key'], doi, cy, overlap))
                else:
                    rec['status'] = 'MISMATCH'
                    print('%-34s MISMATCH  %s' % (e['key'], doi))
                    if not ok_title:
                        print('      bib   : %s' % (e['title'] or '')[:96])
                        print('      xref  : %s' % (msg.get('title') or [''])[0][:96])
                        print('      overlap=%.2f' % overlap)
                    if not ok_year:
                        print('      year: bib=%s crossref=%s' % (e['year'], cy))
            except Exception as ex:
                # arXiv DOIs (10.48550/...) are registered with DataCite, not
                # Crossref, so a Crossref 404 is expected for them.
                if doi.startswith('10.48550/'):
                    try:
                        u = ('https://api.datacite.org/dois/%s'
                             % urllib.parse.quote(doi))
                        rq = urllib.request.Request(u, headers={'User-Agent': 'audit/1.0'})
                        at = json.load(urllib.request.urlopen(rq, timeout=30))['data']['attributes']
                        dt = (at.get('titles') or [{}])[0].get('title', '')
                        ov = 0.0
                        if dt and e['title']:
                            a_, b_ = set(norm(dt).split()), set(norm(e['title']).split())
                            ov = len(a_ & b_) / max(1, min(len(a_), len(b_)))
                        rec.update(crossref_title=dt, title_overlap=round(ov, 3),
                                   verified_via='DataCite')
                        rec['status'] = 'OK' if ov >= 0.75 else 'MISMATCH'
                        print('%-34s %-9s %s (DataCite, overlap %.2f)'
                              % (e['key'], rec['status'], doi, ov))
                        report.append(rec); time.sleep(0.4); continue
                    except Exception as e2:
                        ex = e2
                rec['status'] = 'UNRESOLVED'
                rec['detail'] = '%s: %s' % (type(ex).__name__, str(ex)[:120])
                print('%-34s UNRESOLVED  %s  (%s)' % (e['key'], doi, str(ex)[:60]))
        report.append(rec)
        time.sleep(0.4)

    bad = [r for r in report if r['status'] in ('MISMATCH', 'UNRESOLVED')]
    nod = [r for r in report if r['status'] == 'NO_DOI']
    print('\n' + '=' * 78)
    print('SUMMARY  ok=%d  mismatch=%d  unresolved=%d  no-doi=%d  (of %d)'
          % (sum(1 for r in report if r['status'] == 'OK'), len(bad),
             sum(1 for r in bad if r['status'] == 'UNRESOLVED'),
             len(nod), len(report)))
    print('=' * 78)
    if bad:
        print('\nMUST FIX:')
        for r in bad:
            print('  %-34s %s  %s' % (r['key'], r['status'], r.get('detail', '')))
    if nod:
        print('\nNO DOI (check each is a conference/preprint, not a journal):')
        for r in nod:
            print('  %-34s %-14s %-5s %s'
                  % (r['key'], r['kind'], r['year'],
                     (r['venue'] or r['note'] or '')[:52]))
    json.dump(report, open(OUT, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print('\nsaved -> %s' % OUT)


if __name__ == '__main__':
    main()
