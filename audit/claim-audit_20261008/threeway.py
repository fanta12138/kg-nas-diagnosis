"""Three-way re-read with an explicit source list.

The bundled threeway_reread.py takes a single glob, which cannot cover
main_springer.tex + sections/*.tex + supplementary/*.tex while excluding the
_archive/ backups (a **/*.tex glob would match the backups and report every
"old string" as still present). This script implements the same two-way
assertion with an explicit, auditable file list and the same Unicode folding.

Usage: python threeway.py --config audit_config.json --checks checks.json
"""
import argparse, json, re, sys, unicodedata
from pathlib import Path

FOLD = {
    "\u2212": "-", "\u2013": "--", "\u2014": "--", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u00a0": " ", "\u2009": " ", "\u200a": " ",
    "\u2002": " ", "\u2003": " ", "\u2264": "<=", "\u2265": ">=",
    "\u00b1": "+-", "\u2206": "Delta", "\u0394": "Delta",
    "\u03a3": "Sum", "\u2211": "Sum", "\u03b1": "alpha", "\u03b2": "beta",
    "\u03b3": "gamma", "\u223c": "~", "\u00d7": "x",
}


def normalize(s):
    for k, v in FOLD.items():
        s = s.replace(k, v)
    s = s.replace("\u00ad", "")
    s = re.sub(r"\s+", " ", s)
    return s


def loose(s):
    """collapse whitespace and drop spaces so '&' / layout noise cannot hide a match"""
    return re.sub(r"[\s\u00a0]+", "", normalize(s))


def dehyph(s):
    return s.replace("-", "")


def pdf_text(p):
    try:
        import fitz
        d = fitz.open(p)
        return " ".join(pg.get_text() for pg in d)
    except ImportError:
        pass
    try:
        from pypdf import PdfReader
        r = PdfReader(str(p))
        return " ".join((pg.extract_text() or "") for pg in r.pages)
    except ImportError:
        raise SystemExit("need pymupdf or pypdf")


def found(needle, text_loose, text_dh):
    n = loose(needle)
    return n in text_loose or dehyph(n) in text_dh


ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checks", required=True)
ap.add_argument("--pdf", default=None, help="override output_pdf")
ap.add_argument("--sources", default=None, help="comma-separated source files")
a = ap.parse_args()

cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
paper = Path(cfg["paper_dir"])
pdf = Path(a.pdf) if a.pdf else Path(cfg["output_pdf"])
checks = json.loads(Path(a.checks).read_text(encoding="utf-8"))

src_files = (a.sources.split(",") if a.sources
             else ["main_springer.tex", "sections/1_introduction.tex",
                   "sections/2_related_work.tex", "sections/3_method.tex",
                   "sections/4_experiments.tex", "sections/5_conclusion.tex",
                   "tables/BASELINE_ANALYSIS_TABLE.tex",
                   "supplementary/supplementary.tex"])
src_raw = ""
for f in src_files:
    p = paper / f
    if p.exists():
        src_raw += "\n" + p.read_text(encoding="utf-8", errors="replace")
    else:
        print("  !! missing source %s" % f, file=sys.stderr)

ptext_raw = pdf_text(pdf)
ptext, ptext_dh = loose(ptext_raw), dehyph(loose(ptext_raw))
stext, stext_dh = loose(src_raw), dehyph(loose(src_raw))

print("PDF   : %s (%d bytes)" % (pdf, pdf.stat().st_size))
print("sources: %d files" % len(src_files))
fails = 0
for c in checks:
    probs = []
    for s in c.get("must_appear", []):
        if not found(s, ptext, ptext_dh):
            probs.append("PDF lacks new string: %r" % s)
    for s in c.get("must_appear_in_sources", []):
        if not found(s, stext, stext_dh):
            probs.append("source lacks new string: %r" % s)
    for s in c.get("must_absent", []):
        if found(s, ptext, ptext_dh):
            probs.append("PDF still has old string: %r" % s)
        if c.get("sources", True) and found(s, stext, stext_dh):
            probs.append("source still has old string: %r" % s)
    if probs:
        fails += 1
        print("  [FAIL] %s" % c["name"])
        for p in probs:
            print("         " + p)
    else:
        print("  [OK  ] %s" % c["name"])

print("\n%s (%d failed)" % ("ALL_CHECKS_PASS" if not fails else "CHECKS_FAILED", fails))
sys.exit(1 if fails else 0)
