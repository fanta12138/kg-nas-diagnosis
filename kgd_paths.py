# -*- coding: utf-8 -*-
"""Single source of truth for filesystem roots.

The scripts in this repository were written while the project lived at fixed
absolute paths. Every one of those original roots is redirected here, so the
release runs from any location and on any drive.

Environment overrides (all optional)
------------------------------------
``KGD_DATASETS_ROOT``
    Where the benchmark datasets live. Defaults to ``<repo>/datasets``.
    Expected layout::

        $KGD_DATASETS_ROOT/datasets/mvtec_anomaly_detection/...
        $KGD_DATASETS_ROOT/datasets-2/...            (VisA, official 1cls.csv)
        $KGD_DATASETS_ROOT/mvtec_loco_anomaly_detection/...

``KGD_WORK_ROOT``
    Where scripts write their outputs. Defaults to the repository root, which
    reproduces the original layout (``<repo>/proxy_diag/...``).

``KGD_PAPER_ROOT``
    Where the manuscript lives. Defaults to ``<repo>/paper``.
"""
import os

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASETS_ROOT = os.environ.get('KGD_DATASETS_ROOT', REPO_ROOT)
WORK_ROOT = os.environ.get('KGD_WORK_ROOT', REPO_ROOT)
PAPER_ROOT = os.environ.get('KGD_PAPER_ROOT', os.path.join(REPO_ROOT, 'paper'))

# Convenience aliases used by the patched scripts.
KGD_ROOT = REPO_ROOT
KGD_DATA = DATASETS_ROOT
KGD_WORK = WORK_ROOT
KGD_PAPER = PAPER_ROOT


def bootstrap():
    """Report the resolved roots; call this from a script's ``__main__`` if you
    want the paths printed for troubleshooting."""
    for k, v in (('repo', REPO_ROOT), ('datasets', DATASETS_ROOT),
                 ('work', WORK_ROOT), ('paper', PAPER_ROOT)):
        print('%-9s %s' % (k, v))


if __name__ == '__main__':
    bootstrap()
