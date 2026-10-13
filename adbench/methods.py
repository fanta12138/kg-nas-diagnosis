# -*- coding: utf-8 -*-
"""Training-free anomaly detectors used by the probes.

All are *inference only*: build a model of normality from the normal training
images, then score test images. Kept cheap on purpose so every experiment can be
repeated with many seeds, which is the point of the whole exercise.

  PatchCore  -- memory bank of normal patch features + nearest-neighbour scoring,
                with the *locality radius* as an explicit parameter
  KNNGlobal  -- global pooled-descriptor kNN, a deliberately global baseline

The locality parameter is what makes this harness useful for the probes: setting
`locality_radius=None` reproduces the standard global bank, and small radii make
the model progressively more local. That turns "how local is this method" from a
hand-wave into a measurable axis.
"""
import numpy as np
import torch
import torch.nn.functional as F

import common as C


def _grid(bb, batch_np, g, device):
    """Extract features already resized to g x g and concatenate the scales."""
    feats = bb.features(batch_np, grid=g)
    return torch.cat([torch.as_tensor(np.asarray(f), dtype=torch.float32,
                                      device=device) for f in feats], dim=1)


class PatchCore:
    """Memory bank of normal patch features with nearest-neighbour scoring.

    radius = None : one global bank of `bank_size` rows sampled from all cells
                    (the standard PatchCore memory bank)
    radius = r    : a query patch may only match train patches within r cells
                    (r = 0 is strictly local)
    """

    def __init__(self, backbone, locality_radius=None, bank_size=8192,
                 max_per_cell=64, g=32, seed=0):
        self.bb = backbone
        self.radius = locality_radius
        self.bank_size = bank_size
        self.max_per_cell = max_per_cell
        self.g = g
        self.seed = seed
        self.dev = backbone.device
        self.bank = None          # global bank, (M, C) tensor
        self.cells = None         # per-cell banks, dict (i,j) -> (m, C) tensor
        self.cand = None          # per query cell -> list of cell keys

    # ------------------------------------------------------------------ fit
    def fit(self, train_imgs):
        g, dev = self.g, self.dev
        f = _grid(self.bb, train_imgs, g, dev)                    # (N,C,g,g)
        n, c, gh, gw = f.shape
        rng = np.random.default_rng(self.seed)

        if self.radius is None:
            patch = f.permute(0, 2, 3, 1).reshape(n * gh * gw, c)
            if patch.shape[0] > self.bank_size:
                idx = rng.choice(patch.shape[0], self.bank_size, replace=False)
                patch = patch[torch.as_tensor(idx, device=dev)]
            self.bank = patch.contiguous()
        else:
            cells = {}
            for i in range(gh):
                for j in range(gw):
                    v = f[:, :, i, j]                            # (N,C)
                    if self.max_per_cell and v.shape[0] > self.max_per_cell:
                        sel = rng.choice(v.shape[0], self.max_per_cell, replace=False)
                        v = v[torch.as_tensor(sel, device=dev)]
                    cells[(i, j)] = v.contiguous()
            self.cells = cells
            r = self.radius
            self.cand = {}
            for i in range(gh):
                for j in range(gw):
                    self.cand[(i, j)] = [
                        (a, b)
                        for a in range(max(0, i - r), min(gh, i + r + 1))
                        for b in range(max(0, j - r), min(gw, j + r + 1))]
        self.shape = (gh, gw)
        return self

    # ---------------------------------------------------------------- score
    @torch.no_grad()
    def score(self, test_imgs):
        g, dev = self.g, self.dev
        f = _grid(self.bb, test_imgs, g, dev)                     # (N,C,g,g)
        n, c, gh, gw = f.shape

        if self.radius is None:
            b = self.bank
            bn = (b * b).sum(1)[None, :]
            q = f.permute(0, 2, 3, 1).reshape(n * gh * gw, c)
            # Chunk the queries: a (n_patch x bank) distance matrix is n_patch*bank
            # floats, so a large bank would otherwise need tens of GB.
            block = max(1, int(3e7 // max(1, b.shape[0])))
            out = torch.empty(q.shape[0], dtype=torch.float32, device=dev)
            for s0 in range(0, q.shape[0], block):
                qb = q[s0:s0 + block]
                qn = (qb * qb).sum(1)[:, None]
                d2 = qn + bn - 2.0 * (qb @ b.T)
                out[s0:s0 + block] = torch.sqrt(
                    torch.clamp(d2.min(dim=1).values, min=0))
            maps = out.reshape(n, gh, gw)
        else:
            maps = torch.empty((n, gh, gw), dtype=torch.float32, device=dev)
            for i in range(gh):
                for j in range(gw):
                    b = torch.cat([self.cells[k] for k in self.cand[(i, j)]], 0)
                    q = f[:, :, i, j]                              # (N,C)
                    qn = (q * q).sum(1)[:, None]
                    bn = (b * b).sum(1)[None, :]
                    d2 = qn + bn - 2.0 * (q @ b.T)
                    maps[:, i, j] = torch.sqrt(torch.clamp(d2.min(dim=1).values, min=0))

        sm = F.avg_pool2d(F.pad(maps[:, None], (1, 1, 1, 1), mode='reflect'), 3, 1)
        img = sm.amax(dim=(1, 2, 3))
        return img.double().cpu().numpy(), maps.cpu().numpy()


class KNNGlobal:
    """kNN over a pooled global descriptor. Deliberately global and weak: an
    upper bound on 'no spatial reasoning at all'."""

    def __init__(self, backbone, k=5, g=32):
        self.bb = backbone
        self.k = k
        self.g = g
        self.dev = backbone.device
        self.train = None

    @torch.no_grad()
    def fit(self, train_imgs):
        f = _grid(self.bb, train_imgs, self.g, self.dev)
        self.train = F.normalize(f.mean(dim=(2, 3)), dim=1).contiguous()
        return self

    @torch.no_grad()
    def score(self, test_imgs):
        f = _grid(self.bb, test_imgs, self.g, self.dev)
        d = F.normalize(f.mean(dim=(2, 3)), dim=1)
        sim = d @ self.train.T
        k = min(self.k, sim.shape[1])
        top = torch.topk(sim, k, dim=1).values
        img = (1.0 - top.mean(dim=1)).double().cpu().numpy()
        g = self.g
        flat = np.repeat(img[:, None, None], g * g, 2).reshape(len(d), g, g)
        return img, flat


METHODS = {'patchcore': PatchCore, 'knn_global': KNNGlobal}
