"""Generator interface.

A generator turns a Document into a height field (meters) on a pixel grid.
Panels are the first generator; water, ocean and flow generators can plug in
later and reuse the same bake, export and preview pipeline.
"""
from __future__ import annotations

import math

import numpy as np


class Grid:
    """Pixel grid over the canvas, with optional wrap-around (seamless tiling).

    `nx`, `ny` define the pixel size over the whole canvas. By default the
    arrays cover the whole canvas; with `window=(i0, j0, w, h)` they cover only
    that block of pixels (used for zoomed-in detail bakes). With tiling, the
    window may extend past the canvas edge; those pixels show the wrapped copy.

    Regions are returned with *unwrapped* world coordinates, so a shape that
    crosses the right edge is evaluated continuously and its pixels land on
    the left side of the image when tiling is on.
    """

    def __init__(self, width_m, height_m, nx, ny, tiling=False, window=None):
        self.width_m, self.height_m = float(width_m), float(height_m)
        self.nx, self.ny = int(nx), int(ny)
        self.tiling = bool(tiling)
        self.pmx = self.width_m / self.nx
        self.pmy = self.height_m / self.ny
        self.pm = 0.5 * (self.pmx + self.pmy)     # meters per pixel
        if window is None:
            window = (0, 0, self.nx, self.ny)
        self.wi0, self.wj0, self.wn, self.hn = (int(v) for v in window)
        self.windowed = (self.wi0, self.wj0, self.wn, self.hn) != (0, 0, self.nx, self.ny)
        if self.tiling:
            self.wn, self.hn = min(self.wn, self.nx), min(self.hn, self.ny)

    @property
    def shape(self):
        """Shape of the arrays this grid fills: (rows, cols)."""
        return self.hn, self.wn

    def local_index(self, fx, fy):
        """Full-canvas fractional pixel coordinates -> array coordinates."""
        return fx - self.wi0, fy - self.wj0

    def _axis(self, a0, a1, n, w0, wn):
        """Unwrapped canvas indices [a0, a1) -> (unwrapped kept, array index)."""
        if self.tiling:
            if a1 - a0 >= n:
                a0, a1 = 0, n
            u = np.arange(a0, a1)
            loc = (u - w0) % n
        else:
            u = np.arange(max(a0, 0), min(a1, n))
            loc = u - w0
        keep = (loc >= 0) & (loc < wn)
        return u[keep], loc[keep]

    def region(self, x0, y0, x1, y1, pad=0.0):
        i0 = math.floor((x0 - pad) / self.pmx - 0.5)
        i1 = math.ceil((x1 + pad) / self.pmx + 0.5)
        j0 = math.floor((y0 - pad) / self.pmy - 0.5)
        j1 = math.ceil((y1 + pad) / self.pmy + 0.5)
        ii, li = self._axis(i0, i1, self.nx, self.wi0, self.wn)
        jj, lj = self._axis(j0, j1, self.ny, self.wj0, self.hn)
        if not len(ii) or not len(jj):
            return None
        contiguous = (li[-1] - li[0] == len(li) - 1 and np.all(np.diff(li) == 1)
                      and lj[-1] - lj[0] == len(lj) - 1 and np.all(np.diff(lj) == 1))
        if contiguous:
            key = (slice(int(lj[0]), int(lj[-1]) + 1), slice(int(li[0]), int(li[-1]) + 1))
        else:
            key = np.ix_(lj, li)
        X = ((ii + 0.5) * self.pmx)[None, :]
        Y = ((jj + 0.5) * self.pmy)[:, None]
        return key, X, Y


class Generator:
    name = "base"

    def height(self, doc, grid: Grid, emission=False):
        """Return (height float32 grid.shape in meters, ids int32 or None,
        emission float32 grid.shape + (3,) linear radiance or None,
        wire coverage mask float32 or None).

        Emission is only computed when `emission` is true."""
        raise NotImplementedError
