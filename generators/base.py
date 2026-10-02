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

    Regions are returned with *unwrapped* world coordinates, so a shape that
    crosses the right edge is evaluated continuously and its pixels land on
    the left side of the image when tiling is on.
    """

    def __init__(self, width_m, height_m, nx, ny, tiling=False):
        self.width_m, self.height_m = float(width_m), float(height_m)
        self.nx, self.ny = int(nx), int(ny)
        self.tiling = bool(tiling)
        self.pmx = self.width_m / self.nx
        self.pmy = self.height_m / self.ny
        self.pm = 0.5 * (self.pmx + self.pmy)     # meters per pixel

    def region(self, x0, y0, x1, y1, pad=0.0):
        i0 = math.floor((x0 - pad) / self.pmx - 0.5)
        i1 = math.ceil((x1 + pad) / self.pmx + 0.5)
        j0 = math.floor((y0 - pad) / self.pmy - 0.5)
        j1 = math.ceil((y1 + pad) / self.pmy + 0.5)
        if self.tiling:
            if i1 - i0 >= self.nx:
                i0, i1 = 0, self.nx
            if j1 - j0 >= self.ny:
                j0, j1 = 0, self.ny
            if i1 <= i0 or j1 <= j0:
                return None
            ii, jj = np.arange(i0, i1), np.arange(j0, j1)
            if i0 >= 0 and i1 <= self.nx and j0 >= 0 and j1 <= self.ny:
                key = (slice(j0, j1), slice(i0, i1))
            else:
                key = np.ix_(jj % self.ny, ii % self.nx)
        else:
            i0, i1 = max(i0, 0), min(i1, self.nx)
            j0, j1 = max(j0, 0), min(j1, self.ny)
            if i1 <= i0 or j1 <= j0:
                return None
            ii, jj = np.arange(i0, i1), np.arange(j0, j1)
            key = (slice(j0, j1), slice(i0, i1))
        X = ((ii + 0.5) * self.pmx)[None, :]
        Y = ((jj + 0.5) * self.pmy)[:, None]
        return key, X, Y


class Generator:
    name = "base"

    def height(self, doc, grid: Grid, emission=False):
        """Return (height float32 [ny, nx] in meters, ids int32 [ny, nx] or None,
        emission float32 [ny, nx, 3] linear radiance or None,
        wire coverage mask float32 [ny, nx] or None).

        Emission is only computed when `emission` is true."""
        raise NotImplementedError
