"""Sci-fi panel generator: composites the panel stack into a height field,
plus an optional emission map (linear RGB radiance)."""
from __future__ import annotations

import numpy as np

from core.panel import shape_height, groove_cut
from core import wire as wiremod
from .base import Generator, Grid


def _cover(d, pm):
    """Anti-aliased coverage from an inside distance (meters)."""
    return np.clip(d / pm + 0.5, 0.0, 1.0)


class PanelGenerator(Generator):
    name = "panels"

    def height(self, doc, grid: Grid, emission=False):
        H = np.zeros(grid.shape, np.float32)
        ids = np.zeros(grid.shape, np.int32)
        emit = emission and any(p.visible and p.has_emission() for p in doc.panels)
        E = np.zeros(grid.shape + (3,), np.float32) if emission else None
        pm = grid.pm
        for index, p in enumerate(doc.panels):
            if not p.visible:
                continue
            x0, y0, x1, y1 = p.rect
            reg = grid.region(x0, y0, x1, y1)
            if reg is None:
                continue
            key, X, Y = reg
            d = -p.sdf(X, Y)
            inside = d > 0
            if not inside.any():
                continue
            h = shape_height(d, p.bevel, p.depth, pm)
            g = groove_cut(d, p.groove, pm) if p.groove.enabled else 0.0
            sub = H[key]
            if p.mode == "inset":
                sub = sub - h - g
            elif p.mode == "max":
                sub = np.where(inside, np.maximum(sub, h), sub) - g
            else:
                sub = sub + h - g
            H[key] = sub
            idsub = ids[key]
            ids[key] = np.where(inside, index + 1, idsub)
            if emit:
                self._panel_emission(E, key, d, p, pm)
            for det in p.details:
                self._detail(H, E if emit else None, grid, p, det)
        # Wires lie on top of the panels (and on each other, in order).
        M = np.zeros(grid.shape, np.float32)
        wiremod.composite(H, ids, E, M, grid, doc, len(doc.panels))
        return H, ids, E, M

    @staticmethod
    def _panel_emission(E, key, d, p, pm):
        """The panel surface hides emission beneath it, then adds its own."""
        cov = _cover(d, pm)
        sub = E[key] * (1.0 - cov)[..., None]
        groove_shape = None
        if p.groove.enabled:
            groove_shape = groove_cut(d, p.groove, pm) / max(p.groove.depth, 1e-9)
        face = p.emission.radiance()
        if face is not None:
            bw = p.bevel.width
            m = _cover(d - bw, pm)                       # flat top inside the bevel
            if p.emission.diffuser > 0:
                half = max(min(p.w, p.h) * 0.5 - bw, pm)
                t = np.clip((d - bw) / (0.5 * half), 0.0, 1.0)
                s = t * t * (3 - 2 * t)
                m = m * (1.0 - p.emission.diffuser * (1.0 - s))
            if groove_shape is not None and not p.groove.emission.enabled:
                m = m * (1.0 - groove_shape)
            sub = sub + m[..., None] * face
        if groove_shape is not None:
            glow = p.groove.emission.radiance()
            if glow is not None:
                sub = sub * (1.0 - groove_shape)[..., None] + groove_shape[..., None] * glow
        E[key] = sub

    @staticmethod
    def _detail(H, E, grid, panel, det):
        r = det.bound_radius()
        sign = -1.0 if det.mode == "inset" else 1.0
        rad = det.emission.radiance() if E is not None else None
        for cx, cy in det.centers(*panel.rect):
            reg = grid.region(cx - r, cy - r, cx + r, cy + r)
            if reg is None:
                continue
            key, X, Y = reg
            d = -det.local_sdf(X - cx, Y - cy)
            if not (d > 0).any():
                continue
            h = shape_height(d, det.bevel, det.depth, grid.pm)
            H[key] = H[key] + sign * h
            if E is not None:
                cov = _cover(d, grid.pm)[..., None]
                sub = E[key] * (1.0 - cov)
                if rad is not None:
                    sub = sub + cov * rad
                E[key] = sub
