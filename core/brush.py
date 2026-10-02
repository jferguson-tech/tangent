"""Hand-painted weathering: brush strokes stored in the document.

A stroke is a polyline in canvas meters with a radius, strength, hardness, a
channel (which effect it paints) and a mode (add or erase). Strokes are
rasterized in order onto the weathering simulation grid, so later strokes
paint over earlier ones, and an erase stroke also removes the simulated
effect under it. Being stored as vectors, strokes stay sharp-edged at any
texel density, stay editable (settings, order, visibility) and are part of
undo, save and export like everything else.

A stroke painted on a panel is attached to it: `points` are where it was
painted and `origin` is where the panel's top-left corner was then, so when
the panel moves the stroke moves with it.
"""
from __future__ import annotations

import math
import uuid

import numpy as np

from .panel import _f, _choice

CHANNELS = ("rust", "dirt", "wear", "streaks", "oil", "soot", "heat")
CHANNEL_LABELS = {"rust": "Rust", "dirt": "Dirt", "wear": "Edge wear / chips", "streaks": "Grime streaks",
                  "oil": "Oil stain", "soot": "Soot", "heat": "Heat tint"}
MODES = ("add", "erase")
MAX_STROKES = 600
MAX_POINTS = 4000           # per stroke


def stroke_from(d):
    d = d if isinstance(d, dict) else {}
    mode = _choice(d.get("mode"), MODES, "add")
    chans = CHANNELS + (("all",) if mode == "erase" else ())
    pts = []
    for q in (d.get("points") or [])[:MAX_POINTS]:
        try:
            x, y = float(q[0]), float(q[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            pts.append([x, y])
    pid = d.get("panel")
    org = d.get("origin")
    try:
        origin = [float(org[0]), float(org[1])] if pid else None
        if origin and not all(math.isfinite(v) for v in origin):
            origin = None
    except (TypeError, ValueError, IndexError):
        origin = None
    return {"id": str(d.get("id") or ("s" + uuid.uuid4().hex[:8]))[:40],
            "channel": _choice(d.get("channel"), chans, "rust"), "mode": mode,
            "radius": _f(d.get("radius"), 0.03, 0.001, 1.0),
            "strength": _f(d.get("strength"), 0.8, 0.0, 1.0),
            "hardness": _f(d.get("hardness"), 0.5, 0.0, 1.0),
            "visible": bool(d.get("visible", True)),
            "panel": str(pid)[:40] if pid and origin else None,
            "origin": origin if pid else None,
            "points": pts}


def strokes_from(lst):
    out = [stroke_from(s) for s in (lst or [])[:MAX_STROKES] if isinstance(s, dict)]
    return [s for s in out if s["points"]]


def resolve(strokes, doc):
    """Visible strokes with their points moved along with their panels."""
    out = []
    for s in strokes:
        if not s.get("visible", True):
            continue
        pts = s["points"]
        if s.get("panel") and s.get("origin"):
            p = doc.find(s["panel"])
            if p is not None:
                dx, dy = p.x - s["origin"][0], p.y - s["origin"][1]
                if dx or dy:
                    pts = [[x + dx, y + dy] for x, y in pts]
        out.append({k: s[k] for k in ("channel", "mode", "radius", "strength", "hardness")} | {"points": pts})
    return out


def _distance(X, Y, P):
    """Distance from points (X row, Y column) to polyline P (n, 2)."""
    best = np.full(np.broadcast(X, Y).shape, np.inf)
    if len(P) == 1:
        return np.hypot(X - P[0, 0], Y - P[0, 1])
    for (ax, ay), (bx, by) in zip(P[:-1], P[1:]):
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        if l2 < 1e-18:
            t = 0.0
        else:
            t = np.clip(((X - ax) * dx + (Y - ay) * dy) / l2, 0.0, 1.0)
        np.minimum(best, np.hypot(X - ax - t * dx, Y - ay - t * dy), out=best)
    return best


def _thin(P, step):
    """Drop points closer than `step` to the last kept one (strokes are dense)."""
    keep = [P[0]]
    for p in P[1:]:
        if math.hypot(p[0] - keep[-1][0], p[1] - keep[-1][1]) >= step:
            keep.append(p)
    if len(P) > 1 and keep[-1] is not P[-1]:
        keep.append(P[-1])
    return np.asarray(keep, np.float64)


def rasterize(strokes, grid):
    """Paint `strokes` on `grid` (generators.base.Grid). Returns
    {channel: (add, erase)} with fields in [0, 1] of grid.shape; channels
    without strokes are absent."""
    out = {}
    shape = grid.shape

    def layer(ch):
        if ch not in out:
            out[ch] = (np.zeros(shape, np.float32), np.zeros(shape, np.float32))
        return out[ch]

    for s in strokes:
        r = s["radius"]
        P = _thin(s["points"], max(r * 0.2, grid.pm))
        x0, y0 = P.min(axis=0) - r
        x1, y1 = P.max(axis=0) + r
        reg = grid.region(x0, y0, x1, y1)
        if reg is None:
            continue
        key, X, Y = reg
        t = _distance(X, Y, P) / r
        h = min(s["hardness"], 0.98)
        u = np.clip((t - h) / (1.0 - h), 0.0, 1.0)
        cov = (s["strength"] * (1.0 - u * u * (3 - 2 * u))).astype(np.float32)
        chans = CHANNELS if s["channel"] == "all" else (s["channel"],)
        for ch in chans:
            if s["mode"] == "erase" and ch not in out:
                layer(ch)
            add, era = layer(ch)
            if s["mode"] == "add":
                a = add[key]
                add[key] = 1.0 - (1.0 - a) * (1.0 - cov)
            else:
                add[key] = add[key] * (1.0 - cov)
                e = era[key]
                era[key] = 1.0 - (1.0 - e) * (1.0 - cov)
    return out
