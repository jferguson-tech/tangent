"""Wires: parametric tubes, ribbon cables and hoses lying on the panels.

A wire has two endpoints, each attached to a panel (anchor point + offset, so
it follows the panel when it moves or resizes) or free on the canvas. Its
shape comes from one of two modes:

  sim    a rope simulated in the browser (static/js/wiresim.js); the settled
         polyline is stored in `points`
  route  a Manhattan (optionally 45-degree) path found by core/route.py

Baking composites wires on top of the panel height field: each wire rests on
whatever is beneath it (panels, then earlier wires), so crossings stack.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from . import sdf
from .panel import Emission, Bevel, shape_height, _f, _choice

WIRE_MODES = ("sim", "route")
WIRE_PROFILES = ("round", "ribbon", "hose")
END_ANCHORS = ("tl", "t", "tr", "l", "c", "r", "bl", "b", "br")
DEFAULT_SIM = {"gravity_angle": 90.0, "gravity": 1.0, "collide": True, "auto_resettle": False}


def new_wire_id():
    return "w" + uuid.uuid4().hex[:8]


def anchor_point(rect, anchor):
    x0, y0, x1, y1 = rect
    x = x0 if anchor.endswith("l") else x1 if anchor.endswith("r") else (x0 + x1) * 0.5
    y = y0 if anchor.startswith("t") else y1 if anchor.startswith("b") else (y0 + y1) * 0.5
    return x, y


@dataclass
class End:
    panel: Optional[str] = None   # attached panel id, or None for a free point
    anchor: str = "c"
    dx: float = 0.0               # offset from the anchor point (m)
    dy: float = 0.0
    x: float = 0.0                # free position, or last resolved position (fallback)
    y: float = 0.0

    def resolve(self, doc):
        if self.panel:
            p = doc.find(self.panel)
            if p is not None:
                ax, ay = anchor_point(p.rect, self.anchor)
                return ax + self.dx, ay + self.dy
        return self.x, self.y

    @classmethod
    def from_dict(cls, d) -> "End":
        d = d or {}
        pid = d.get("panel")
        return cls(panel=str(pid)[:40] if pid else None,
                   anchor=_choice(d.get("anchor"), END_ANCHORS, "c"),
                   dx=_f(d.get("dx"), 0, -100, 100), dy=_f(d.get("dy"), 0, -100, 100),
                   x=_f(d.get("x"), 0, -100, 100), y=_f(d.get("y"), 0, -100, 100))


@dataclass
class Wire:
    id: str = field(default_factory=new_wire_id)
    name: str = "Wire"
    mode: str = "sim"
    a: End = field(default_factory=End)
    b: End = field(default_factory=End)
    profile: str = "hose"             # default wire: a ribbed hose
    radius: float = 0.014             # tube radius / ribbon thickness basis (m)
    slack: float = 1.15               # sim: rope length / straight distance
    corner_radius: float = 0.025      # route: bend radius (m)
    allow45: bool = False             # route: allow 45-degree runs
    clearance: float = 0.01           # route: gap kept to earlier routed wires (m)
    connectors: bool = True
    clip_spacing: float = 0.0         # 0 = no clips
    bundle: int = 1                   # parallel strands (harness)
    bundle_spacing: float = 0.0       # 0 = automatic (just touching)
    emission: Emission = field(default_factory=Emission)
    points: List[List[float]] = field(default_factory=list)   # centerline polyline (m)
    settled: bool = False             # sim: points come from a finished simulation
    locked: bool = False
    visible: bool = True

    # ---- geometry helpers ------------------------------------------------
    def strand_spacing(self):
        if self.bundle_spacing > 0:
            return self.bundle_spacing
        return (2.0 * self.half_width()) * 1.02

    def half_width(self):
        """In-plane half width of one strand."""
        return self.radius * (2.5 if self.profile == "ribbon" else 1.0)

    def total_half_width(self):
        n = max(1, self.bundle)
        return (n - 1) * 0.5 * self.strand_spacing() + self.half_width()

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d) -> "Wire":
        d = d or {}
        pts = []
        for q in (d.get("points") or [])[:4000]:
            try:
                pts.append([float(q[0]), float(q[1])])
            except (TypeError, ValueError, IndexError):
                continue
        return cls(id=str(d.get("id") or new_wire_id())[:40],
                   name=str(d.get("name", "Wire"))[:64],
                   mode=_choice(d.get("mode"), WIRE_MODES, "sim"),
                   a=End.from_dict(d.get("a")), b=End.from_dict(d.get("b")),
                   profile=_choice(d.get("profile"), WIRE_PROFILES, "hose"),
                   radius=_f(d.get("radius"), 0.014, 0.0005, 0.1),
                   slack=_f(d.get("slack"), 1.15, 1.0, 3.0),
                   corner_radius=_f(d.get("corner_radius"), 0.025, 0.0, 1.0),
                   allow45=bool(d.get("allow45", False)),
                   clearance=_f(d.get("clearance"), 0.01, 0.0, 1.0),
                   connectors=bool(d.get("connectors", True)),
                   clip_spacing=_f(d.get("clip_spacing"), 0.0, 0.0, 10.0),
                   bundle=int(_f(d.get("bundle"), 1, 1, 16)),
                   bundle_spacing=_f(d.get("bundle_spacing"), 0.0, 0.0, 0.5),
                   emission=Emission.from_dict(d.get("emission")),
                   points=pts, settled=bool(d.get("settled", False)),
                   locked=bool(d.get("locked", False)),
                   visible=bool(d.get("visible", True)))


# ---- polylines --------------------------------------------------------------
def polyline_length(P):
    P = np.asarray(P, np.float64)
    return float(np.hypot(*np.diff(P, axis=0).T).sum()) if len(P) > 1 else 0.0


def resample(P, step):
    """Uniform arc-length resampling (keeps both ends)."""
    P = np.asarray(P, np.float64)
    if len(P) < 2:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L <= 0:
        return P[:1].repeat(2, axis=0)
    n = max(2, int(math.ceil(L / max(step, 1e-6))) + 1)
    t = np.linspace(0.0, L, n)
    return np.stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])], axis=1)


def offset_polyline(P, off):
    """Shift a polyline sideways by `off` (positive = left of the direction)."""
    if off == 0:
        return P
    P = np.asarray(P, np.float64)
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return P + n * off


def gravity_vector(sim):
    a = math.radians(float(sim.get("gravity_angle", 90.0)))
    return math.cos(a), math.sin(a)     # image space: +y is down


def sag_depth(L, target):
    """Depth of a parabola over chord L with arc length `target` (bisection;
    mirrors static/js/wiresim.js)."""
    if target <= L * (1 + 1e-9):
        return 0.0
    t = np.linspace(0.0, 1.0, 65)

    def arc(d):
        return float(np.hypot(np.diff(L * t), np.diff(4 * d * t * (1 - t))).sum())
    lo, hi = 0.0, target
    for _ in range(50):
        m = (lo + hi) / 2
        lo, hi = (m, hi) if arc(m) < target else (lo, m)
    return (lo + hi) / 2


def sag_curve(a, b, slack, gravity_angle=90.0, n=48):
    """Parabolic sag of the right length: the starting shape of a sim wire."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    L = float(np.hypot(*(b - a)))
    t = np.linspace(0.0, 1.0, n)
    P = a[None] + (b - a)[None] * t[:, None]
    if L < 1e-9:
        return P
    depth = sag_depth(L, max(slack, 1.0) * L)
    g = np.array(gravity_vector({"gravity_angle": gravity_angle}))
    chord = (b - a) / L
    perp = g - chord * float(g @ chord)          # gravity across the chord
    if np.linalg.norm(perp) < 1e-6:              # hanging straight along gravity
        perp = np.array([-chord[1], chord[0]])
    perp /= np.linalg.norm(perp)
    return P + perp[None] * (4.0 * depth * t * (1 - t))[:, None]


def resolved_path(wire, doc):
    """Centerline for baking: stored points with their ends moved to the
    current endpoint positions (the change is blended along the wire)."""
    a = np.array(wire.a.resolve(doc))
    b = np.array(wire.b.resolve(doc))
    P = np.asarray(wire.points, np.float64)
    if len(P) < 2:
        if wire.mode == "sim":
            return sag_curve(a, b, wire.slack, getattr(doc, "wire_sim", DEFAULT_SIM)["gravity_angle"])
        return np.stack([a, b])
    da, db = a - P[0], b - P[-1]
    if np.abs(da).max() < 1e-9 and np.abs(db).max() < 1e-9:
        return P
    seg = np.hypot(*np.diff(P, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    w = (s / s[-1])[:, None] if s[-1] > 0 else np.linspace(0, 1, len(P))[:, None]
    return P + da[None] * (1 - w) + db[None] * w


def is_stale(wire, doc, tol=1e-5):
    if wire.mode != "sim" or len(wire.points) < 2:
        return wire.mode == "sim"
    a = np.array(wire.a.resolve(doc))
    b = np.array(wire.b.resolve(doc))
    return bool(np.abs(a - wire.points[0]).max() > tol or np.abs(b - wire.points[-1]).max() > tol
                or not wire.settled)


# ---- baking -------------------------------------------------------------------
def _sample(H, grid, x, y):
    """Bilinear height samples at world points (wraps when the grid tiles)."""
    fx = np.asarray(x) / grid.pmx - 0.5
    fy = np.asarray(y) / grid.pmy - 0.5
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx, ty = fx - x0, fy - y0
    if grid.tiling:
        xi = lambda i: i % grid.nx
        yi = lambda j: j % grid.ny
    else:
        xi = lambda i: np.clip(i, 0, grid.nx - 1)
        yi = lambda j: np.clip(j, 0, grid.ny - 1)
    a = H[yi(y0), xi(x0)] * (1 - tx) + H[yi(y0), xi(x0 + 1)] * tx
    b = H[yi(y0 + 1), xi(x0)] * (1 - tx) + H[yi(y0 + 1), xi(x0 + 1)] * tx
    return a * (1 - ty) + b * ty


def _bottom_heights(H, grid, P, hw, span):
    """Height the wire bottom rests on at each vertex: the highest point under
    its footprint, then a running max over `span` so it bridges narrow gaps."""
    d = np.gradient(P, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    z = _sample(H, grid, P[:, 0], P[:, 1])
    for f in (-1.0, -0.6, 0.6, 1.0):
        q = P + n * hw * f
        z = np.maximum(z, _sample(H, grid, q[:, 0], q[:, 1]))
    k = max(0, int(round(span)))
    if k:
        zp = np.pad(z, k, mode="edge")
        z = np.max(np.stack([zp[i:i + len(z)] for i in range(2 * k + 1)]), axis=0)
        zp = np.pad(z, k, mode="edge")
        z = np.mean(np.stack([zp[i:i + len(z)] for i in range(2 * k + 1)]), axis=0)
    return z


def _profile(wire, d, s):
    """Height above the wire bottom for distance d from the centerline and
    arc length s, plus coverage (0..1) of the strand."""
    r = wire.radius
    if wire.profile == "ribbon":
        hw, th = 2.5 * r, 0.8 * r
        e = np.clip((hw - d) / (0.45 * th), 0.0, 1.0)
        edge = np.sqrt(e * (2 - e))                                  # rounded edges
        ridges = 0.06 * r * (0.5 + 0.5 * np.cos(d * (2 * np.pi / (0.55 * r))))
        return (th - ridges) * edge, (d < hw).astype(np.float32)
    rr = r
    if wire.profile == "hose":
        rr = r * (0.88 + 0.12 * (0.5 + 0.5 * np.cos(s * (2 * np.pi / (1.1 * r)))))
    inside = d < rr
    core = rr + np.sqrt(np.clip(rr * rr - d * d, 0.0, None))
    u = np.clip((d - rr) / (0.35 * r), 0.0, 1.0)                     # soft contact fillet
    fil = rr * (1 - u * u * (3 - 2 * u)) * 0.6
    return np.where(inside, core, fil), inside.astype(np.float32)


CHUNK = 24


def _stamp_strand(H, E, M, ids, grid, wire, P, s_vert, zb, rad, wid):
    """Composite one strand polyline into the maps."""
    hw = wire.half_width() * 1.2 + wire.radius * 0.4
    for c0 in range(0, len(P) - 1, CHUNK):
        c1 = min(len(P) - 1, c0 + CHUNK)
        A, B = P[c0:c1], P[c0 + 1:c1 + 1]
        lo = np.minimum(A.min(0), B.min(0)) - hw
        hi = np.maximum(A.max(0), B.max(0)) + hw
        reg = grid.region(lo[0], lo[1], hi[0], hi[1])
        if reg is None:
            continue
        key, X, Y = reg
        AB = (B - A)
        L2 = np.maximum((AB ** 2).sum(1), 1e-18)
        px = X[None] - A[:, 0, None, None]
        py = Y[None] - A[:, 1, None, None]
        t = np.clip((px * AB[:, 0, None, None] + py * AB[:, 1, None, None]) / L2[:, None, None], 0, 1)
        dx = px - t * AB[:, 0, None, None]
        dy = py - t * AB[:, 1, None, None]
        dd = dx * dx + dy * dy
        k = np.argmin(dd, axis=0)
        d = np.sqrt(np.take_along_axis(dd, k[None], 0)[0])
        tk = np.take_along_axis(t, k[None], 0)[0]
        i = k + c0
        bottom = zb[i] + (zb[i + 1] - zb[i]) * tk
        s = s_vert[i] + (s_vert[i + 1] - s_vert[i]) * tk
        off, cov = _profile(wire, d, s)
        sub = H[key]
        top = bottom + off
        H[key] = np.where(off > 0, np.maximum(sub, top), sub)
        if M is not None:
            M[key] = np.maximum(M[key], cov)
        if ids is not None:
            ids[key] = np.where(cov > 0.5, wid, ids[key])
        if E is not None:
            sub = E[key] * (1 - cov)[..., None]
            if rad is not None:
                sub = sub + cov[..., None] * rad
            E[key] = sub


def _stamp_box(H, E, M, ids, grid, cx, cy, ang, hl, hw, z_top, bottom_z, wid, bevel):
    """A small rounded box (connector or clip) whose top is at z_top."""
    r = math.hypot(hl, hw) + bevel
    reg = grid.region(cx - r, cy - r, cx + r, cy + r)
    if reg is None:
        return
    key, X, Y = reg
    lx, ly = sdf.rotate(X - cx, Y - cy, math.degrees(ang))
    d = -sdf.sd_round_box(lx, ly, hl, hw, [min(hl, hw) * 0.35] * 4)
    h = shape_height(d, Bevel(width=bevel, profile="round"), z_top - bottom_z, grid.pm)
    cov = np.clip(d / grid.pm + 0.5, 0, 1).astype(np.float32)
    sub = H[key]
    H[key] = np.where(d > 0, np.maximum(sub, bottom_z + h), sub)
    if M is not None:
        M[key] = np.maximum(M[key], cov)
    if ids is not None:
        ids[key] = np.where(cov > 0.5, wid, ids[key])
    if E is not None:
        E[key] = E[key] * (1 - cov)[..., None]


def composite(H, ids, E, M, grid, doc, id_base):
    """Add every visible wire to the maps, in document order."""
    wires = [w for w in getattr(doc, "wires", []) if w.visible]
    if not wires:
        return
    from . import route
    route.route_all(doc)
    for wi, w in enumerate(wires):
        P = resolved_path(w, doc)
        if len(P) < 2 or polyline_length(P) < 1e-6:
            continue
        r = w.radius
        P = resample(P, max(min(r * 0.6, 0.004), grid.pm * 0.75))
        span_vert = 2.0 * r / max(polyline_length(P) / (len(P) - 1), 1e-9)
        n = max(1, w.bundle)
        sp = w.strand_spacing()
        bundle_hw = w.total_half_width()
        # The whole harness rests on the highest point under its footprint.
        zb = _bottom_heights(H, grid, P, bundle_hw, span_vert)
        rad = w.emission.radiance() if E is not None else None
        wid = id_base + wi + 1
        for k in range(n):
            Q = offset_polyline(P, (k - (n - 1) * 0.5) * sp)
            sv = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(Q, axis=0).T))])
            _stamp_strand(H, E, M, ids, grid, w, Q, sv, zb, rad, wid)
        thick = 2 * r if w.profile != "ribbon" else 0.8 * r
        if w.clip_spacing > 0:
            s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(P, axis=0).T))])
            L = s[-1]
            first = 6 * r if w.connectors else 2 * r
            for sc in np.arange(first + w.clip_spacing * 0.5, L - first, w.clip_spacing):
                j = int(np.searchsorted(s, sc))
                j = min(max(j, 1), len(P) - 1)
                d = P[j] - P[j - 1]
                ang = math.atan2(d[1], d[0])
                _stamp_box(H, E, M, ids, grid, P[j, 0], P[j, 1], ang, 0.55 * r + 0.002,
                           bundle_hw + 0.5 * r + 0.002, zb[j] + thick + 0.25 * r, zb[j], wid,
                           max(0.3 * r, grid.pm))
        if w.connectors:
            for end, nb in ((0, 1), (len(P) - 1, len(P) - 2)):
                d = P[end] - P[nb]
                ln = math.hypot(*d) or 1.0
                ang = math.atan2(d[1], d[0])
                hl = 1.9 * r + 0.002
                c = P[end] - d / ln * (hl * 0.7)
                _stamp_box(H, E, M, ids, grid, c[0], c[1], ang, hl, bundle_hw + 0.5 * r,
                           zb[end] + thick + 0.4 * r, zb[end], wid, max(0.35 * r, grid.pm))
