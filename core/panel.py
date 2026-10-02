"""Panel model.

A Panel is a parametric description, never pixels. Bevel width, depth, corner
sizes and detail sizes are stored in meters, so resizing a panel changes only
its rectangle and every edge treatment keeps its world size. The height field
is regenerated from these parameters on every bake.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np

from . import sdf, profiles
from .filters import srgb_to_linear
from .units import m_to_px, MIN_BEVEL_PX, EMISSION_UNIT

CORNER_STYLES = ("chamfer", "round")
PANEL_MODES = ("raise", "inset", "max")
DETAIL_MODES = ("raise", "inset")
DETAIL_SHAPES = ("circle", "hex", "rect")
ANCHORS = ("tl", "t", "tr", "l", "c", "r", "bl", "b", "br",
           "corners", "left-right", "top-bottom")


def new_id(prefix="p"):
    return prefix + uuid.uuid4().hex[:8]


def _f(v, default, lo=None, hi=None):
    try:
        x = float(v)
    except (TypeError, ValueError):
        x = float(default)
    if not np.isfinite(x):
        x = float(default)
    if lo is not None:
        x = max(lo, x)
    if hi is not None:
        x = min(hi, x)
    return x


def _choice(v, options, default):
    return v if v in options else default


@dataclass
class Bevel:
    width: float = 0.008          # meters
    profile: str = "linear"
    points: List[List[float]] = field(default_factory=lambda: [[0, 0], [0.5, 0.7], [1, 1]])

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Bevel":
        d = d or {}
        pts = d.get("points") or [[0, 0], [0.5, 0.7], [1, 1]]
        try:
            pts = [[_f(p[0], 0, 0, 1), _f(p[1], 0, 0, 1)] for p in pts][:16]
        except (TypeError, IndexError):
            pts = [[0, 0], [1, 1]]
        return cls(width=_f(d.get("width"), 0.008, 0.0, 10.0),
                   profile=_choice(d.get("profile"), profiles.PROFILES, "linear"),
                   points=pts)


@dataclass
class Emission:
    """Light output. `color` is a display (sRGB) color, `strength` is relative:
    1 is about as bright as a white surface under the default spotlight."""
    enabled: bool = False
    color: List[float] = field(default_factory=lambda: [0.35, 0.85, 1.0])
    strength: float = 4.0
    diffuser: float = 0.0         # panel faces only: 1 = edges fade, center brightest

    def radiance(self):
        """Linear RGB radiance, or None when off."""
        if not self.enabled or self.strength <= 0:
            return None
        return (srgb_to_linear(np.clip(self.color, 0, 1)) * self.strength * EMISSION_UNIT).astype(np.float32)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Emission":
        d = d or {}
        col = d.get("color", [0.35, 0.85, 1.0])
        if not isinstance(col, (list, tuple)) or len(col) != 3:
            col = [0.35, 0.85, 1.0]
        return cls(enabled=bool(d.get("enabled", False)),
                   color=[_f(c, 1.0, 0, 1) for c in col],
                   strength=_f(d.get("strength"), 4.0, 0, 1000),
                   diffuser=_f(d.get("diffuser"), 0.0, 0, 1))


@dataclass
class Groove:
    """A channel cut into the panel top, parallel to its outline."""
    offset: float = 0.02          # distance in from the panel edge (m)
    width: float = 0.0            # 0 disables
    depth: float = 0.002
    emission: Emission = field(default_factory=Emission)

    @property
    def enabled(self):
        return self.width > 0 and self.depth > 0

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Groove":
        d = d or {}
        return cls(offset=_f(d.get("offset"), 0.02, 0, 10),
                   width=_f(d.get("width"), 0.0, 0, 10),
                   depth=_f(d.get("depth"), 0.002, 0, 1),
                   emission=Emission.from_dict(d.get("emission")))


@dataclass
class Detail:
    """A small shape attached to a panel by an anchor plus an inward offset."""
    kind: str = "bolt"            # UI label only
    shape: str = "circle"
    anchor: str = "corners"
    ox: float = 0.025             # inward offset from the anchor (m)
    oy: float = 0.025
    w: float = 0.016
    h: float = 0.016
    radius: float = 0.0           # rect corner radius
    rotation: float = 0.0         # degrees
    count: int = 1                # array along the local y axis
    spacing: float = 0.0
    depth: float = 0.003
    mode: str = "raise"
    bevel: Bevel = field(default_factory=lambda: Bevel(width=0.006, profile="round"))
    emission: Emission = field(default_factory=Emission)

    def centers(self, x0, y0, x1, y1):
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        a = self.anchor
        if a == "corners":
            base = [(x0 + self.ox, y0 + self.oy), (x1 - self.ox, y0 + self.oy),
                    (x1 - self.ox, y1 - self.oy), (x0 + self.ox, y1 - self.oy)]
        elif a == "left-right":
            base = [(x0 + self.ox, cy + self.oy), (x1 - self.ox, cy + self.oy)]
        elif a == "top-bottom":
            base = [(cx + self.ox, y0 + self.oy), (cx + self.ox, y1 - self.oy)]
        else:
            if a.endswith("l"):
                px = x0 + self.ox
            elif a.endswith("r"):
                px = x1 - self.ox
            else:
                px = cx + self.ox
            if a.startswith("t"):
                py = y0 + self.oy
            elif a.startswith("b"):
                py = y1 - self.oy
            else:
                py = cy + self.oy
            base = [(px, py)]
        n = max(1, int(self.count))
        if n == 1:
            return base
        rad = np.deg2rad(self.rotation)
        dx, dy = -np.sin(rad), np.cos(rad)
        out = []
        for bx, by in base:
            for k in range(n):
                s = (k - (n - 1) * 0.5) * self.spacing
                out.append((bx + dx * s, by + dy * s))
        return out

    def local_sdf(self, lx, ly):
        lx, ly = sdf.rotate(lx, ly, self.rotation)
        if self.shape == "circle":
            return sdf.sd_circle(lx, ly, self.w * 0.5)
        if self.shape == "hex":
            return sdf.sd_hexagon(lx, ly, self.w * 0.5)
        r = min(self.radius, self.w * 0.5, self.h * 0.5)
        return sdf.sd_round_box(lx, ly, self.w * 0.5, self.h * 0.5, [r] * 4)

    def bound_radius(self):
        return 0.5 * float(np.hypot(self.w, self.h)) + 1e-6

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Detail":
        d = d or {}
        return cls(kind=str(d.get("kind", "bolt"))[:32],
                   shape=_choice(d.get("shape"), DETAIL_SHAPES, "circle"),
                   anchor=_choice(d.get("anchor"), ANCHORS, "corners"),
                   ox=_f(d.get("ox"), 0.025, -10, 10), oy=_f(d.get("oy"), 0.025, -10, 10),
                   w=_f(d.get("w"), 0.016, 0.0005, 10), h=_f(d.get("h"), 0.016, 0.0005, 10),
                   radius=_f(d.get("radius"), 0.0, 0, 10),
                   rotation=_f(d.get("rotation"), 0.0, -360, 360),
                   count=int(_f(d.get("count"), 1, 1, 64)),
                   spacing=_f(d.get("spacing"), 0.0, 0, 10),
                   depth=_f(d.get("depth"), 0.003, 0, 1),
                   mode=_choice(d.get("mode"), DETAIL_MODES, "raise"),
                   bevel=Bevel.from_dict(d.get("bevel") or {"width": 0.006, "profile": "round"}),
                   emission=Emission.from_dict(d.get("emission")))


@dataclass
class Panel:
    id: str = field(default_factory=new_id)
    name: str = "Panel"
    x: float = 0.0                # top-left corner, meters
    y: float = 0.0
    w: float = 0.5
    h: float = 0.5
    corner_style: str = "chamfer"
    corners: List[float] = field(default_factory=lambda: [0.02, 0.02, 0.02, 0.02])
    mode: str = "raise"
    depth: float = 0.012          # height of the flat top above the base (m)
    bevel: Bevel = field(default_factory=Bevel)
    groove: Groove = field(default_factory=Groove)
    details: List[Detail] = field(default_factory=list)
    emission: Emission = field(default_factory=Emission)   # glowing face (light panel)
    locked: bool = False
    visible: bool = True

    # ---- geometry -------------------------------------------------------
    @property
    def rect(self):
        return self.x, self.y, self.x + self.w, self.y + self.h

    def resize(self, w, h, anchor="tl"):
        """Change dimensions only. Bevel, corners and details keep their size.

        `anchor` names the point that stays fixed: tl, t, tr, l, c, r, bl, b, br.
        """
        w, h = max(w, 0.001), max(h, 0.001)
        if anchor.endswith("r"):
            self.x += self.w - w
        elif not anchor.endswith("l"):
            self.x += (self.w - w) * 0.5
        if anchor.startswith("b"):
            self.y += self.h - h
        elif not anchor.startswith("t"):
            self.y += (self.h - h) * 0.5
        self.w, self.h = w, h

    def sdf(self, X, Y):
        hx, hy = self.w * 0.5, self.h * 0.5
        px, py = X - (self.x + hx), Y - (self.y + hy)
        if self.corner_style == "round":
            return sdf.sd_round_box(px, py, hx, hy, self.corners)
        return sdf.sd_chamfer_box(px, py, hx, hy, self.corners)

    def has_emission(self):
        return (self.emission.enabled or (self.groove.enabled and self.groove.emission.enabled)
                or any(d.emission.enabled for d in self.details))

    # ---- diagnostics ----------------------------------------------------
    def warnings(self, density):
        out = []

        def check(label, width):
            px = m_to_px(width, density)
            if 0 < px < MIN_BEVEL_PX:
                out.append(f"{label} is {px:.1f} px at {density:g} px/m and will alias "
                           f"(keep it above {MIN_BEVEL_PX:g} px).")
        check("Bevel", self.bevel.width)
        if self.bevel.width * 2 > min(self.w, self.h):
            out.append("Bevel is wider than half the panel, so the top will not be flat.")
        if self.groove.enabled:
            check("Groove", self.groove.width)
        for i, d in enumerate(self.details):
            check(f"Detail {i + 1} ({d.kind}) bevel", d.bevel.width)
        return out

    # ---- serialization -------------------------------------------------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Panel":
        corners = d.get("corners", [0.02] * 4)
        if not isinstance(corners, (list, tuple)) or len(corners) != 4:
            corners = [0.02] * 4
        return cls(id=str(d.get("id") or new_id())[:40],
                   name=str(d.get("name", "Panel"))[:64],
                   x=_f(d.get("x"), 0, -100, 100), y=_f(d.get("y"), 0, -100, 100),
                   w=_f(d.get("w"), 0.5, 0.001, 100), h=_f(d.get("h"), 0.5, 0.001, 100),
                   corner_style=_choice(d.get("corner_style"), CORNER_STYLES, "chamfer"),
                   corners=[_f(c, 0.02, 0, 100) for c in corners],
                   mode=_choice(d.get("mode"), PANEL_MODES, "raise"),
                   depth=_f(d.get("depth"), 0.012, 0, 10),
                   bevel=Bevel.from_dict(d.get("bevel")),
                   groove=Groove.from_dict(d.get("groove")),
                   details=[Detail.from_dict(x) for x in (d.get("details") or [])][:64],
                   emission=Emission.from_dict(d.get("emission")),
                   locked=bool(d.get("locked", False)),
                   visible=bool(d.get("visible", True)))


def shape_height(d, bevel: Bevel, depth, px_m):
    """Height of a beveled shape given inside-distance d (meters, >0 inside).

    The bevel always spans `bevel.width` meters from the outline. A bevel
    under half a pixel collapses to a one-pixel ramp for anti-aliasing.
    """
    width = bevel.width if bevel.width >= 0.5 * px_m else px_m
    t = np.clip(d / width, 0.0, 1.0)
    return depth * profiles.evaluate(bevel.profile, t, bevel.points)


def groove_cut(d, groove: Groove, px_m):
    """Depth removed by an inner groove, with soft walls (positive values)."""
    w = max(groove.width, px_m)
    u = (d - groove.offset) / w
    wall = 0.3
    a = np.clip(u / wall, 0, 1)
    b = np.clip((1 - u) / wall, 0, 1)
    s = (a * a * (3 - 2 * a)) * (b * b * (3 - 2 * b))
    return groove.depth * s
