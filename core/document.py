"""Document: canvas, texel density, output settings and the panel stack."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .panel import Panel, _f
from .wire import Wire, DEFAULT_SIM
from .units import TEXEL_DENSITY_DEFAULT, MAX_RESOLUTION

NORMAL_CONVENTIONS = ("gl", "dx")


@dataclass
class Document:
    canvas_w: float = 2.0                 # meters
    canvas_h: float = 2.0
    texel_density: float = TEXEL_DENSITY_DEFAULT   # px per meter
    tiling: bool = False                  # wrap distance fields across edges
    normal_strength: float = 1.0
    normal_convention: str = "gl"         # gl = Y+ (OpenGL), dx = Y- (DirectX)
    generator: str = "panels"
    panels: List[Panel] = field(default_factory=list)
    layout: dict = field(default_factory=dict)   # last auto-layout params (UI state)
    wires: List[Wire] = field(default_factory=list)
    wire_sim: dict = field(default_factory=lambda: dict(DEFAULT_SIM))

    def resolution(self):
        w = int(round(self.canvas_w * self.texel_density))
        h = int(round(self.canvas_h * self.texel_density))
        return max(8, min(MAX_RESOLUTION, w)), max(8, min(MAX_RESOLUTION, h))

    def find(self, pid):
        for p in self.panels:
            if p.id == pid:
                return p
        return None

    def find_wire(self, wid):
        for w in self.wires:
            if w.id == wid:
                return w
        return None

    def warnings(self):
        out = {}
        for p in self.panels:
            w = p.warnings(self.texel_density)
            if w:
                out[p.id] = w
        return out

    def to_dict(self):
        return {
            "version": 1,
            "canvas_w": self.canvas_w, "canvas_h": self.canvas_h,
            "texel_density": self.texel_density, "tiling": self.tiling,
            "normal_strength": self.normal_strength,
            "normal_convention": self.normal_convention,
            "generator": self.generator,
            "panels": [p.to_dict() for p in self.panels],
            "layout": self.layout,
            "wires": [w.to_dict() for w in self.wires],
            "wire_sim": self.wire_sim,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Document":
        d = d or {}
        density = _f(d.get("texel_density"), TEXEL_DENSITY_DEFAULT, 16, 8192)
        max_m = MAX_RESOLUTION / density
        conv = d.get("normal_convention", "gl")
        return cls(
            canvas_w=_f(d.get("canvas_w"), 2.0, 0.05, max_m),
            canvas_h=_f(d.get("canvas_h"), 2.0, 0.05, max_m),
            texel_density=density,
            tiling=bool(d.get("tiling", False)),
            normal_strength=_f(d.get("normal_strength"), 1.0, 0.0, 20.0),
            normal_convention=conv if conv in NORMAL_CONVENTIONS else "gl",
            generator=str(d.get("generator", "panels")),
            panels=[Panel.from_dict(p) for p in (d.get("panels") or [])][:2000],
            layout=d.get("layout") if isinstance(d.get("layout"), dict) else {},
            wires=[Wire.from_dict(w) for w in (d.get("wires") or [])][:500],
            wire_sim=_sim_settings(d.get("wire_sim")),
        )


def _sim_settings(d):
    d = d if isinstance(d, dict) else {}
    return {
        "gravity_angle": _f(d.get("gravity_angle"), 90.0, -360, 360) % 360.0,
        "gravity": _f(d.get("gravity"), 1.0, 0.0, 10.0),
        "collide": bool(d.get("collide", True)),
        "auto_resettle": bool(d.get("auto_resettle", False)),
    }
