"""Liquid leaks: a shallow-liquid simulation over the panel height field.

Each leak is a source (attached to a panel like a wire end, so it follows the
panel) pouring one liquid. Liquids flow with the "virtual pipes" model: every
cell exchanges liquid with its four neighbors through a flux driven by the
difference in surface level (terrain + liquid depth) plus the drop in the
gravity direction, the panel being a wall. Raised edges hold liquid back until
it pools high enough to spill over or find a way around; seams and grooves
channel it.

Per liquid the run records:
    wet      liquid still present at the end (a leak that is still leaking)
    stain    where the liquid has been (darkened, for oil and coolant)
    residue  what is left as liquid evaporates, heaviest at the drying
             edges, which is what makes tide lines and mineral rings
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np

from .filters import gauss_blur
from .panel import _f, _choice
from .wire import End, END_ANCHORS

FLUID_PX_PER_M = 100.0         # simulation resolution cap (pixels per meter): 1 cm cells
MAX_LEAKS = 64

# tilt: how strongly the gravity direction drives the flow (viscous liquids
# creep); damp: flux kept from the previous step (inertia); evap: depth lost
# per step (m); rate: depth poured per step at amount 1 (m).
LIQUIDS = {
    "water":   {"label": "Water", "color": [0.55, 0.55, 0.50], "tilt": 1.0, "damp": 0.985,
                "evap": 2.5e-7, "rate": 1.2e-4},
    "oil":     {"label": "Oil", "color": [0.025, 0.02, 0.012], "tilt": 0.5, "damp": 0.9,
                "evap": 1.0e-8, "rate": 1.4e-4},
    "coolant": {"label": "Coolant", "color": [0.10, 0.55, 0.32], "tilt": 0.8, "damp": 0.96,
                "evap": 1.2e-7, "rate": 1.0e-4},
}
AUTO_MIX = (("water", 0.5), ("oil", 0.3), ("coolant", 0.2))


def new_leak_id():
    import uuid
    return "k" + uuid.uuid4().hex[:8]


@dataclass
class Leak:
    id: str = ""
    liquid: str = "water"
    amount: float = 0.6                    # 0..1: how much has leaked
    color: tuple = (0.10, 0.55, 0.32)      # coolant tint
    panel: str = None                      # attachment, as for a wire end
    anchor: str = "c"
    dx: float = 0.0
    dy: float = 0.0
    x: float = 0.0
    y: float = 0.0

    def end(self):
        return End(panel=self.panel, anchor=self.anchor, dx=self.dx, dy=self.dy, x=self.x, y=self.y)

    def resolve(self, doc):
        return self.end().resolve(doc)

    def to_dict(self):
        d = asdict(self)
        d["color"] = list(self.color)
        return d

    @classmethod
    def from_dict(cls, d) -> "Leak":
        d = d if isinstance(d, dict) else {}
        e = End.from_dict(d)
        col = d.get("color")
        if not (isinstance(col, (list, tuple)) and len(col) == 3):
            col = LIQUIDS["coolant"]["color"]
        return cls(id=str(d.get("id") or new_leak_id())[:40],
                   liquid=_choice(d.get("liquid"), tuple(LIQUIDS), "water"),
                   amount=_f(d.get("amount"), 0.6, 0.0, 1.0),
                   color=tuple(_f(c, 0.5, 0.0, 1.0) for c in col),
                   panel=e.panel, anchor=_choice(e.anchor, END_ANCHORS, "c"),
                   dx=e.dx, dy=e.dy, x=e.x, y=e.y)


def leaks_from(lst):
    return [Leak.from_dict(k) for k in (lst or [])[:MAX_LEAKS] if isinstance(k, dict)]


# ---- sources -------------------------------------------------------------------------
def auto_sites(doc, count, seed):
    """Leak sources picked from bolts, vents and wire connectors (seeded)."""
    if count <= 0:
        return []
    from .wire import resolved_path
    sites = []
    for p in doc.panels:
        if not p.visible:
            continue
        for det in p.details:
            if det.kind in ("bolt", "vent") or det.shape in ("circle", "hex"):
                for cx, cy in det.centers(*p.rect):
                    sites.append((cx, cy + 0.6 * det.h))
    for w in doc.wires:
        if w.visible:
            P = resolved_path(w, doc)
            if len(P) >= 2:
                sites += [tuple(P[0]), tuple(P[-1])]
    if not sites:
        return []
    rng = np.random.default_rng(seed + 7919)
    pick = rng.choice(len(sites), size=min(count, len(sites)), replace=False)
    names = [n for n, _ in AUTO_MIX]
    wts = np.array([w for _, w in AUTO_MIX])
    out = []
    for i in pick:
        liquid = names[int(rng.choice(len(names), p=wts / wts.sum()))]
        out.append({"x": float(sites[i][0]), "y": float(sites[i][1]), "liquid": liquid,
                    "amount": float(rng.uniform(0.35, 0.9)), "color": LIQUIDS["coolant"]["color"]})
    return out


def sources(doc, auto_count, seed):
    out = []
    for k in getattr(doc, "leaks", []):
        if k.amount <= 0:
            continue
        x, y = k.resolve(doc)
        out.append({"x": x, "y": y, "liquid": k.liquid, "amount": k.amount, "color": list(k.color)})
    return out + auto_sites(doc, auto_count, seed)


# ---- the simulation --------------------------------------------------------------------
def _shift_in(a, di, dj, wrap):
    """Value of the neighbor at (+dj rows, +di cols); zero beyond the edge unless wrapping."""
    out = np.roll(a, (-dj, -di), axis=(0, 1))
    if not wrap:
        if di == 1:
            out[:, -1] = 0
        elif di == -1:
            out[:, 0] = 0
        if dj == 1:
            out[-1, :] = 0
        elif dj == -1:
            out[0, :] = 0
    return out


def run(H, pm, wrap, gvec, srcs, liquid, steps):
    """Pipe-model flow of one liquid from `srcs` [(col, row, weight)] on grid H (m).
    Returns (wet, stain, residue) fields in about [0, 1]."""
    L = LIQUIDS[liquid]
    ny, nx = H.shape
    H = H.astype(np.float32)
    d = np.zeros((ny, nx), np.float32)
    # Directions: right, left, down, up as (di, dj); terrain step and gravity drop per pipe.
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    drop = []
    for di, dj in dirs:
        Hn = _shift_in(H, di, dj, True)
        g = L["tilt"] * (gvec[0] * di + gvec[1] * dj) * pm
        dz = H - Hn + np.float32(g)
        if not wrap:                      # liquid leaves freely over the canvas edge
            if di == 1:
                dz[:, -1] = g + 0.002
            elif di == -1:
                dz[:, 0] = g + 0.002
            if dj == 1:
                dz[-1, :] = g + 0.002
            elif dj == -1:
                dz[0, :] = g + 0.002
        drop.append(dz.astype(np.float32))
    F = [np.zeros((ny, nx), np.float32) for _ in dirs]
    src = np.zeros((ny, nx), np.float32)
    for c, r, w in srcs:
        ci, ri = int(c) % nx, int(r) % ny
        for oj in (-1, 0, 1):
            for oi in (-1, 0, 1):
                jj, ii = ri + oj, ci + oi
                if wrap:
                    jj, ii = jj % ny, ii % nx
                if 0 <= jj < ny and 0 <= ii < nx:
                    src[jj, ii] += w * (0.25 if oi or oj else 1.0) / 3.0
    src *= np.float32(L["rate"])
    c = np.float32(0.2)
    damp = np.float32(L["damp"])
    evap = np.float32(L["evap"])
    stain = np.zeros((ny, nx), np.float32)
    residue = np.zeros((ny, nx), np.float32)
    flow = np.zeros((ny, nx), np.float32)       # time-integrated depth: how much passed
    wet_ref = np.float32(4e-5)          # a film this deep reads as fully wet
    stain_ref = np.float32(1.5e-5)      # any thin film that passed leaves a stain
    pour = steps                         # still leaking: the trail ends wet
    for step in range(steps):
        if step < pour:
            d += src
        out = np.zeros_like(d)
        for k, (di, dj) in enumerate(dirs):
            dn = _shift_in(d, di, dj, wrap)
            f = damp * F[k] + c * (drop[k] + d - dn)
            np.maximum(f, 0, out=f)
            F[k] = f
            out += f
        scale = np.minimum(1.0, d / np.maximum(out, 1e-12)).astype(np.float32)
        inflow = np.zeros_like(d)
        for k, (di, dj) in enumerate(dirs):
            F[k] *= scale
            # Liquid sent right by the left neighbor arrives here, and so on.
            inflow += _shift_in(F[k], -di, -dj, wrap)
        d = d - F[0] - F[1] - F[2] - F[3] + inflow
        lost = np.minimum(d, evap)
        d -= lost
        np.maximum(d, 0, out=d)
        # Residue: evaporation leaves solids behind, most where the film is thin (its edge).
        residue += lost * (1.0 + 3.0 * (d < 0.3 * wet_ref))
        np.maximum(stain, np.minimum(d / stain_ref, 1.0), out=stain)
        flow += d
    wet = np.clip(d / wet_ref, 0, 1)
    # Heavier where more liquid passed (near the source, where it pooled), fading
    # along the trail; every touched cell keeps a faint mark.
    if (flow > 0).any():
        ref = float(np.percentile(flow[stain > 0.5], 85)) if (stain > 0.5).any() else float(flow.max())
        stain = np.maximum(0.3 * stain, np.sqrt(np.clip(flow / max(ref, 1e-12), 0, 1)) * (stain > 0.05))
    res = residue / max(float(np.percentile(residue[residue > 0], 98)) if (residue > 0).any() else 1.0, 1e-12)
    return wet, stain, np.clip(res, 0, 1)


def simulate(H, W, Hc, wrap, gvec, srcs, seed=1):
    """Run every liquid over height field H (any resolution; resampled to the
    fluid grid). Returns per-liquid fields on the fluid grid and its size."""
    ny0, nx0 = H.shape
    ppm = min(nx0 / W, FLUID_PX_PER_M)
    nx, ny = max(16, int(round(W * ppm))), max(16, int(round(Hc * ppm)))
    fy, fx = ny0 / ny, nx0 / nx
    if fx > 1.01 or fy > 1.01:
        # Box-average down to the fluid grid (keeps seams as shallow channels).
        yi = (np.arange(ny) * fy).astype(int)
        xi = (np.arange(nx) * fx).astype(int)
        Hb = gauss_blur(H, 0.5 * max(fx, fy), wrap)
        Hf = Hb[yi][:, xi]
    else:
        Hf = H
    pm = 0.5 * (W / nx + Hc / ny)
    # Micro-roughness (a fraction of a millimeter) makes the flow wander and branch.
    from .weather import fbm
    X = ((np.arange(nx) + 0.5) * (W / nx))[None, :]
    Y = ((np.arange(ny) + 0.5) * (Hc / ny))[:, None]
    per = (W, Hc) if wrap else None
    Hf = Hf + 0.012 * fbm(X, Y, 0.04, seed + 41, 3, pm, per)
    out = {}
    for liquid in LIQUIDS:
        S = [(s["x"] / W * nx, s["y"] / Hc * ny, s["amount"]) for s in srcs if s["liquid"] == liquid]
        if not wrap:
            S = [s for s in S if 0 <= s[0] < nx and 0 <= s[1] < ny]
        if not S:
            continue
        amount = max(s[2] for s in S)
        steps = int((0.8 + 2.6 * amount) * max(nx, ny))
        out[liquid] = run(Hf, pm, wrap, gvec, S, liquid, steps)
    # Coolant tint: weighted by the sources' colors.
    cool = [s for s in srcs if s["liquid"] == "coolant"]
    tint = np.mean([s["color"] for s in cool], axis=0) if cool else np.array(LIQUIDS["coolant"]["color"])
    return {"fields": out, "grid": (W, Hc, nx, ny, wrap), "coolant_color": np.asarray(tint, float)}
