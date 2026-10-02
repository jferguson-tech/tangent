"""Surface materials and weathering.

Two layers, both recomputed from document settings on every bake:

* Materials: a bare-metal base, an optional paint layer over it, the wire
  material, and per-panel variation. These make the base color, roughness and
  metallic maps even with weathering off.
* Weathering, driven by Age (0 = new, 1 = abandoned):
    - edge wear on convex edges: chips through paint, or polishes bare metal
    - dirt in crevices, seams and where water pools
    - water streaks: droplets run across the height field in the gravity
      direction (the wire simulation's), slide along raised edges, pool behind
      them and leave grime where they travel
    - rust: grows from moisture, crevices and exposed metal; blisters paint at
      its rim, pits the metal, and its stain is carried downhill by water
    - heat and soot: vents and bright lights temper nearby bare metal (straw
      to blue), scorch paint, and leave soot rising against gravity
    - wire wear: hoses rub paint off the edges they rest on and get scuffed
      there, rubber fades and cracks, and rain running along a sagging wire
      drips from its low points, streaking the panels below
  plus liquid leaks (core/fluids.py) and hand-painted strokes (core/brush.py).

Large-scale fields come from simulations on grids fixed in world units,
cached per document in stages keyed only on what each depends on (a brush
stroke does not re-run the water simulation). Every bake of a document
(preview, zoom detail, export at any size) resamples the same fields. Fine
detail (chip edges, pitting, blisters, cracks) comes from noise anchored to
world coordinates and is evaluated at each bake's own resolution.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict

import numpy as np

from .filters import gauss_blur
from . import fluids, brush

SIM_PX_PER_M = 512.0           # weathering simulation resolution cap (pixels per meter)
PLUME_PX_PER_M = 128.0         # heat and soot plumes are smooth: a coarser grid
PAINT_THICKNESS = 0.00012      # m: chips through paint read as a small step
DIRT_COLOR = np.array([0.075, 0.062, 0.05])
STREAK_COLOR = np.array([0.11, 0.10, 0.085])
RUST_DARK = np.array([0.16, 0.055, 0.02])
RUST_LIGHT = np.array([0.50, 0.21, 0.06])
STAIN_COLOR = np.array([0.36, 0.15, 0.05])
SOOT_COLOR = np.array([0.018, 0.017, 0.016])
SCORCH_COLOR = np.array([0.13, 0.075, 0.03])
CHALK_COLOR = np.array([0.20, 0.20, 0.19])
MINERAL_COLOR = np.array([0.60, 0.60, 0.56])
# Temper colors of heated steel, coolest to hottest (linear RGB).
TEMPER = np.array([[0.62, 0.50, 0.30], [0.70, 0.46, 0.20], [0.50, 0.26, 0.13],
                   [0.38, 0.15, 0.32], [0.14, 0.21, 0.50], [0.30, 0.34, 0.40]])

DEFAULT_MATERIALS = {
    "metal": {"color": [0.42, 0.44, 0.47], "roughness": 0.45},      # bare metal (metallic)
    "paint": {"enabled": False, "color": [0.30, 0.32, 0.34], "roughness": 0.5, "metallic": 0.0},
    "wire": {"color": [0.035, 0.035, 0.04], "roughness": 0.5, "metallic": 0.0},
    "variation": 0.15,                                             # per-panel color/roughness spread
}

SLIDERS = ("age", "edge_wear", "dirt", "streaks", "rust", "heat", "wire_wear")
DEFAULT_WEATHERING = {
    "enabled": False, "preset": "clean", "age": 0.0, "seed": 1,
    "edge_wear": 0.5, "dirt": 0.5, "streaks": 0.5, "rust": 0.5,
    "heat": 0.5, "wire_wear": 0.5, "auto_leaks": 0,
}
MAX_AUTO_LEAKS = 16

PRESETS = {
    "clean": {"label": "Clean", "age": 0.0, "edge_wear": 0.5, "dirt": 0.5, "streaks": 0.5, "rust": 0.5,
              "heat": 0.5, "wire_wear": 0.5, "auto_leaks": 0},
    "field": {"label": "Field use", "age": 0.45, "edge_wear": 0.6, "dirt": 0.5, "streaks": 0.45, "rust": 0.25,
              "heat": 0.4, "wire_wear": 0.5, "auto_leaks": 0},
    "abandoned": {"label": "Abandoned", "age": 0.9, "edge_wear": 0.7, "dirt": 0.7, "streaks": 0.7, "rust": 0.75,
                  "heat": 0.3, "wire_wear": 0.8, "auto_leaks": 2},
    "leaking": {"label": "Leaking reactor", "age": 0.7, "edge_wear": 0.55, "dirt": 0.6, "streaks": 0.8,
                "rust": 0.6, "heat": 0.9, "wire_wear": 0.6, "auto_leaks": 6},
}

MASKS = ("wear", "dirt", "streaks", "rust", "wet", "fluid", "residue", "soot", "heat")


# ---- settings ----------------------------------------------------------------------
def _num(v, d, lo, hi):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return d
    return d if not math.isfinite(x) else min(hi, max(lo, x))


def _color(v, d):
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return [_num(c, dc, 0.0, 1.0) for c, dc in zip(v, d)]
    return list(d)


def materials_from(d):
    d = d if isinstance(d, dict) else {}
    D = DEFAULT_MATERIALS
    m, p, w = (d.get(k) if isinstance(d.get(k), dict) else {} for k in ("metal", "paint", "wire"))
    return {
        "metal": {"color": _color(m.get("color"), D["metal"]["color"]),
                  "roughness": _num(m.get("roughness"), D["metal"]["roughness"], 0.02, 1.0)},
        "paint": {"enabled": bool(p.get("enabled", D["paint"]["enabled"])),
                  "color": _color(p.get("color"), D["paint"]["color"]),
                  "roughness": _num(p.get("roughness"), D["paint"]["roughness"], 0.02, 1.0),
                  "metallic": _num(p.get("metallic"), D["paint"]["metallic"], 0.0, 1.0)},
        "wire": {"color": _color(w.get("color"), D["wire"]["color"]),
                 "roughness": _num(w.get("roughness"), D["wire"]["roughness"], 0.02, 1.0),
                 "metallic": _num(w.get("metallic"), D["wire"]["metallic"], 0.0, 1.0)},
        "variation": _num(d.get("variation"), D["variation"], 0.0, 1.0),
    }


def weathering_from(d):
    d = d if isinstance(d, dict) else {}
    D = DEFAULT_WEATHERING
    out = {k: _num(d.get(k), D[k], 0.0, 1.0) for k in SLIDERS}
    out["enabled"] = bool(d.get("enabled", D["enabled"]))
    out["preset"] = d.get("preset") if d.get("preset") in PRESETS or d.get("preset") == "custom" else D["preset"]
    out["seed"] = int(_num(d.get("seed"), D["seed"], 0, 2 ** 31 - 1))
    out["auto_leaks"] = int(_num(d.get("auto_leaks"), D["auto_leaks"], 0, MAX_AUTO_LEAKS))
    return out


def active(doc):
    """Weathering does something: aged, or leaks or brush strokes to apply."""
    w = doc.weathering
    return w["enabled"] and (w["age"] > 0 or w["auto_leaks"] > 0 or bool(getattr(doc, "leaks", None))
                             or bool(getattr(doc, "strokes", None)))


# ---- world-anchored noise ------------------------------------------------------------
def _hash(ix, iy, seed):
    """Deterministic pseudo-random values in [0, 1) for integer lattice points."""
    h = (ix.astype(np.uint64) * np.uint64(0x9E3779B1) + iy.astype(np.uint64) * np.uint64(0x85EBCA77)
         + np.uint64(seed) * np.uint64(0xC2B2AE3D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    h = (h * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(12)
    h = (h * np.uint64(0x297A2D39)) & np.uint64(0xFFFFFFFF)
    h ^= h >> np.uint64(15)
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0x1000000)


def value_noise(X, Y, cell, seed, period=None):
    """Smooth value noise in [0, 1] at world points (meters): X is a row of
    x coordinates (1, w), Y a column of y coordinates (h, 1). With `period`
    (canvas width, height) the noise tiles seamlessly across the canvas.

    The lattice is hashed once and interpolated separably (rows, then
    columns), which is far cheaper than hashing four corners per pixel."""
    fx, fy = np.ravel(X) / cell, np.ravel(Y) / cell
    x0, y0 = np.floor(fx), np.floor(fy)
    tx, ty = fx - x0, fy - y0
    tx = tx * tx * (3 - 2 * tx)
    ty = ty * ty * (3 - 2 * ty)
    x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
    # Lattice columns and rows this region touches.
    lx = np.arange(x0.min(), x0.max() + 2)
    ly = np.arange(y0.min(), y0.max() + 2)
    if period:
        lx = np.mod(lx, max(1, int(round(period[0] / cell))))
        ly = np.mod(ly, max(1, int(round(period[1] / cell))))
    # Offset keeps lattice indices non-negative for the unsigned hash.
    off = 1 << 20
    T = _hash(lx[None, :] + off, ly[:, None] + off, seed)
    ix = x0 - x0.min()
    iy = y0 - y0.min()
    rows = T[:, ix] * (1 - tx) + T[:, ix + 1] * tx                  # (lattice rows, w)
    return rows[iy] * (1 - ty)[:, None] + rows[iy + 1] * ty[:, None]


def fbm(X, Y, cell, seed, octaves=4, pixel=None, period=None):
    """Fractal noise in about [-0.5, 0.5]. Octaves finer than ~2 pixels fade
    out, so a coarse bake does not alias what a fine bake resolves."""
    total, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        c = cell / (2 ** o)
        w = amp
        if pixel is not None:
            w *= float(np.clip((c / pixel - 1.0) / 2.0, 0.0, 1.0))
        if w > 0:
            total = total + w * (value_noise(X, Y, c, seed + 101 * o, period) - 0.5)
        norm += amp
        amp *= 0.5
    return total / norm if norm else 0.0


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---- staged, cached simulations ---------------------------------------------------------
_lock = threading.Lock()           # guards _cache and _key_locks
_key_locks = {}                    # one lock per stage key: each result is computed once
_cache = OrderedDict()
CACHE_SIZE = 24


def _canon(o):
    """Numbers as floats, so 1 and 1.0 hash alike (JSON from the browser has no 1.0)."""
    if isinstance(o, bool) or o is None or isinstance(o, str):
        return o
    if isinstance(o, (int, float, np.floating, np.integer)):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    return o


def _digest(obj):
    return hashlib.sha1(json.dumps(_canon(obj), sort_keys=True).encode()).hexdigest()


def sim_px_per_m(doc):
    return float(min(doc.texel_density, SIM_PX_PER_M))


class _Ctx:
    """A document plus the hashes its stages are keyed on (computed once per bake)."""

    def __init__(self, doc):
        self.doc = doc
        d = doc.to_dict()
        self.geom = _digest({k: d[k] for k in ("canvas_w", "canvas_h", "tiling", "panels", "wires", "generator")}
                            | {"ppm": sim_px_per_m(doc)})
        self.gravity = float(d["wire_sim"]["gravity_angle"])
        a = math.radians(self.gravity)
        self.gvec = (math.cos(a), math.sin(a))
        self.w = doc.weathering
        # Strokes as painted now: visible ones, moved along with their panels.
        self.strokes = brush.resolve(brush.strokes_from(d.get("strokes")), doc)


def _memo(stage, parts, fn):
    key = stage + ":" + _digest(parts)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        klock = _key_locks.setdefault(key, threading.Lock())
    # The editor's bake and the pathtracer's scene bake often ask for the same
    # document at once: the second waits for the first and reuses its result.
    # Different stages (and documents) compute in parallel.
    with klock:
        with _lock:
            if key in _cache:
                _cache.move_to_end(key)
                return _cache[key]
        out = fn()
        with _lock:
            _cache[key] = out
            _cache.move_to_end(key)
            while len(_cache) > CACHE_SIZE:
                old, _ = _cache.popitem(last=False)
                _key_locks.pop(old, None)
        return out


def _sim_grid(doc, ppm):
    from generators import Grid
    W, Hc = doc.canvas_w, doc.canvas_h
    nx, ny = max(16, int(round(W * ppm))), max(16, int(round(Hc * ppm)))
    return Grid(W, Hc, nx, ny, doc.tiling)


def _pad(a, r, wrap):
    return np.pad(a, [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2), mode="wrap" if wrap else "edge")


def _laplacian(H, pm, wrap):
    P = _pad(H, 1, wrap)
    return (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] - 4 * H) / (pm * pm)


def _gradient(H, pm, wrap):
    P = _pad(H, 1, wrap)
    return (P[1:-1, 2:] - P[1:-1, :-2]) / (2 * pm), (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * pm)


def _droplets(H, pm, wrap, gvec, n, rng, sources=None, max_steps=500):
    """Water droplets running over the height field.

    Each droplet heads in the gravity direction (with a little meander). A
    raised edge in its way blocks it: it slides along the edge, and if it
    cannot, it stops and pools there. Returns (visits, pools) per pixel."""
    ny, nx = H.shape
    gx, gy = _gradient(H, pm, wrap)
    if sources is None:
        px = rng.random(n) * nx
        py = rng.random(n) * ny
    else:
        flat = sources.ravel()
        p = flat / flat.sum()
        idx = rng.choice(flat.size, size=n, p=p)
        py, px = np.divmod(idx, nx)
        px = px + rng.random(n)
        py = py + rng.random(n)
    vx = np.full(n, gvec[0])
    vy = np.full(n, gvec[1])
    life = rng.integers(max_steps // 4, max_steps, n)
    alive = np.ones(n, bool)
    wgt = np.ones(n)
    visits = np.zeros(nx * ny)
    pools = np.zeros(nx * ny)
    for step in range(max_steps):
        idx = np.flatnonzero(alive)
        if not len(idx):
            break
        ix = np.floor(px[idx]).astype(np.int64)
        iy = np.floor(py[idx]).astype(np.int64)
        if wrap:
            ix %= nx
            iy %= ny
        cell = iy * nx + ix
        np.add.at(visits, cell, wgt[idx])
        sx, sy = gx.ravel()[cell], gy.ravel()[cell]
        # Desired direction: gravity plus inertia plus meander.
        dx = 0.65 * gvec[0] + 0.35 * vx[idx] + 0.35 * (rng.random(len(idx)) - 0.5)
        dy = 0.65 * gvec[1] + 0.35 * vy[idx] + 0.35 * (rng.random(len(idx)) - 0.5)
        # A rising slope ahead (> ~17 degrees) blocks: slide along it.
        g2 = sx * sx + sy * sy
        up = dx * sx + dy * sy
        blocked = (up > 0.3 * np.sqrt(np.maximum(g2, 1e-12))) & (g2 > 0.09)
        k = np.where(blocked, up / np.maximum(g2, 1e-12), 0.0)
        dx = dx - k * sx
        dy = dy - k * sy
        dl = np.hypot(dx, dy)
        stuck = dl < 0.15
        np.add.at(pools, cell[stuck], wgt[idx][stuck])
        dl = np.maximum(dl, 1e-9)
        vx[idx], vy[idx] = dx / dl, dy / dl
        px[idx] += vx[idx]
        py[idx] += vy[idx]
        wgt[idx] *= 0.997
        dead = stuck | (step >= life[idx])
        if not wrap:
            dead |= (px[idx] < 0) | (px[idx] >= nx) | (py[idx] < 0) | (py[idx] >= ny)
        alive[idx[dead]] = False
    return visits.reshape(ny, nx), pools.reshape(ny, nx)


def _terrain(c):
    """Height (with wires), wire coverage and emission on the simulation grid."""
    return _memo("terrain", [c.geom], lambda: _terrain_fields(c))


def _terrain_fields(c):
    from generators import get_generator
    doc = c.doc
    grid = _sim_grid(doc, sim_px_per_m(doc))
    H, ids, E, M = get_generator(doc.generator).height(doc, grid, emission=True)
    lum = (E[..., 0] * 0.2126 + E[..., 1] * 0.7152 + E[..., 2] * 0.0722) if E is not None else None
    return {"grid": (doc.canvas_w, doc.canvas_h, grid.nx, grid.ny, doc.tiling), "pm": grid.pm,
            "H": H.astype(np.float64), "M": M, "lum": lum}


def simulate(doc, _c=None):
    """Cached base weathering fields (edges, water streaks, rust, dirt)."""
    c = _c or _Ctx(doc)
    w = c.w
    return _memo("base", [c.geom, c.gravity, w["seed"], w["age"], w["streaks"], w["rust"]],
                 lambda: _base(c))


def _base(c):
    from .bake import ambient_occlusion
    T = _terrain(c)
    W, Hc, nx, ny, wrap = T["grid"]
    pm = T["pm"]
    H = T["H"]
    wset = c.w
    seed = wset["seed"]
    rng = np.random.default_rng(seed)
    period = (W, Hc) if wrap else None
    X = ((np.arange(nx) + 0.5) * (W / nx))[None, :]
    Y = ((np.arange(ny) + 0.5) * (Hc / ny))[:, None]

    # Edges and crevices from curvature measured over a few millimeters.
    Hb = gauss_blur(H, 1.0, wrap)
    curv = -_laplacian(Hb, pm, wrap)
    edge = np.clip(curv / 200.0, 0.0, 1.0)
    crev = np.clip(-curv / 200.0, 0.0, 1.0)
    ao = ambient_occlusion(H.astype(np.float32), pm, 0.03, wrap)
    occl = np.clip((1.0 - ao) * 1.6, 0.0, 1.0)

    # Water: droplets in the gravity direction.
    gvec = c.gvec
    area = W * Hc
    n_drops = int(min(60000, 5000 * area * (0.3 + wset["streaks"])))
    visits, pools = _droplets(H, pm, wrap, gvec, n_drops, rng)
    density = n_drops * 250.0 / (nx * ny)                    # expected visits per pixel
    streak = 1.0 - np.exp(-gauss_blur(visits, 0.7, wrap) / (2.5 * density))
    streak = np.clip(streak - 0.1, 0, 1) / 0.9
    moist = 1.0 - np.exp(-gauss_blur(pools, 2.5, wrap) / (0.05 * density + 1e-9))
    moist = np.clip(moist + 0.5 * crev + 0.35 * occl, 0.0, 1.0)

    # Scores compared against Age-driven thresholds (finished per bake with fine noise).
    big = fbm(X, Y, 0.06, seed + 3, 3, pm, period)
    mid = fbm(X, Y, 0.015, seed + 5, 3, pm, period)
    wear_score = edge + 0.35 * mid + 0.15 * big
    # Rust needs moisture or a crevice, and appears in patches, not solid bands.
    patch = np.clip(0.5 + 1.6 * big, 0.0, 1.0)
    rust_score = (0.6 * moist + 0.4 * crev + 0.15 * occl) * (0.35 + 0.9 * patch) + 0.25 * mid + 0.12 * big
    age = wset["age"]
    # Rust spreads further with age: blend in a blurred copy of itself.
    spread = gauss_blur(rust_score, 2.0 + 10.0 * age, wrap)
    rust_score = np.maximum(rust_score, 0.92 * spread)

    # Rust stain: water carries it downhill from where rust has formed.
    rust_th = _rust_threshold(wset)
    coarse_rust = _smooth(rust_th, rust_th + 0.1, rust_score)
    stain = np.zeros_like(H)
    if coarse_rust.sum() > 1.0 and wset["rust"] > 0:
        n_stain = int(min(40000, 2500 * area * (0.3 + wset["rust"])))
        sv, _ = _droplets(H, pm, wrap, gvec, n_stain, rng, sources=coarse_rust + 1e-9, max_steps=320)
        sd = n_stain * 120.0 / max(1.0, coarse_rust.sum())
        stain = 1.0 - np.exp(-gauss_blur(sv, 0.8, wrap) / (0.12 * sd + 1e-9))

    dirt = np.clip(0.55 * occl + 0.45 * crev + 0.5 * moist * 0.6 + 0.25 * streak + 0.3 * big, 0.0, 1.0)

    return {
        "grid": T["grid"],
        "wear_score": wear_score.astype(np.float32), "rust_score": rust_score.astype(np.float32),
        "streak": streak.astype(np.float32), "dirt": dirt.astype(np.float32),
        "stain": stain.astype(np.float32), "moist": moist.astype(np.float32),
    }


def _wire_stage(c):
    """Wire wear: where wires rest on convex edges (abrasion), and the streaks
    left by water dripping from their low points."""
    w = c.w
    return _memo("wires", [c.geom, c.gravity, w["seed"]], lambda: _wires(c))


def _wires(c):
    from generators import get_generator
    from .wire import resolved_path, resample
    doc = c.doc
    T = _terrain(c)
    W, Hc, nx, ny, wrap = T["grid"]
    pm = T["pm"]
    M = T["M"].astype(np.float64)
    # The panels without wires: the edges the wires press on.
    bare = type(doc).from_dict(doc.to_dict())
    bare.wires = []
    grid = _sim_grid(doc, sim_px_per_m(doc))
    H0 = get_generator(doc.generator).height(bare, grid)[0].astype(np.float64)
    edge0 = np.clip(-_laplacian(gauss_blur(H0, 1.0, wrap), pm, wrap) / 150.0, 0.0, 1.0)
    touch = gauss_blur(M, 1.0, wrap) * edge0
    on_wire = np.clip(3.0 * gauss_blur(touch, 1.5, wrap) * M, 0.0, 1.0)
    # Paint is rubbed off the edge around the contact, a little beyond the wire.
    rub = np.clip(4.0 * gauss_blur(touch, 0.008 / pm, wrap) * edge0, 0.0, 1.0)

    # Drips: rain runs along each wire to its low points (in the gravity
    # direction) and drips from there, with the water collected on the way.
    gx, gy = c.gvec
    src = np.zeros((ny, nx))
    total = 0.0
    for wr in doc.wires:
        if not wr.visible:
            continue
        P = resolved_path(wr, doc)
        if len(P) < 3:
            continue
        P = resample(P, 0.01)
        s = P[:, 0] * gx + P[:, 1] * gy                 # how far down each point is
        seg = np.hypot(*np.diff(P, axis=0).T)
        n = len(P)
        # Each segment drains toward its lower end; basins end at local highs.
        lows = [i for i in range(n) if (i == 0 or s[i] >= s[i - 1]) and (i == n - 1 or s[i] >= s[i + 1])]
        for i in lows:
            L = 0.0
            j = i
            while j > 0 and s[j - 1] <= s[j]:
                L += seg[j - 1]
                j -= 1
            j = i
            while j < n - 1 and s[j + 1] <= s[j]:
                L += seg[j]
                j += 1
            if i in (0, n - 1):
                L *= 0.5                                # an end runs into its connector
            if L < 0.02:
                continue
            off = wr.total_half_width() + 0.004
            x, y = P[i, 0] + gx * off, P[i, 1] + gy * off
            ix, iy = int(x / W * nx), int(y / Hc * ny)
            if wrap:
                ix, iy = ix % nx, iy % ny
            if 0 <= ix < nx and 0 <= iy < ny:
                src[iy, ix] += L
                total += L
    drips = np.zeros((ny, nx))
    if total > 0:
        rng = np.random.default_rng(c.w["seed"] + 17)
        n_drops = int(min(20000, 3000 * total))
        visits, _ = _droplets(T["H"], pm, wrap, c.gvec, n_drops, rng, sources=src, max_steps=int(0.6 / pm))
        per = n_drops / max(1, np.count_nonzero(src))
        drips = 1.0 - np.exp(-gauss_blur(visits, 1.2, wrap) / (0.06 * per + 1e-9))
    return {"grid": T["grid"], "on_wire": on_wire.astype(np.float32), "rub": rub.astype(np.float32),
            "drips": drips.astype(np.float32)}


def _shift(A, sx, sy, wrap):
    """A moved by (sx, sy) pixels (bilinear); zero flows in from outside unless wrapping."""
    ix, fx = int(math.floor(sx)), sx - math.floor(sx)
    iy, fy = int(math.floor(sy)), sy - math.floor(sy)
    out = 0.0
    for di, wx in ((ix, 1 - fx), (ix + 1, fx)):
        for dj, wy in ((iy, 1 - fy), (iy + 1, fy)):
            if wx * wy == 0:
                continue
            R = np.roll(A, (dj, di), axis=(0, 1))
            if not wrap:
                if di > 0:
                    R[:, :di] = 0
                elif di < 0:
                    R[:, di:] = 0
                if dj > 0:
                    R[:dj, :] = 0
                elif dj < 0:
                    R[dj:, :] = 0
            out = out + wx * wy * R
    return out


def _plume(src, gvec, pm, length, wrap, spread=0.35):
    """Smoke-like plume from `src`, rising against gravity for `length` m and widening."""
    step = 1.5
    n = max(1, int(length / (step * pm)))
    P = src.copy()
    acc = np.zeros_like(src)
    for k in range(n):
        P = _shift(P, -gvec[0] * step, -gvec[1] * step, wrap)
        P = gauss_blur(P, spread, wrap)
        acc += P * (0.965 ** k)
    return acc / max(1.0, sum(0.965 ** k for k in range(n)) * 0.25)


def _heat_stage(c):
    """Heat (temper tint) and soot around vents and bright lights."""
    return _memo("heat", [c.geom, c.gravity], lambda: _heat(c))


def _vent_mask(doc, grid):
    from .panel import shape_height
    V = np.zeros(grid.shape)
    for p in doc.panels:
        if not p.visible:
            continue
        for det in p.details:
            if det.kind != "vent":
                continue
            r = det.bound_radius()
            for cx, cy in det.centers(*p.rect):
                reg = grid.region(cx - r, cy - r, cx + r, cy + r)
                if reg is None:
                    continue
                key, X, Y = reg
                d = -det.local_sdf(X - cx, Y - cy)
                V[key] = np.maximum(V[key], np.clip(d / grid.pm + 0.5, 0.0, 1.0))
    return V


def _heat(c):
    doc = c.doc
    T = _terrain(c)
    W, Hc, _, _, wrap = T["grid"]
    grid = _sim_grid(doc, min(sim_px_per_m(doc), PLUME_PX_PER_M))
    nx, ny = grid.nx, grid.ny
    pm = grid.pm
    vents = _vent_mask(doc, grid)
    lum = T["lum"]
    emit = np.zeros((ny, nx))
    if lum is not None and lum.max() > 0:
        X = ((np.arange(nx) + 0.5) * (W / nx))[None, :]
        Y = ((np.arange(ny) + 0.5) * (Hc / ny))[:, None]
        L = _resample(gauss_blur(lum.astype(np.float64), 0.5 * T["grid"][2] / nx, wrap), T, X, Y)
        emit = 1.0 - np.exp(-0.25 * np.maximum(L, 0.0))
    hot = np.clip(vents + 0.45 * emit, 0.0, 1.0)
    near = gauss_blur(hot, 0.012 / pm, wrap)
    far = gauss_blur(hot, 0.035 / pm, wrap)
    rise = _plume(hot, c.gvec, pm, 0.08, wrap, spread=0.4)
    heat = np.clip(1.4 * near + 0.7 * far + 0.5 * rise, 0.0, 1.0)
    soot_src = np.clip(vents + 0.08 * emit, 0.0, 1.0)
    soot = np.clip(0.9 * _plume(soot_src, c.gvec, pm, 0.3, wrap, spread=0.55)
                   + 0.5 * gauss_blur(soot_src, 0.01 / pm, wrap), 0.0, 1.0)
    return {"grid": (W, Hc, nx, ny, wrap), "heat": heat.astype(np.float32), "soot": soot.astype(np.float32)}


def _leak_stage(c):
    doc = c.doc
    w = c.w
    srcs = fluids.sources(doc, w["auto_leaks"], w["seed"])
    if not srcs:
        return None
    return _memo("leaks", [c.geom, c.gravity, w["seed"], srcs], lambda: _leaks(c, srcs))


def _leaks(c, srcs):
    T = _terrain(c)
    W, Hc, nx, ny, wrap = T["grid"]
    r = fluids.simulate(T["H"], W, Hc, wrap, c.gvec, srcs, c.w["seed"])
    fn = lambda a, s, k: np.clip(gauss_blur(a.astype(np.float64), s, wrap) * k, 0.0, 1.0).astype(np.float32)
    for liquid, (wet, stain, res) in r["fields"].items():
        wide = 1.6 if liquid == "oil" else 1.0         # oil spreads into wide, soft stains
        st = fn(stain, wide, 2.2)
        # Mineral rings and crust: the edges of the trail, plus some where it dried.
        sb = gauss_blur(st.astype(np.float64), 0.8, wrap)
        gy, gx = np.gradient(sb)
        rim = np.clip(np.hypot(gx, gy) * 3.0, 0.0, 1.0)
        r["fields"][liquid] = (fn(wet, wide, 2.0), st,
                               np.clip(0.75 * rim + 0.35 * fn(res, 1.0, 1.0) * st, 0, 1).astype(np.float32))
    return r


def _stroke_stage(c):
    if not c.strokes:
        return None
    doc = c.doc
    return _memo("strokes", [doc.canvas_w, doc.canvas_h, doc.tiling, sim_px_per_m(doc), c.strokes],
                 lambda: _strokes(c))


def _strokes(c):
    doc = c.doc
    grid = _sim_grid(doc, sim_px_per_m(doc))
    layers = brush.rasterize(c.strokes, grid)
    return {"grid": (doc.canvas_w, doc.canvas_h, grid.nx, grid.ny, doc.tiling), "layers": layers}


_pool = None


def _prefetch(c):
    """Compute the stages this bake needs in parallel (each is cached, so the
    composition below then just reads them)."""
    global _pool
    w = c.w
    doc = c.doc
    jobs = []
    if w["age"] > 0:
        jobs.append(lambda: simulate(doc, c))
        if w["wire_wear"] > 0 and any(wr.visible for wr in doc.wires):
            jobs.append(lambda: _wire_stage(c))
        if w["heat"] > 0:
            jobs.append(lambda: _heat_stage(c))
    jobs.append(lambda: _leak_stage(c))
    jobs.append(lambda: _stroke_stage(c))
    if len(jobs) < 2:
        return
    _terrain(c)                      # shared by all of them: compute it once first
    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="weather")
    for f in [_pool.submit(j) for j in jobs]:
        f.result()


def _wear_threshold(w):
    return 1.15 - 0.95 * w["age"] * (0.4 + 1.2 * w["edge_wear"])


def _rust_threshold(w):
    return 1.25 - 0.85 * w["age"] * (0.3 + 1.4 * w["rust"])


class _Sampler:
    """Bilinear sampling of fields on one simulation grid at a bake's pixels
    (X a row, Y a column of world coordinates). Indices are computed once."""

    def __init__(self, grid, X, Y):
        W, Hc, nx, ny, wrap = grid
        fx = np.ravel(X) / (W / nx) - 0.5
        fy = np.ravel(Y) / (Hc / ny) - 0.5
        x0 = np.floor(fx).astype(np.int64)
        y0 = np.floor(fy).astype(np.int64)
        self.tx = (fx - x0).astype(np.float32)
        self.ty = (fy - y0).astype(np.float32)[:, None]
        if wrap:
            fix_x, fix_y = (lambda i: np.mod(i, nx)), (lambda j: np.mod(j, ny))
        else:
            fix_x, fix_y = (lambda i: np.clip(i, 0, nx - 1)), (lambda j: np.clip(j, 0, ny - 1))
        self.x0, self.x1 = fix_x(x0), fix_x(x0 + 1)
        self.y0, self.y1 = fix_y(y0), fix_y(y0 + 1)
        self.shape = (len(fy), len(fx))

    def __call__(self, field):
        if not field.any():
            return np.zeros(self.shape, np.float32)
        f = field.astype(np.float32, copy=False)
        rows = f[:, self.x0] * (1 - self.tx) + f[:, self.x1] * self.tx
        return rows[self.y0] * (1 - self.ty) + rows[self.y1] * self.ty


def _resample(field, sim, X, Y):
    """Bilinear samples of a simulation-grid field at world points."""
    return _Sampler(sim["grid"], X, Y)(field)


# ---- per-bake composition ---------------------------------------------------------------
def _panel_variation(ids):
    v = np.full(ids.shape, 0.5)
    m = ids > 0
    x = np.sin(ids[m].astype(np.float64) * 12.9898 + 4.1414) * 43758.5453
    v[m] = x - np.floor(x)
    return v


def _temper(t):
    """Temper color for heat t in [0, 1] (0 = barely warm, 1 = hottest)."""
    x = np.clip(t, 0.0, 1.0) * (len(TEMPER) - 1)
    i = np.minimum(np.floor(x).astype(int), len(TEMPER) - 2)
    f = (x - i)[..., None]
    return TEMPER[i] * (1 - f) + TEMPER[i + 1] * f


def _mix(col, target, a):
    """Blend color array toward target by a (per pixel)."""
    if not np.any(a):
        return col
    a = np.asarray(a, np.float32)[..., None]
    return col * (1 - a) + np.asarray(target, np.float32) * a


def _brushed(layers, ch, field, X, Y, fine):
    """Apply hand-painted strokes of channel `ch` over a simulated field."""
    if ch not in layers:
        return field
    add, era = layers[ch]
    field = field * (1.0 - era)
    if ch in ("rust", "wear"):
        # Crisp, irregular edges like the simulated effect.
        shaped = _smooth(0.12, 0.4, add + 0.22 * fine * (add > 0.02))
    else:
        shaped = np.clip(add * (1.0 + 0.5 * fine), 0.0, 1.0)
    return np.maximum(field, shaped)


_surf_cache = OrderedDict()


def surface(doc, grid, ids, wire_cov, weathering=True):
    """Material maps and weathering for one bake grid.

    Returns (height_delta [m], basecolor linear RGB, roughness, metallic,
    masks dict). All arrays have the grid's shape. Whole-canvas results are
    kept briefly: the editor and the pathtracer bake the same document at the
    same size right after every edit."""
    if grid.windowed:
        return _surface(doc, grid, ids, wire_cov, weathering)
    key = _digest([doc.to_dict(), grid.nx, grid.ny, bool(weathering)])
    with _lock:
        klock = _key_locks.setdefault("surface:" + key, threading.Lock())
    with klock:
        with _lock:
            if key in _surf_cache:
                _surf_cache.move_to_end(key)
                return _surf_cache[key]
        out = _surface(doc, grid, ids, wire_cov, weathering)
        with _lock:
            _surf_cache[key] = out
            while len(_surf_cache) > 2:
                old, _ = _surf_cache.popitem(last=False)
                _key_locks.pop("surface:" + old, None)
    return out


def _surface(doc, grid, ids, wire_cov, weathering=True):
    mats = doc.materials
    shape = grid.shape
    X = ((grid.wi0 + np.arange(grid.wn) + 0.5) * grid.pmx)[None, :]
    Y = ((grid.wj0 + np.arange(grid.hn) + 0.5) * grid.pmy)[:, None]
    pm = grid.pm
    metal_c = np.array(mats["metal"]["color"])
    paint = mats["paint"]
    var = _panel_variation(ids) - 0.5
    k = 1.0 + mats["variation"] * var

    z = np.zeros(shape, np.float32)
    dh = z.copy()
    wear = rust = blister = streak = stain = dirt = z
    heat = soot = wet = z
    water = water_res = oil = cool = cool_res = z
    wire_rub = wire_age = z
    cool_color = np.array(fluids.LIQUIDS["coolant"]["color"])
    on = weathering and active(doc)
    if on:
        c = _Ctx(doc)
        w = c.w
        age = w["age"]
        _prefetch(c)
        period = (doc.canvas_w, doc.canvas_h) if doc.tiling else None
        seed = w["seed"]
        fine = fbm(X, Y, 0.004, seed + 11, 3, pm, period)          # chip edges, rust rims
        samplers = {}

        def S(f, st):
            g = st["grid"]
            if g not in samplers:
                samplers[g] = _Sampler(g, X, Y)
            return samplers[g](f)

        # Liquid leaks (independent of Age: they are placed by hand or counted).
        L = _leak_stage(c)
        if L is not None:
            F = L["fields"]
            # Ragged trail edges: centimeter-scale noise on top of the fine noise,
            # only where a trail is (never outside it).
            edge = 0.25 * fine + 0.3 * fbm(X, Y, 0.015, seed + 61, 2, pm, period)
            trail = lambda a, gain, e=1.0: np.clip(a * gain + e * edge, 0, 1) * _smooth(0.0, 0.15, a)
            if "water" in F:
                wt, st, rs = (S(a, L) for a in F["water"])
                water = trail(st, 1.3)
                wet = np.maximum(wet, trail(wt, 1.5))
                water_res = trail(rs, 1.2, 0.4)
            if "oil" in F:
                wt, st, _ = (S(a, L) for a in F["oil"])
                oil = trail(st, 1.3)
                wet = np.maximum(wet, trail(wt, 1.5) * 0.8)
            if "coolant" in F:
                wt, st, rs = (S(a, L) for a in F["coolant"])
                cool = trail(st, 1.2)
                wet = np.maximum(wet, trail(wt, 1.5))
                cool_res = trail(rs, 1.2, 0.4)
                cool_color = L["coolant_color"]

        if age > 0:
            sim = simulate(doc, c)
            wt_, rt = _wear_threshold(w), _rust_threshold(w)
            wear = _smooth(wt_, wt_ + 0.06, S(sim["wear_score"], sim) + 0.35 * fine)
            # Leaking water rusts what it runs over.
            rs = S(sim["rust_score"], sim) + 0.18 * fine + 0.45 * water * w["rust"]
            rust = _smooth(rt, rt + 0.06, rs)
            blister = (_smooth(rt - 0.12, rt - 0.02, rs) - rust).clip(0, 1) if paint["enabled"] else z
            streak = np.clip(S(sim["streak"], sim) * age * (0.4 + 1.2 * w["streaks"]), 0, 1)
            stain = np.clip(S(sim["stain"], sim) * w["rust"] * min(1.0, 1.4 * age), 0, 1)
            dirt = np.clip((S(sim["dirt"], sim) - 0.25 + 0.1 * fine) * age * (0.5 + 1.5 * w["dirt"]), 0, 1)
            ww = w["wire_wear"] * age
            if ww > 0 and any(wr.visible for wr in doc.wires):
                ws = _wire_stage(c)
                wear = np.maximum(wear, _smooth(0.25, 0.5, S(ws["rub"], ws) * min(1.0, 2.0 * ww) + 0.3 * fine))
                streak = np.maximum(streak, np.clip(S(ws["drips"], ws) * min(1.0, 1.6 * ww), 0, 1))
                wire_rub = np.clip(S(ws["on_wire"], ws) * min(1.0, 2.0 * ww) * (0.7 + 0.6 * fine), 0, 1)
                wire_age = np.full(shape, ww)
            if w["heat"] > 0:
                hs = _heat_stage(c)
                amt = w["heat"] * (0.35 + 0.65 * age)
                heat = np.clip(S(hs["heat"], hs) * amt * 1.3, 0, 1)
                soot = np.clip(S(hs["soot"], hs) * w["heat"] * age * (1.2 + 0.5 * fine), 0, 1)

        # Hand-painted strokes go over (or erase) the simulated effects.
        B = _stroke_stage(c)
        if B is not None:
            layers = {ch: (S(a, B), S(e, B)) for ch, (a, e) in B["layers"].items()}
            rust = _brushed(layers, "rust", rust, X, Y, fine)
            wear = _brushed(layers, "wear", wear, X, Y, fine)
            dirt = _brushed(layers, "dirt", dirt, X, Y, fine)
            streak = _brushed(layers, "streaks", streak, X, Y, fine)
            oil = _brushed(layers, "oil", oil, X, Y, fine)
            soot = _brushed(layers, "soot", soot, X, Y, fine)
            heat = _brushed(layers, "heat", heat, X, Y, fine)
            if "rust" in layers:
                stain = stain * (1.0 - layers["rust"][1])

        # Height detail: pits in rust, bumps under blistered paint, mineral crust.
        pit = fbm(X, Y, 0.0025, seed + 21, 3, pm, period)
        bump = fbm(X, Y, 0.006, seed + 23, 2, pm, period)
        dh = dh - rust * (0.00018 + 0.00035 * np.clip(pit + 0.5, 0, 1)) + rust * 0.00008 * (bump + 0.5)
        dh = dh + blister * 0.0003 * np.clip(bump + 0.5, 0, 1)
        dh = dh + 0.00004 * (water_res + cool_res) * np.clip(bump + 0.7, 0, 1)

    f32 = lambda a: np.asarray(a, np.float32)
    wear, rust, blister, streak, stain, dirt = map(f32, (wear, rust, blister, streak, stain, dirt))
    heat, soot, wet, water, water_res, oil, cool, cool_res, wire_rub, wire_age = map(
        f32, (heat, soot, wet, water, water_res, oil, cool, cool_res, wire_rub, wire_age))
    var = f32(var)
    k = f32(k)

    # Panels: paint over metal, worn and rusted through where weathered.
    if paint["enabled"]:
        present = (1.0 - wear) * (1.0 - rust)
        col = metal_c[None, None] * (1 - present[..., None]) + np.array(paint["color"])[None, None] * present[..., None]
        rough = mats["metal"]["roughness"] * (1 - present) + paint["roughness"] * present
        met = 1.0 * (1 - present) + paint["metallic"] * present
        dh = dh + PAINT_THICKNESS * present
    else:
        present = z
        col = metal_c[None, None] * (1.0 + 0.18 * wear[..., None]) * np.ones(shape + (3,))
        rough = mats["metal"]["roughness"] * (1.0 - 0.55 * wear)
        met = np.ones(shape)
    col = (col * k[..., None]).astype(np.float32)
    rough = (rough + 0.6 * mats["variation"] * var).astype(np.float32)
    met = met.astype(np.float32)
    if on:
        # Heat: temper colors on bare metal, scorched paint.
        if heat.any():
            a = _smooth(0.04, 0.35, heat) * (1 - present)
            col = _mix(col, _temper(heat) * (0.75 + 0.25 * k[..., None]), 0.85 * a)
            s = np.clip(heat - 0.2, 0, 1) ** 1.5 * present
            col = _mix(col, SCORCH_COLOR, 0.8 * s)
            rough = rough + (0.7 - rough) * s
        # Grime, streaks and stain over everything panel.
        col = col * (1 - 0.45 * streak[..., None]) + STREAK_COLOR * (0.2 * streak[..., None])
        rough = rough + (0.78 - rough) * 0.6 * streak
        col = _mix(col, STAIN_COLOR, 0.75 * stain)
        rough = rough + (0.82 - rough) * stain
        met = met * (1 - 0.7 * stain)
        col = _mix(col, DIRT_COLOR, 0.8 * dirt)
        rough = rough + (0.92 - rough) * dirt
        met = met * (1 - 0.85 * dirt)
        # Rust: blistered paint darkens, rust itself is orange-brown, rough, not metallic.
        col = col * (1 - 0.12 * blister[..., None])
        if rust.any():
            tone = np.clip(0.5 + 1.4 * fbm(X, Y, 0.01, doc.weathering["seed"] + 31, 3, pm,
                                           (doc.canvas_w, doc.canvas_h) if doc.tiling else None), 0, 1)
            rc = RUST_DARK[None, None] * (1 - tone[..., None]) + RUST_LIGHT[None, None] * tone[..., None]
            col = col * (1 - rust[..., None]) + rc * rust[..., None]
            rough = rough + (0.88 - rough) * rust
            met = met * (1 - rust)
        # Leaks: water darkens and leaves mineral rings, coolant a tinted crust,
        # oil a dark glossy stain.
        col = _mix(col, STREAK_COLOR, 0.35 * water)          # water trails carry grime
        rough = rough + (0.7 - rough) * 0.4 * water
        col = _mix(col, MINERAL_COLOR, 0.6 * water_res)
        rough = rough + (0.85 - rough) * water_res
        met = met * (1 - 0.6 * water_res)
        col = _mix(col, cool_color * 0.55, 0.6 * cool)
        col = _mix(col, cool_color * 0.7 + 0.25, 0.6 * cool_res)
        rough = rough + (0.75 - rough) * cool_res
        met = met * (1 - 0.5 * np.maximum(cool, cool_res))
        col = _mix(col, fluids.LIQUIDS["oil"]["color"], 0.85 * oil)
        rough = rough + (0.2 - rough) * oil
        met = met * (1 - 0.8 * oil)
        # Soot on top, then a wet film (glossy, slightly darker).
        col = _mix(col, SOOT_COLOR, 0.9 * soot)
        rough = rough + (0.95 - rough) * soot
        met = met * (1 - soot)
        col = col * (1 - 0.3 * wet[..., None])
        rough = rough + (0.06 - rough) * wet

    # Wires keep their own material: they fade and crack with age, scuff where
    # they rub, and get some grime, soot and wetness.
    wc = np.clip(wire_cov, 0, 1)
    wm = mats["wire"]
    wcol = np.array(wm["color"])[None, None] * np.ones(shape + (3,))
    wr = np.full(shape, wm["roughness"])
    wmet = np.full(shape, wm["metallic"])
    if on:
        fade = np.clip(wire_age * (0.55 + 0.4 * fbm(X, Y, 0.03, seed + 51, 2, pm, period)), 0, 1)
        wcol = _mix(wcol, CHALK_COLOR, 0.45 * fade)
        wr = wr + (0.85 - wr) * fade
        wcol = _mix(wcol, CHALK_COLOR * 1.4, 0.6 * wire_rub)
        wr = wr + (0.9 - wr) * wire_rub
        wcol = _mix(wcol, DIRT_COLOR, 0.5 * dirt)
        wr = wr + (0.9 - wr) * 0.5 * dirt
        wcol = _mix(wcol, SOOT_COLOR, 0.7 * soot)
        wcol = _mix(wcol, fluids.LIQUIDS["oil"]["color"], 0.7 * oil)
        wr = wr + (0.1 - wr) * np.maximum(wet, 0.8 * oil)
        if wire_age.any():
            # Cracks: thin lines where a noise field crosses zero.
            cr = np.abs(fbm(X, Y, 0.012, seed + 53, 2, pm, period))
            crack = (1.0 - _smooth(0.0, 0.02, cr)) * _smooth(0.3, 0.8, wire_age)
            dh = dh - wc * crack * 0.00025
            wcol = wcol * (1 - 0.5 * crack[..., None] * wc[..., None])
    col = col * (1 - wc[..., None]) + wcol * wc[..., None]
    rough = rough * (1 - wc) + wr * wc
    met = met * (1 - wc) + wmet * wc

    masks = {m: z for m in MASKS}
    if on:
        pw = 1 - wc
        masks = {"wear": wear * pw, "dirt": dirt, "streaks": streak, "rust": np.clip(rust + 0.6 * stain, 0, 1) * pw,
                 "wet": wet, "fluid": np.clip(np.maximum(oil, cool), 0, 1),
                 "residue": np.clip(np.maximum(water_res, cool_res), 0, 1) * pw,
                 "soot": soot, "heat": heat * pw}

    return (dh.astype(np.float32), np.clip(col, 0, 1).astype(np.float32),
            np.clip(rough, 0.02, 1).astype(np.float32), np.clip(met, 0, 1).astype(np.float32),
            {m: np.broadcast_to(v, shape).astype(np.float32) for m, v in masks.items()})
