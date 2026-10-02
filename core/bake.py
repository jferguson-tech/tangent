"""Bake a document into texture maps.

Heights come from the generator as continuous float values evaluated from
distance fields, so central differences give clean normals with no
quantization stair-steps. Optional supersampling averages normals down.
"""
from __future__ import annotations

import colorsys
import time

import numpy as np

from generators import Grid, get_generator
from .filters import gauss_blur, linear_to_srgb
from .png import to_uint

# Exportable maps. "spill" (blurred emission) is a preview helper for the editor.
ALL_MAPS = ("normal", "height", "ao", "curvature", "id", "emissive", "wires")
# Preview-only maps: blurred emission for the editor's fake spill, and the raw
# float height field (panels only when the client hides wires) for the wire sim.
PREVIEW_MAPS = ALL_MAPS + ("spill", "height_raw")

# Fake light spill for the realtime view: blur radii (meters) and weights.
SPILL_RADII = ((0.012, 0.5), (0.04, 0.33), (0.12, 0.17))
SPILL_MAX_RES = 256          # spill is a soft field, so it is computed small


def _pad(a, r, tiling):
    return np.pad(a, r, mode="wrap" if tiling else "edge")


def normals_from_height(H, pmx, pmy, strength, tiling):
    """Unit normals in tangent space, OpenGL convention (green = +V = up)."""
    P = _pad(H, 1, tiling)
    dhdx = (P[1:-1, 2:] - P[1:-1, :-2]) / (2 * pmx)
    dhdrow = (P[2:, 1:-1] - P[:-2, 1:-1]) / (2 * pmy)
    # Image rows run downward while +V runs up, so dH/dv = -dH/drow.
    n = np.empty(H.shape + (3,), np.float32)
    n[..., 0] = -strength * dhdx
    n[..., 1] = strength * dhdrow
    n[..., 2] = 1.0
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def _downsample(a, ss):
    if ss == 1:
        return a
    h, w = a.shape[0] // ss, a.shape[1] // ss
    shp = (h, ss, w, ss) + a.shape[2:]
    return a[: h * ss, : w * ss].reshape(shp).mean(axis=(1, 3))


def ambient_occlusion(H, pm, radius_m, tiling, dirs=8, steps=6):
    """Horizon-based AO from the height field (1 = unoccluded)."""
    rmax = float(np.clip(radius_m / pm, 2.0, 64.0))
    R = int(np.ceil(rmax)) + 1
    P = _pad(H, R, tiling)
    ny, nx = H.shape
    radii = np.geomspace(1.0, rmax, steps)
    occ = np.zeros_like(H, dtype=np.float32)
    for k in range(dirs):
        a = 2 * np.pi * k / dirs
        best = np.zeros_like(H, dtype=np.float32)
        seen = set()
        for r in radii:
            ox, oy = int(round(np.cos(a) * r)), int(round(np.sin(a) * r))
            if (ox, oy) in seen or (ox == 0 and oy == 0):
                continue
            seen.add((ox, oy))
            win = P[R + oy: R + oy + ny, R + ox: R + ox + nx]
            tan = (win - H) / (np.hypot(ox, oy) * pm)
            np.maximum(best, tan, out=best)
        occ += best / np.sqrt(1.0 + best * best)
    return np.clip(1.0 - occ / dirs, 0.0, 1.0)


def curvature(H, pm, tiling):
    """Convex edges bright, concave dark, 0.5 = flat. Auto-normalized."""
    P = _pad(H, 1, tiling)
    lap = (P[:-2, 1:-1] + P[2:, 1:-1] + P[1:-1, :-2] + P[1:-1, 2:] - 4 * H) / (pm * pm)
    c = -lap
    Q = _pad(c, 1, tiling)
    c = sum(Q[j: j + H.shape[0], i: i + H.shape[1]] for j in range(3) for i in range(3)) / 9.0
    scale = float(np.percentile(np.abs(c), 99.5))
    if scale <= 0:
        return np.full(H.shape, 0.5, np.float32)
    return (0.5 + 0.5 * np.clip(c / scale, -1.0, 1.0)).astype(np.float32)


def id_colors(ids):
    """Stable distinct color per panel index, black for the base."""
    n = int(ids.max()) + 1
    lut = np.zeros((max(n, 1), 3), np.float32)
    for i in range(1, n):
        hue = (i * 0.61803398875) % 1.0
        lut[i] = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return lut[ids]


class BakeResult:
    def __init__(self, maps, info):
        self.maps = maps
        self.info = info

    def encode(self, name, bits=8, convention="gl"):
        """Return a uint8/uint16 array ready for PNG encoding."""
        m = self.maps[name]
        if name == "normal":
            rgb = m * 0.5 + 0.5
            if convention == "dx":
                rgb = rgb.copy()
                rgb[..., 1] = 1.0 - rgb[..., 1]
            return to_uint(rgb, bits)
        if name == "height":
            lo, hi = self.info["height_min"], self.info["height_max"]
            v = (m - lo) / (hi - lo) if hi > lo else np.full_like(m, 0.5)
            return to_uint(v, bits)
        if name == "wires":
            return to_uint(m, 8)
        if name == "id":
            return to_uint(m, 8)
        if name in ("emissive", "spill"):
            # sRGB color normalized so the brightest texel is white; the scale is in info.
            scale = self.info.get(f"{name}_scale", 0.0)
            v = m / scale if scale > 0 else np.zeros_like(m)
            return to_uint(linear_to_srgb(v), 8)
        return to_uint(m, bits)


def bake(doc, maps=("normal", "height"), max_res=None, ss=1, ao_radius=0.05):
    t0 = time.perf_counter()
    nx, ny = doc.resolution()
    if max_res and max(nx, ny) > max_res:
        k = max_res / max(nx, ny)
        nx, ny = max(8, int(round(nx * k))), max(8, int(round(ny * k)))
    ss = int(max(1, min(4, ss)))
    grid = Grid(doc.canvas_w, doc.canvas_h, nx * ss, ny * ss, doc.tiling)
    gen = get_generator(doc.generator)
    want = set(maps)
    need_emit = bool(want & {"emissive", "spill"})
    Hs, ids_s, Es, Ms = gen.height(doc, grid, emission=need_emit)

    out = {}
    if "normal" in want:
        n = normals_from_height(Hs, grid.pmx, grid.pmy, doc.normal_strength, doc.tiling)
        n = _downsample(n, ss)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        out["normal"] = n.astype(np.float32)
    H = _downsample(Hs, ss).astype(np.float32)
    pm = 0.5 * (doc.canvas_w / nx + doc.canvas_h / ny)
    out["height"] = H
    if "ao" in want:
        out["ao"] = ambient_occlusion(H, pm, ao_radius, doc.tiling)
    if "curvature" in want:
        out["curvature"] = curvature(H, pm, doc.tiling)
    if Ms is not None and ("wires" in want or "wiremask" in want):  # wiremask: internal alias
        out["wires"] = _downsample(Ms, ss).astype(np.float32)
    if "height_raw" in want:
        out["height_raw"] = H
    if ids_s is not None and ("id" in want or "ids" in want):
        ids = ids_s[ss // 2:: ss, ss // 2:: ss][:ny, :nx]
        if "id" in want:
            out["id"] = id_colors(ids)
        if "ids" in want:
            out["ids"] = ids
    info = {}
    if need_emit:
        E = _downsample(Es, ss).astype(np.float32)
        out["emissive"] = E
        info["emissive_scale"] = float(E.max())
        info["has_emission"] = bool(E.max() > 0)
        if "spill" in want:
            f = max(1, int(np.ceil(max(nx, ny) / SPILL_MAX_RES)))
            h2, w2 = ny // f, nx // f
            Es = E[: h2 * f, : w2 * f].reshape(h2, f, w2, f, 3).mean(axis=(1, 3))
            S = np.zeros_like(Es)
            if info["has_emission"]:
                for radius, wgt in SPILL_RADII:
                    S += wgt * gauss_blur(Es, radius / (pm * f), wrap=doc.tiling)
            out["spill"] = S
            info["spill_scale"] = float(S.max())
            info["spill_radius_m"] = SPILL_RADII[1][0]
    info.update({
        "width": nx, "height": ny,
        "full_width": doc.resolution()[0], "full_height": doc.resolution()[1],
        "height_min": float(H.min()), "height_max": float(H.max()),
        "pixel_m": pm, "ms": round((time.perf_counter() - t0) * 1000, 1),
    })
    return BakeResult(out, info)
