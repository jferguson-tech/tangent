"""Build pathtracer scene arrays from a Document and preview settings.

Pure NumPy, so it is importable and testable without Numba.
"""
from __future__ import annotations

import copy
import json
import math
import threading

import numpy as np

from core import bake as bakemod
from core.filters import bloom as apply_bloom
from .presets import LIGHT_PRESETS, MATERIAL_PRESETS, DEFAULT_RENDER
from .hdri import EnvCache

ENV_CACHE = EnvCache()
BACKGROUNDS = {"hdri": 0, "black": 1, "blurred": 2}

RENDER_TEX_MAX = 1024        # height/normal texture size used by the tracer

# Camera, light and material changes restart the render but keep the document,
# so the last bake is reused instead of baking again.
_BAKE_CACHE = {"key": None, "res": None}
_BAKE_LOCK = threading.Lock()


def _scene_bake(doc):
    key = json.dumps(doc.to_dict(), sort_keys=True)
    with _BAKE_LOCK:
        if _BAKE_CACHE["key"] == key:
            return _BAKE_CACHE["res"]
    res = bakemod.bake(doc, maps=("normal", "height", "ids", "emissive", "basecolor", "roughness", "metallic"),
                       max_res=RENDER_TEX_MAX)
    with _BAKE_LOCK:
        _BAKE_CACHE["key"], _BAKE_CACHE["res"] = key, res
    return res


def merge_settings(user):
    s = copy.deepcopy(DEFAULT_RENDER)
    for k, v in (user or {}).items():
        if k in s and isinstance(s[k], dict) and isinstance(v, dict):
            s[k].update({kk: vv for kk, vv in v.items() if kk in s[k] or kk == "preset"})
        elif k in s:
            s[k] = v
    s["width"] = int(np.clip(int(s["width"]), 64, 1920))
    s["height"] = int(np.clip(int(s["height"]), 64, 1200))
    s["max_spp"] = int(np.clip(int(s["max_spp"]), 1, 65536))
    s["bounces"] = int(np.clip(int(s["bounces"]), 0, 8))
    s["tiles"] = 3 if int(s["tiles"]) >= 3 else 1
    s["displacement"] = bool(s["displacement"])
    e = s["environment"]
    e["hdri"] = str(e.get("hdri") or "dark_hangar")
    e["intensity"] = float(np.clip(float(e.get("intensity", 1.0)), 0.0, 100.0))
    e["rotation"] = float(e.get("rotation", 0.0)) % 360.0
    if e.get("background") not in BACKGROUNDS:
        e["background"] = "hdri"
    s["scene_light"] = float(np.clip(float(s.get("scene_light", 1.0)), 0.0, 4.0))
    b = s["bloom"]
    b["pathtrace"] = bool(b.get("pathtrace", True))
    b["editor"] = bool(b.get("editor", True))
    b["threshold"] = float(np.clip(float(b.get("threshold", 1.5)), 0.0, 100.0))
    b["intensity"] = float(np.clip(float(b.get("intensity", 0.2)), 0.0, 10.0))
    b["radius"] = float(np.clip(float(b.get("radius", 1.0)), 0.1, 5.0))
    return s


def emitter_tables(Em, width_m, height_m):
    """Sampling tables for emissive texels: power-proportional CDF and the
    matching density per unit area of the canvas plane."""
    Em = np.ascontiguousarray(Em, np.float64)
    h, w = Em.shape[:2]
    lum = Em @ np.array([0.2126, 0.7152, 0.0722])
    total = float(lum.sum())
    texel_area = (width_m / w) * (height_m / h)
    if total <= 0:
        return Em, np.zeros((h, w)), np.array([0.0, 1.0]), np.zeros(1, np.int64)
    epd = lum / total / texel_area
    idx = np.flatnonzero(lum.ravel() > 0).astype(np.int64)   # only emissive texels
    cdf = np.concatenate([[0.0], np.cumsum(lum.ravel()[idx]) / total])
    cdf[-1] = 1.0
    return Em, np.ascontiguousarray(epd), cdf, idx


def _camera(c, S, aspect):
    yaw, pitch = math.radians(float(c["yaw"])), math.radians(float(c["pitch"]))
    dist = float(c["distance"]) * S
    pos = np.array([dist * math.cos(pitch) * math.sin(yaw),
                    -dist * math.cos(pitch) * math.cos(yaw),
                    dist * math.sin(pitch)])
    f = -pos / np.linalg.norm(pos)
    r = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0.0, 0.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    tan_half = math.tan(math.radians(float(c["fov"])) * 0.5)
    return np.concatenate([pos, f, r, u, [tan_half, aspect]]).astype(np.float64)


def _lights(ls, S):
    preset = LIGHT_PRESETS.get(ls.get("preset"), LIGHT_PRESETS["scifi_spot"])
    rows = []
    for i, spec in enumerate(preset["lights"]):
        spec = dict(spec)
        if i == 0:  # the key light follows the user controls
            for k in ("azimuth", "elevation", "cone", "color"):
                if k in ls:
                    spec[k] = ls[k]
            spec["power"] = spec["power"] * float(ls.get("intensity", 1.0))
        az, el = math.radians(float(spec["azimuth"])), math.radians(float(spec["elevation"]))
        D = float(spec["distance"]) * S
        aim = np.array([spec["aim"][0] * S, spec["aim"][1] * S, 0.0])
        pos = aim + D * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                  math.sin(el)])
        d = aim - pos
        d /= np.linalg.norm(d)
        half = math.radians(float(spec["cone"])) * 0.5
        rgb = np.array(spec["color"], np.float64) * float(spec["power"]) * D * D
        rows.append([*pos, *d, *rgb, math.cos(half * (1 - float(spec["softness"]))),
                     math.cos(half), float(spec["radius"]) * S, 0.0, 0.0])
    return np.array(rows, np.float64).reshape(-1, 14), np.array(preset["env"], np.float64).ravel()


def light_positions(doc, settings):
    """Key/fill light positions in canvas UV (for drawing gizmos in the editor)."""
    s = merge_settings(settings)
    S = max(doc.canvas_w, doc.canvas_h)
    L, _ = _lights(s["light"], S)
    out = []
    for row in L:
        out.append({"u": row[0] / doc.canvas_w + 0.5, "v": 0.5 - row[1] / doc.canvas_h,
                    "z": row[2] / S})
    return out


def build_scene(doc, settings):
    s = merge_settings(settings)
    res = _scene_bake(doc)
    hs = float(s["height_scale"])
    Hm = res.maps["height"].astype(np.float64) * hs
    Nm = res.maps["normal"].astype(np.float64)
    Mt = np.ascontiguousarray(np.concatenate([
        res.maps["basecolor"].astype(np.float64), res.maps["metallic"][..., None].astype(np.float64),
        res.maps["roughness"][..., None].astype(np.float64)], axis=-1))
    W, Hh = doc.canvas_w, doc.canvas_h
    t = s["tiles"]
    texel = W / Hm.shape[1]
    S = max(W, Hh)
    sc = np.array([
        -0.5 * W * t, 0.5 * W * t, -0.5 * Hh * t, 0.5 * Hh * t,
        float(Hm.min()), float(Hm.max()), W, Hh, texel,
        1.0 if (t > 1 or doc.tiling) else 0.0,
        1.0 if s["displacement"] else 0.0,
        0.0, hs, 0.0,
        max(1e-6, 0.02 * texel),
    ], np.float64)
    # Panel and wire materials come per texel in Mt; mat keeps the floor around the panel.
    mat = np.array([0, 0, 0, 0, 0, 0.03, 0.03, 0.035, 0.6], np.float64)
    cam = _camera(s["camera"], S * (t if t > 1 else 1), s["width"] / s["height"])
    lights, gradient = _lights(s["light"], S)
    sl = s["scene_light"]                       # dims spotlights and HDRI together
    lights[:, 6:9] *= sl
    e = s["environment"]
    em = ENV_CACHE.envmap(e["hdri"], gradient.reshape(3, 3).tolist())
    env_k = e["intensity"] * sl
    ep = np.array([env_k, math.radians(e["rotation"]), BACKGROUNDS[e["background"]],
                   1.0 if (em.enabled and env_k > 0) else 0.0], np.float64)
    Em, EPD, ECDF, EIDX = emitter_tables(res.maps["emissive"], W, Hh)
    emp = np.array([1.0 if res.info.get("has_emission") else 0.0, float(t), 1.0], np.float64)
    opt = np.array([s["bounces"], float(s["clamp"]), 0.02], np.float64)
    return {
        "settings": s, "Hm": Hm, "Nm": Nm, "Mt": Mt, "sc": sc, "cam": cam,
        "mat": mat, "lights": lights, "opt": opt,
        "E": em.img, "EB": em.blur, "P": em.pdf, "M": em.marg, "C": em.cond, "ep": ep,
        "Em": Em, "EPD": EPD, "ECDF": ECDF, "EIDX": EIDX, "emp": emp,
        "bake_ms": res.info["ms"],
    }


def tonemap(accum, spp, exposure=0.0, bloom=None):
    """Optional bloom, then ACES-fitted tonemap and sRGB encoding. Returns uint8 RGB."""
    x = accum / max(spp, 1) * (2.0 ** float(exposure))
    x = np.maximum(x, 0.0)
    if bloom and bloom.get("pathtrace"):
        x = apply_bloom(x, bloom["threshold"], bloom["intensity"], bloom["radius"])
    y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    y = np.clip(y, 0.0, 1.0)
    srgb = np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)
    return (srgb * 255.0 + 0.5).astype(np.uint8)
