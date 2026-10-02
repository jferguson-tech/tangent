"""Auto-layout: recursive guillotine subdivision into sci-fi panels.

Everything is in meters. The output is ordinary Panel objects, so generated
panels are fully editable. Locked panels survive a re-roll. Generated panels
that would overlap a locked one are dropped.
"""
from __future__ import annotations

import copy
import math
import random

from .panel import Panel, Bevel, Groove, Detail, Emission, new_id
from .units import MIN_BEVEL_PX

DEFAULT_PARAMS = {
    "seed": 1,
    "min_size": 0.18,        # smallest panel side (m)
    "max_size": 0.9,         # panels larger than this are always split (m)
    "max_depth": 6,
    "stop_chance": 0.3,      # chance to stop splitting early
    "seam": 0.008,           # gap between panels (m)
    "border": 0.02,          # margin at the canvas edge (m), ignored when tiling
    "grid": 0.01,            # snap cuts to this grid (m), 0 = off
    "symmetry": "none",      # none | x | y | xy
    "wrap_offset": True,     # tiling only: shift the layout so panels cross edges
    "depth": 0.012,
    "depth_variation": 0.35,
    "bevel_width": 0.008,
    "profile": "linear",
    "corner_style": "chamfer",
    "corner": 0.03,
    "inset_chance": 0.15,
    "nest_chance": 0.35,
    "groove_chance": 0.2,
    "detail_density": 0.5,
    "emissive_density": 0.25,  # light strips, indicator dots, glowing grooves, light panels
    "light_palette": "mixed",  # mixed | cyan | amber | white | red
}

LIGHT_COLORS = {
    "cyan": [0.35, 0.85, 1.0],
    "amber": [1.0, 0.62, 0.18],
    "white": [0.95, 0.97, 1.0],
    "red": [1.0, 0.18, 0.12],
}


_CHOICES = {
    "symmetry": ("none", "x", "y", "xy"),
    "corner_style": ("chamfer", "round"),
    "light_palette": ("mixed", "cyan", "amber", "white", "red"),
}


def params_from(d):
    from .profiles import PROFILES
    p = dict(DEFAULT_PARAMS)
    for k, v in (d or {}).items():
        if k not in p:
            continue
        default = DEFAULT_PARAMS[k]
        try:
            if isinstance(default, bool):
                p[k] = bool(v)
            elif isinstance(default, int):
                p[k] = int(float(v))
            elif isinstance(default, float):
                p[k] = float(v)
            else:
                options = PROFILES if k == "profile" else _CHOICES.get(k, ())
                p[k] = v if v in options else default
        except (TypeError, ValueError):
            p[k] = default
    p["min_size"] = max(0.02, p["min_size"])
    p["max_size"] = max(p["min_size"] * 2, p["max_size"])
    p["max_depth"] = max(0, min(12, p["max_depth"]))
    p["seam"] = max(0.0, p["seam"])
    return p


def _snap(v, origin, grid):
    if grid <= 0:
        return v
    return origin + round((v - origin) / grid) * grid


def _split(rect, depth, p, rng, out):
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    mn = p["min_size"]
    can_x, can_y = w >= 2 * mn, h >= 2 * mn
    force = w > p["max_size"] or h > p["max_size"]
    stop = (depth >= p["max_depth"] or not (can_x or can_y)
            or (not force and depth > 0 and rng.random() < p["stop_chance"]))
    if stop:
        out.append(rect)
        return
    if can_x and can_y:
        axis = "x" if rng.random() < (w * w) / (w * w + h * h) else "y"
    else:
        axis = "x" if can_x else "y"
    lo, hi = (x0, x1) if axis == "x" else (y0, y1)
    size = hi - lo
    for _ in range(6):
        if rng.random() < 0.6:
            f = rng.choice([0.5, 1 / 3, 2 / 3, 0.25, 0.75])
        else:
            f = rng.uniform(0.2, 0.8)
        cut = _snap(lo + size * f, lo, p["grid"])
        if cut - lo >= mn - 1e-9 and hi - cut >= mn - 1e-9:
            break
    else:
        out.append(rect)
        return
    if axis == "x":
        _split((x0, y0, cut, y1), depth + 1, p, rng, out)
        _split((cut, y0, x1, y1), depth + 1, p, rng, out)
    else:
        _split((x0, y0, x1, cut), depth + 1, p, rng, out)
        _split((x0, cut, x1, y1), depth + 1, p, rng, out)


def _details(panel, p, rng):
    dens = p["detail_density"]
    mb = p["_min_bevel"]          # generated bevels stay above the alias limit
    w, h = panel.w, panel.h
    small = min(w, h)
    out = []
    if small > 0.14 and rng.random() < 0.65 * dens:
        size = 0.014 if small < 0.3 else 0.018
        inset = panel.bevel.width + size * 0.5 + 0.01
        shape = "hex" if rng.random() < 0.35 else "circle"
        out.append(Detail(kind="bolt", shape=shape, anchor="corners", ox=inset, oy=inset,
                          w=size, h=size, depth=0.003,
                          bevel=Bevel(width=max(size * 0.35, mb), profile="round")))
    if small > 0.2 and rng.random() < 0.35 * dens:
        vertical = h > w
        n = rng.randint(3, 7)
        slot_l = min(0.35 * max(w, h), 0.16)
        slot_w = 0.012
        spacing = slot_w * 2.2
        across = n * spacing
        ext_x, ext_y = (across, slot_l) if vertical else (slot_l, across)
        anchor = rng.choice(["c", "b", "t"] if vertical else ["c", "r", "l"])
        out.append(Detail(kind="vent", shape="rect", anchor=anchor,
                          ox=ext_x * 0.5 + 0.05 if anchor in ("l", "r") else 0.0,
                          oy=ext_y * 0.5 + 0.05 if anchor in ("t", "b") else 0.0,
                          w=slot_l, h=slot_w, radius=slot_w * 0.5,
                          rotation=90.0 if vertical else 0.0,
                          count=n, spacing=spacing, depth=0.004, mode="inset",
                          bevel=Bevel(width=max(0.003, mb), profile="linear")))
    if small > 0.16 and rng.random() < 0.25 * dens:
        cw, ch = min(0.08, w * 0.3), min(0.05, h * 0.25)
        out.append(Detail(kind="cutout", shape="rect", anchor=rng.choice(["tl", "tr", "bl", "br"]),
                          ox=cw * 0.5 + 0.035, oy=ch * 0.5 + 0.035, w=cw, h=ch, radius=0.006,
                          depth=0.004, mode="inset", bevel=Bevel(width=max(0.004, mb), profile="linear")))
    if max(w, h) > 0.35 and rng.random() < 0.2 * dens:
        horizontal = w >= h
        length = (w if horizontal else h) * 0.6
        out.append(Detail(kind="groove", shape="rect", anchor=rng.choice(["t", "b"] if horizontal else ["l", "r"]),
                          ox=0.0 if horizontal else 0.045, oy=0.045 if horizontal else 0.0,
                          w=length if horizontal else 0.006, h=0.006 if horizontal else length,
                          radius=0.003, depth=0.002, mode="inset",
                          bevel=Bevel(width=max(0.002, mb), profile="round")))
    return out


def _make_panel(rect, p, rng, n):
    x0, y0, x1, y1 = rect
    s = p["seam"] * 0.5
    x0, y0, x1, y1 = x0 + s, y0 + s, x1 - s, y1 - s
    w, h = x1 - x0, y1 - y0
    small = min(w, h)
    c = min(p["corner"], small * 0.3)
    corners = [c if rng.random() < 0.55 else min(0.004, c) for _ in range(4)]
    var = p["depth_variation"]
    depth = p["depth"] * (1.0 + rng.uniform(-var, var))
    inset = rng.random() < p["inset_chance"]
    panel = Panel(id=new_id(), name=f"Panel {n}", x=x0, y=y0, w=w, h=h,
                  corner_style=p["corner_style"], corners=corners,
                  mode="inset" if inset else "raise",
                  depth=depth * (0.5 if inset else 1.0),
                  bevel=Bevel(width=p["bevel_width"], profile=p["profile"]))
    if rng.random() < p["groove_chance"] and small > 0.15:
        panel.groove = Groove(offset=p["bevel_width"] + 0.015, width=max(0.005, 2 * p["_min_bevel"]),
                              depth=0.0025)
    result = [panel]
    if not inset and small > 0.3 and rng.random() < p["nest_chance"]:
        m = max(0.035, min(0.12, small * 0.18))
        sub_inset = rng.random() < 0.6
        child = Panel(id=new_id(), name=f"Panel {n} inner", x=x0 + m, y=y0 + m,
                      w=w - 2 * m, h=h - 2 * m, corner_style=p["corner_style"],
                      corners=[max(0.0, cc - m * 0.3) for cc in corners],
                      mode="inset" if sub_inset else "raise",
                      depth=depth * 0.35,
                      bevel=Bevel(width=max(0.003, p["bevel_width"] * 0.6, p["_min_bevel"]),
                                  profile=p["profile"]))
        child.details = _details(child, p, rng)
        result.append(child)
    else:
        panel.details = _details(panel, p, rng)
    return result


_MIRROR_X = {"tl": "tr", "tr": "tl", "l": "r", "r": "l", "bl": "br", "br": "bl"}
_MIRROR_Y = {"tl": "bl", "bl": "tl", "t": "b", "b": "t", "tr": "br", "br": "tr"}


def _mirror(panel, W, H, axis):
    q = copy.deepcopy(panel)
    q.id = new_id()
    q.name = f"{panel.name} {'X' if axis == 'x' else 'Y'}-mirror"
    tl, tr, br, bl = q.corners
    if axis == "x":
        q.x = W - (q.x + q.w)
        q.corners = [tr, tl, bl, br]
        table = _MIRROR_X
    else:
        q.y = H - (q.y + q.h)
        q.corners = [bl, br, tr, tl]
        table = _MIRROR_Y
    for d in q.details:
        d.anchor = table.get(d.anchor, d.anchor)
        if axis == "x" and d.anchor in ("c", "t", "b", "top-bottom"):
            d.ox = -d.ox
        if axis == "y" and d.anchor in ("c", "l", "r", "left-right"):
            d.oy = -d.oy
        d.rotation = -d.rotation
    return q


def _light_color(p, rng, indicator=False):
    pal = p["light_palette"]
    if pal != "mixed":
        return list(LIGHT_COLORS[pal])
    if indicator:
        return list(LIGHT_COLORS[rng.choice(["amber", "red", "cyan"])])
    r = rng.random()
    return list(LIGHT_COLORS["cyan" if r < 0.5 else "white" if r < 0.85 else "amber"])


def _add_lights(panels, p, rng):
    """Sprinkle emitters over generated panels. Uses its own RNG so the panel
    geometry for a seed does not change with the emissive settings."""
    dens = p["emissive_density"]
    if dens <= 0:
        return
    mb = p["_min_bevel"]
    for q in panels:
        small, big = min(q.w, q.h), max(q.w, q.h)
        used = {d.anchor for d in q.details}
        # Glowing groove.
        if q.groove.enabled and rng.random() < min(1.0, 1.2 * dens):
            q.groove.emission = Emission(True, _light_color(p, rng), 3.0)
        # Light panel: an inset child panel whose face glows through a diffuser.
        if q.mode == "inset" and q.name.endswith("inner") and rng.random() < 0.6 * dens:
            q.emission = Emission(True, _light_color(p, rng), 1.6, diffuser=0.6)
            continue
        # Light strip along the long axis, on an edge without other details.
        if q.mode == "raise" and small > 0.16 and big > 0.3 and rng.random() < 0.7 * dens:
            horizontal = q.w >= q.h
            sides = [s for s in (("b", "t") if horizontal else ("r", "l")) if s not in used]
            if sides:
                side = sides[0]
                length = min(0.55 * big, 0.45)
                inset = q.bevel.width + 0.03
                q.details.append(Detail(
                    kind="light strip", shape="rect", anchor=side,
                    ox=0.0 if horizontal else inset, oy=inset if horizontal else 0.0,
                    w=length if horizontal else 0.012, h=0.012 if horizontal else length,
                    radius=0.006, depth=0.002, mode="raise",
                    bevel=Bevel(width=max(0.004, mb), profile="smooth"),
                    emission=Emission(True, _light_color(p, rng), 4.0)))
                used.add(side)
        # Indicator dots: a short row of small lights near a top corner.
        if small > 0.14 and rng.random() < 0.6 * dens:
            corner = next((c for c in ("tr", "tl") if c not in used), None)
            if corner:
                q.details.append(Detail(
                    kind="indicator", shape="circle", anchor=corner,
                    ox=q.bevel.width + 0.06, oy=q.bevel.width + 0.022,
                    w=0.008, h=0.008, rotation=90.0, count=rng.choice([2, 3]), spacing=0.016,
                    depth=0.0015, mode="raise", bevel=Bevel(width=max(0.003, mb), profile="round"),
                    emission=Emission(True, _light_color(p, rng, indicator=True), 6.0)))


def _overlaps(a, b, gap):
    return not (a.x + a.w + gap <= b.x or b.x + b.w + gap <= a.x
                or a.y + a.h + gap <= b.y or b.y + b.h + gap <= a.y)


def generate(doc, params=None):
    """Return a new panel list: locked panels kept, the rest regenerated."""
    p = params_from(params)
    p["_min_bevel"] = MIN_BEVEL_PX * 1.25 / doc.texel_density
    rng = random.Random(int(p["seed"]))
    W, H = doc.canvas_w, doc.canvas_h
    tiling = doc.tiling
    b = 0.0 if tiling else p["border"]
    sym = p["symmetry"]
    rx1 = W * 0.5 if sym in ("x", "xy") else W - b
    ry1 = H * 0.5 if sym in ("y", "xy") else H - b
    # A symmetric half should mirror cleanly, so half-edges get half a seam.
    rects = []
    _split((b, b, rx1, ry1), 0, p, rng, rects)
    locked = [q for q in doc.panels if q.locked]
    panels = []
    for i, r in enumerate(rects):
        panels.extend(_make_panel(r, p, rng, len(locked) + i + 1))
    _add_lights(panels, p, random.Random(int(p["seed"]) * 7919 + 13))
    if sym in ("x", "xy"):
        panels += [_mirror(q, W, H, "x") for q in panels]
    if sym in ("y", "xy"):
        panels += [_mirror(q, W, H, "y") for q in panels]
    if tiling and p["wrap_offset"] and sym == "none":
        ox = _snap(rng.uniform(0.1, 0.9) * W, 0.0, p["grid"])
        oy = _snap(rng.uniform(0.1, 0.9) * H, 0.0, p["grid"])
        for q in panels:
            q.x = (q.x + ox) % W
            q.y = (q.y + oy) % H
    if locked:
        panels = [q for q in panels
                  if not any(_overlaps(q, L, p["seam"] * 0.5) for L in locked)]
    return locked + panels


# ---- wires ----------------------------------------------------------------------
WIRE_PARAMS = {
    "wire_density": 0.12,    # wires per generated panel
    "wire_sim_share": 0.5,   # share of simulated (vs routed) wires
    "harness_chance": 0.25,  # share of multi-strand harnesses
    "wire_seed": 0,          # 0 = derive from the layout seed; set by "Shuffle wires"
}
DEFAULT_PARAMS.update(WIRE_PARAMS)


MAX_WIRES_PER_PANEL = 4
MAX_WIRES_PER_PAIR = 2


def _wire_end(panel, rng, toward):
    """An endpoint on a panel edge facing `toward`, offset inward past the bevel."""
    from .wire import End
    cx, cy = panel.x + panel.w / 2, panel.y + panel.h / 2
    dx, dy = toward[0] - cx, toward[1] - cy
    inset = panel.bevel.width + 0.035
    if abs(dx) * panel.h > abs(dy) * panel.w:
        anchor = "r" if dx > 0 else "l"
        lat = rng.uniform(-0.3, 0.3) * panel.h
        return End(panel=panel.id, anchor=anchor, dx=-inset if dx > 0 else inset, dy=lat)
    anchor = "b" if dy > 0 else "t"
    lat = rng.uniform(-0.3, 0.3) * panel.w
    return End(panel=panel.id, anchor=anchor, dx=lat, dy=-inset if dy > 0 else inset)


def generate_wires(doc, panels, params=None, keep=()):
    """New wires connecting generated panels, plus `keep` (locked wires)."""
    from .wire import Wire, sag_curve, new_wire_id
    p = params_from(params)
    wseed = int(p["wire_seed"]) or int(p["seed"])
    rng = random.Random(wseed * 104729 + 7)
    # Any panel big enough for a connector can take a wire end, raised or inset.
    cands = [q for q in panels if q.visible and min(q.w, q.h) > 0.12]
    n = int(round(p["wire_density"] * len(panels)))
    wires = list(keep)
    if len(cands) < 2 or n <= 0:
        return wires
    S = max(doc.canvas_w, doc.canvas_h)
    center = lambda q: (q.x + q.w / 2, q.y + q.h / 2)
    # Spread the wires out: a panel takes at most MAX_PER_PANEL ends and a pair
    # of panels at most MAX_PER_PAIR wires; less-used panels are preferred. When
    # no allowed pair is left, fewer wires are made instead of piling them up.
    use = {q.id: 0 for q in cands}
    pairs = {}
    for w in keep:
        for e in (w.a, w.b):
            if e.panel in use:
                use[e.panel] += 1
        if w.a.panel and w.b.panel:
            k = frozenset((w.a.panel, w.b.panel))
            pairs[k] = pairs.get(k, 0) + 1
    weight = lambda q: 1.0 / (1 + use[q.id]) ** 2

    def partners(a):
        ca = center(a)
        return [q for q in cands if q is not a and use[q.id] < MAX_WIRES_PER_PANEL
                and pairs.get(frozenset((a.id, q.id)), 0) < MAX_WIRES_PER_PAIR
                and 0.15 * S < math.hypot(center(q)[0] - ca[0], center(q)[1] - ca[1]) < 0.65 * S]

    for i in range(n):
        starts = [q for q in cands if use[q.id] < MAX_WIRES_PER_PANEL and partners(q)]
        if not starts:
            break
        a = rng.choices(starts, weights=[weight(q) for q in starts])[0]
        far = partners(a)
        b = rng.choices(far, weights=[weight(q) for q in far])[0]
        use[a.id] += 1
        use[b.id] += 1
        k = frozenset((a.id, b.id))
        pairs[k] = pairs.get(k, 0) + 1
        ca, cb = center(a), center(b)
        sim = rng.random() < p["wire_sim_share"]
        r = rng.random()
        profile = "hose" if r < 0.2 else "ribbon" if r < 0.32 else "round"
        radius = {"hose": rng.choice([0.007, 0.009, 0.012]),
                  "ribbon": rng.choice([0.0025, 0.003]),
                  "round": rng.choice([0.003, 0.004, 0.005])}[profile]
        harness = profile == "round" and rng.random() < p["harness_chance"]
        w = Wire(id=new_wire_id(), name=f"Wire {len(wires) + 1}", mode="sim" if sim else "route",
                 a=_wire_end(a, rng, cb), b=_wire_end(b, rng, ca), profile=profile, radius=radius,
                 slack=rng.uniform(1.06, 1.3), corner_radius=rng.choice([0.02, 0.03, 0.04]),
                 connectors=rng.random() < 0.8,
                 clip_spacing=rng.choice([0.12, 0.18]) if (not sim and rng.random() < 0.6) else 0.0,
                 bundle=rng.choice([2, 3, 4]) if harness else 1)
        if p["emissive_density"] > 0 and profile == "round" and not harness and rng.random() < 0.12:
            w.emission = Emission(True, _light_color(p, rng), 3.0)
        wires.append(w)
    return wires


def place_new_sim_wires(doc):
    """Give simulated wires without points a sagging starting shape."""
    from .wire import sag_curve
    for w in doc.wires:
        if w.mode == "sim" and len(w.points) < 2:
            pts = sag_curve(w.a.resolve(doc), w.b.resolve(doc), w.slack, doc.wire_sim["gravity_angle"])
            w.points = [[float(x), float(y)] for x, y in pts]
            w.settled = False
