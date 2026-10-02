"""Manhattan wire routing (A* on a grid over the panel height field).

The cost field comes from a panels-only bake at a fixed resolution, so a route
depends only on the document, never on the preview or export resolution:

  * cells with bevels, edges or raised details cost extra (paths go around)
  * cells lower than their surroundings (seams, grooves) cost less
  * turning costs extra, so paths use few bends
  * cells near earlier routed wires cost extra, so groups run in parallel

Moves are 4-way (right angles) or 8-way when the wire allows 45-degree runs.
Results are cached by everything that affects them.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import threading
from collections import OrderedDict

import numpy as np

MAX_CELLS = 128           # cells along the longer canvas side
MIN_CELL = 0.005          # meters
SUB = 4                   # height samples per cell side

_lock = threading.Lock()
_cost_cache = OrderedDict()
_route_cache = OrderedDict()


def _lru_put(cache, key, value, cap):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > cap:
        cache.popitem(last=False)


def grid_for(doc):
    cell = max(MIN_CELL, max(doc.canvas_w, doc.canvas_h) / MAX_CELLS)
    nx = max(2, int(math.ceil(doc.canvas_w / cell)))
    ny = max(2, int(math.ceil(doc.canvas_h / cell)))
    return cell, nx, ny


def _panel_key(doc):
    return json.dumps({"p": [p.to_dict() for p in doc.panels], "w": doc.canvas_w,
                       "h": doc.canvas_h, "t": doc.tiling}, sort_keys=True)


def cost_field(doc):
    """(cost [ny, nx], cell size, panel key). Cached by panel geometry."""
    key = _panel_key(doc)
    with _lock:
        if key in _cost_cache:
            _cost_cache.move_to_end(key)
            return _cost_cache[key]
    from generators.base import Grid
    from generators.panels import PanelGenerator
    from .document import Document
    cell, nx, ny = grid_for(doc)
    panels_only = Document(canvas_w=nx * cell, canvas_h=ny * cell, texel_density=doc.texel_density,
                           tiling=doc.tiling, panels=doc.panels)
    H = PanelGenerator().height(panels_only, Grid(nx * cell, ny * cell, nx * SUB, ny * SUB, doc.tiling))[0]
    blocks = H.reshape(ny, SUB, nx, SUB)
    hmax, hmin, hmean = blocks.max((1, 3)), blocks.min((1, 3)), blocks.mean((1, 3))
    mode = "wrap" if doc.tiling else "edge"
    P = np.pad(hmean, 2, mode=mode)
    around = sum(P[j:j + ny, i:i + nx] for j in range(5) for i in range(5)) / 25.0
    cost = np.ones((ny, nx))
    cost += 6.0 * np.clip((hmax - hmin) / 0.003, 0.0, 1.0)            # edges, bevels, details
    cost += 4.0 * np.clip((hmean - around - 0.0015) / 0.003, 0.0, 1.0)  # raised bumps
    cost = np.where(around - hmean > 0.001, cost * 0.5, cost)          # seams and grooves
    out = (cost, cell, key)
    with _lock:
        _lru_put(_cost_cache, key, out, 8)
    return out


DIRS4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIRS8 = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]


def astar(cost, start, goal, allow45=False, occupancy=None, wrap=False, turn=2.5):
    """Lowest-cost path from start to goal cell (x, y). Returns the list of
    unwrapped integer steps' cells, start and goal included."""
    ny, nx = cost.shape
    dirs = DIRS8 if allow45 else DIRS4
    nd = len(dirs)
    c = cost if occupancy is None else cost + occupancy
    cflat = c.ravel().tolist()
    cmin = float(c.min())
    lens = [math.hypot(dx, dy) for dx, dy in dirs]
    # Turn penalty by the change in direction, in units of 45 degrees.
    step45 = 8 // nd
    tpen = []
    for k in range(nd):
        steps = min(k, nd - k) * step45
        tpen.append([0.0, turn * 0.35, turn, turn * 2.5, math.inf][steps])
    gx, gy = goal
    sx, sy = start

    def h(x, y):
        dx, dy = abs(x - gx), abs(y - gy)
        if wrap:
            dx, dy = min(dx, nx - dx), min(dy, ny - dy)
        if allow45:
            return cmin * (max(dx, dy) + (math.sqrt(2) - 1) * min(dx, dy))
        return cmin * (dx + dy)

    N = nx * ny * nd
    best = [math.inf] * N
    parent = [-1] * N
    heap = []
    for d in range(nd):
        s = (sy * nx + sx) * nd + d
        best[s] = 0.0
        heapq.heappush(heap, (h(sx, sy), 0.0, s))
    goal_cell = gy * nx + gx
    end = -1
    while heap:
        f, g, s = heapq.heappop(heap)
        if g > best[s]:
            continue
        cellid, d = divmod(s, nd)
        if cellid == goal_cell:
            end = s
            break
        y, x = divmod(cellid, nx)
        for k in range(nd):
            tp = tpen[(k - d) % nd]
            if tp == math.inf:
                continue
            dx, dy = dirs[k]
            x2, y2 = x + dx, y + dy
            if wrap:
                x2 %= nx
                y2 %= ny
            elif not (0 <= x2 < nx and 0 <= y2 < ny):
                continue
            c2 = y2 * nx + x2
            g2 = g + cflat[c2] * lens[k] + tp
            s2 = c2 * nd + k
            if g2 < best[s2]:
                best[s2] = g2
                parent[s2] = s
                heapq.heappush(heap, (g2 + h(x2, y2), g2, s2))
    if end < 0:
        return None
    states = []
    s = end
    while s >= 0:
        states.append(s)
        s = parent[s]
    states.reverse()
    # Rebuild unwrapped coordinates by accumulating the steps taken.
    cells = [(sx, sy)]
    for s in states[1:]:
        dx, dy = dirs[s % nd]
        px, py = cells[-1]
        cells.append((px + dx, py + dy))
    return cells


def _simplify(P):
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = np.asarray(out[-1]), np.asarray(P[i]), np.asarray(P[i + 1])
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > 1e-12:
            out.append(P[i])
    out.append(P[-1])
    return [list(map(float, p)) for p in out]


def round_corners(P, radius, steps_per_90=8):
    """Replace each corner with a circular arc of the given radius (clamped so
    neighbouring arcs never overlap)."""
    P = [np.asarray(p, np.float64) for p in P]
    if radius <= 0 or len(P) < 3:
        return [list(p) for p in P]
    out = [P[0]]
    for i in range(1, len(P) - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        u = a - b
        v = c - b
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-12 or lv < 1e-12:
            continue
        u /= lu
        v /= lv
        theta = math.acos(max(-1.0, min(1.0, float(u @ v))))    # interior angle
        if theta > math.pi - 1e-6:
            out.append(b)
            continue
        tlen = radius / math.tan(theta / 2)
        tlen = min(tlen, lu * 0.5, lv * 0.5)
        r = tlen * math.tan(theta / 2)
        p0, p1 = b + u * tlen, b + v * tlen
        bis = u + v
        bis /= np.linalg.norm(bis)
        center = b + bis * math.hypot(tlen, r)
        a0 = math.atan2(*(p0 - center)[::-1])
        a1 = math.atan2(*(p1 - center)[::-1])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        n = max(2, int(math.ceil(abs(da) / (math.pi / 2) * steps_per_90)))
        for k in range(n + 1):
            ang = a0 + da * k / n
            out.append(center + r * np.array([math.cos(ang), math.sin(ang)]))
    out.append(P[-1])
    return [list(map(float, p)) for p in out]


def _occupancy_add(occ, P, cell, radius_cells, wrap):
    ny, nx = occ.shape
    P = np.asarray(P, np.float64)
    seg = np.hypot(*np.diff(P, axis=0).T)
    pts = [P[0]]
    for i, L in enumerate(seg):
        n = max(1, int(L / (cell * 0.5)))
        for k in range(1, n + 1):
            pts.append(P[i] + (P[i + 1] - P[i]) * k / n)
    R = int(math.ceil(radius_cells))
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
    disk = np.hypot(xx, yy)
    pen_core = np.where(disk <= max(radius_cells - 1, 0.5), 12.0, 0.0)
    pen_ring = np.where(disk <= radius_cells, 6.0, 0.0)
    pen = np.maximum(pen_core, pen_ring)
    for p in pts:
        cx, cy = int(p[0] // cell), int(p[1] // cell)
        ys, xs = cy + yy, cx + xx
        if wrap:
            ys %= ny
            xs %= nx
            np.maximum.at(occ, (ys, xs), pen)
        else:
            m = (ys >= 0) & (ys < ny) & (xs >= 0) & (xs < nx)
            np.maximum.at(occ, (ys[m], xs[m]), pen[m])


def route_wire(doc, wire, cost, cell, occupancy=None):
    """Polyline (meters) for one routed wire, from its resolved endpoints."""
    ny, nx = cost.shape
    a = np.array(wire.a.resolve(doc), np.float64)
    b = np.array(wire.b.resolve(doc), np.float64)
    W, H = nx * cell, ny * cell
    wrap = doc.tiling

    def to_cell(p):
        q = np.array([p[0] % W, p[1] % H]) if wrap else p
        return (int(np.clip(q[0] // cell, 0, nx - 1)), int(np.clip(q[1] // cell, 0, ny - 1)))

    sa, sb = to_cell(a), to_cell(b)
    cells = astar(cost, sa, sb, wire.allow45, occupancy, wrap) if sa != sb else [sa, sb]
    if cells is None:
        cells = [sa, sb]
    # Cells are unwrapped from the start cell; place them next to the real start.
    base = np.array([a[0] - (a[0] % W if wrap else a[0]) , a[1] - (a[1] % H if wrap else a[1])])
    centers = [base + (np.array(c) + 0.5) * cell for c in cells]
    end = centers[-1]
    # The goal may be a wrapped copy of b: pick b's copy nearest the path end.
    if wrap:
        b = b + np.array([round((end[0] - b[0]) / W) * W, round((end[1] - b[1]) / H) * H])
    pts = [a]
    if len(centers) >= 2:
        d0 = centers[1] - centers[0]
        # Join the exact endpoint to the cell lane with an axis-aligned jog.
        if abs(d0[0]) > 1e-12 and abs(d0[1]) < 1e-12:
            pts.append(np.array([a[0], centers[0][1]]))
        elif abs(d0[1]) > 1e-12 and abs(d0[0]) < 1e-12:
            pts.append(np.array([centers[0][0], a[1]]))
        pts.extend(centers)
        d1 = centers[-1] - centers[-2]
        if abs(d1[0]) > 1e-12 and abs(d1[1]) < 1e-12:
            pts[-1] = np.array([b[0], centers[-1][1]])
            pts.append(np.array([b[0], b[1]]))
        elif abs(d1[1]) > 1e-12 and abs(d1[0]) < 1e-12:
            pts[-1] = np.array([centers[-1][0], b[1]])
            pts.append(np.array([b[0], b[1]]))
        else:
            pts.append(b)
    else:
        pts.append(b)
    clean = [pts[0]]
    for p in pts[1:]:
        if np.hypot(*(p - clean[-1])) > 1e-9:
            clean.append(p)
    if len(clean) < 2:
        clean.append(b)
    simple = _simplify(clean)
    return round_corners(simple, wire.corner_radius)


def _wire_key(doc, w):
    return json.dumps({"a": w.a.resolve(doc), "b": w.b.resolve(doc), "45": w.allow45,
                       "cr": w.corner_radius, "cl": w.clearance, "hw": w.total_half_width()})


def route_all(doc):
    """Fill `points` of every visible routed wire (in document order, each
    keeping clear of the ones before it)."""
    routed = [w for w in doc.wires if w.mode == "route" and w.visible]
    if not routed:
        return
    cost, cell, pkey = cost_field(doc)
    occ = np.zeros_like(cost)
    chain = hashlib.sha1(pkey.encode()).hexdigest()
    for w in routed:
        chain = hashlib.sha1((chain + _wire_key(doc, w)).encode()).hexdigest()
        key = chain
        with _lock:
            hit = _route_cache.get(key)
        if hit is None:
            hit = route_wire(doc, w, cost, cell, occ if occ.any() else None)
            with _lock:
                _lru_put(_route_cache, key, hit, 512)
        w.points = [list(p) for p in hit]
        radius_cells = (w.total_half_width() * 2 + w.clearance) / cell + 0.5
        _occupancy_add(occ, w.points, cell, radius_cells, doc.tiling)
        chain = hashlib.sha1((chain + json.dumps(w.points)).encode()).hexdigest()
