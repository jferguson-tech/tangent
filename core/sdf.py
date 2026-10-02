"""Vectorized 2D signed distance functions (NumPy).

Convention: negative inside, positive outside, units are whatever the inputs
are (meters throughout this project). Image space is y-down, so corner order
is [top-left, top-right, bottom-right, bottom-left].
"""
import numpy as np

SQRT1_2 = 0.7071067811865476


def sd_box(px, py, hx, hy):
    qx = np.abs(px) - hx
    qy = np.abs(py) - hy
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside


def _pick_corner(px, py, corners):
    tl, tr, br, bl = corners
    right = px > 0
    bottom = py > 0
    return np.where(right, np.where(bottom, br, tr), np.where(bottom, bl, tl))


def sd_round_box(px, py, hx, hy, corners):
    """Rounded box with a separate radius per corner (exact distance)."""
    lim = min(hx, hy)
    cs = [min(max(c, 0.0), lim) for c in corners]
    r = _pick_corner(px, py, cs)
    qx = np.abs(px) - hx + r
    qy = np.abs(py) - hy + r
    return (np.minimum(np.maximum(qx, qy), 0.0)
            + np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0)) - r)


def sd_chamfer_box(px, py, hx, hy, corners):
    """Box with a 45-degree chamfer per corner.

    The chamfer leg length is measured along each edge. The result is exact
    inside the shape, which is where bevels are evaluated.
    """
    lim = min(hx, hy)
    cs = [min(max(c, 0.0), lim) for c in corners]
    c = _pick_corner(px, py, cs)
    d_box = sd_box(px, py, hx, hy)
    d_cut = (np.abs(px) + np.abs(py) - (hx + hy - c)) * SQRT1_2
    return np.maximum(d_box, d_cut)


def sd_circle(px, py, r):
    return np.hypot(px, py) - r


def sd_hexagon(px, py, r):
    """Regular hexagon, r is the inradius (flat-to-center distance)."""
    kx, ky, kz = -0.866025404, 0.5, 0.577350269
    ax = np.abs(px)
    ay = np.abs(py)
    dot = np.minimum(kx * ax + ky * ay, 0.0)
    ax = ax - 2.0 * dot * kx
    ay = ay - 2.0 * dot * ky
    ax = ax - np.clip(ax, -kz * r, kz * r)
    ay = ay - r
    return np.hypot(ax, ay) * np.sign(ay)


def rotate(px, py, degrees):
    """Rotate sample coordinates into a shape's local frame."""
    if not degrees:
        return px, py
    a = np.deg2rad(-degrees)
    c, s = np.cos(a), np.sin(a)
    return c * px - s * py, s * px + c * py
