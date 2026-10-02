"""Bevel profile curves.

A profile maps t in [0, 1] to a height fraction in [0, 1]. t = 0 is the
outer edge of the shape and t = 1 is where the bevel meets the flat top.
t is distance-from-edge divided by bevel width, so the bevel keeps its world
width no matter how large the panel is.
"""
import numpy as np


def _linear(t, pts):
    return t


def _round(t, pts):  # convex quarter circle
    return np.sqrt(np.clip(1.0 - (1.0 - t) ** 2, 0.0, 1.0))


def _cove(t, pts):  # concave quarter circle
    return 1.0 - np.sqrt(np.clip(1.0 - t * t, 0.0, 1.0))


def _smooth(t, pts):
    return t * t * (3.0 - 2.0 * t)


def _ogee(t, pts):  # cove into round, an S-curve
    lo = 0.5 * _cove(np.clip(2.0 * t, 0, 1), pts)
    hi = 0.5 + 0.5 * _round(np.clip(2.0 * t - 1.0, 0, 1), pts)
    return np.where(t < 0.5, lo, hi)


def _stepped(t, pts):  # bevel, ledge, bevel
    a, b, s = 0.35, 0.65, 0.5
    first = s * np.clip(t / a, 0, 1)
    second = s + (1 - s) * np.clip((t - b) / (1 - b), 0, 1)
    return np.where(t < b, first, second)


def _custom(t, pts):
    if not pts or len(pts) < 2:
        return t
    p = sorted((float(x), float(y)) for x, y in pts)
    xs = np.clip([q[0] for q in p], 0, 1)
    ys = np.clip([q[1] for q in p], 0, 1)
    return np.interp(t, xs, ys, left=ys[0], right=ys[-1])


PROFILES = {
    "linear": _linear,
    "round": _round,
    "cove": _cove,
    "smooth": _smooth,
    "ogee": _ogee,
    "stepped": _stepped,
    "custom": _custom,
}


def evaluate(name, t, points=None):
    fn = PROFILES.get(name, _linear)
    return fn(np.clip(t, 0.0, 1.0), points)


def sample(name, n=48, points=None):
    """Sampled curve for UI thumbnails, so the browser never re-implements it."""
    t = np.linspace(0.0, 1.0, n)
    return [round(float(v), 4) for v in evaluate(name, t, points)]
