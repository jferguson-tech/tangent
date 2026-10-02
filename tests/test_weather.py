"""Material maps and weathering (core/weather.py)."""
import numpy as np
import pytest

from core import weather
from core.bake import bake, bake_region, DETAIL_MAPS
from core.document import Document
from core.panel import Panel, Bevel
from core.wire import Wire, End


def _doc(age=0.9, gravity=90, tiling=False, paint=False, **w):
    d = Document(canvas_w=0.5, canvas_h=0.5, texel_density=512, tiling=tiling)
    d.panels = [Panel(id="p", x=0.15, y=0.15, w=0.2, h=0.2, depth=0.02, bevel=Bevel(width=0.005)),
                Panel(id="q", x=0.38, y=0.05, w=0.1, h=0.4, depth=0.01, bevel=Bevel(width=0.004))]
    d.wires = [Wire(mode="route", a=End(x=0.05, y=0.45), b=End(x=0.45, y=0.42), radius=0.006)]
    d.weathering.update(enabled=True, age=age, **w)
    d.materials["paint"]["enabled"] = paint
    d.wire_sim["gravity_angle"] = gravity
    return d


MAT = ("basecolor", "roughness", "metallic", "mask_wear", "mask_dirt", "mask_streaks", "mask_rust")


def test_defaults_are_clean_and_sanitized():
    d = Document()
    assert not weather.active(d)
    w = weather.weathering_from({"age": 5, "dirt": -1, "seed": "x", "bogus": 1})
    assert w["age"] == 1 and w["dirt"] == 0 and "bogus" not in w
    m = weather.materials_from({"metal": {"color": [2, -1, 0.5]}})
    assert m["metal"]["color"] == [1, 0, 0.5]
    # Round trip through the document format.
    d2 = Document.from_dict(_doc().to_dict())
    assert d2.weathering["age"] == 0.9 and weather.active(d2)


def test_age_zero_has_no_weathering():
    d = _doc(age=0.0)
    r = bake(d, MAT)
    for m in ("mask_wear", "mask_dirt", "mask_streaks", "mask_rust"):
        assert r.maps[m].max() == 0, m
    # Same height as with weathering disabled.
    d2 = _doc(age=0.0)
    d2.weathering["enabled"] = False
    assert np.array_equal(bake(d, ("height",)).maps["height"], bake(d2, ("height",)).maps["height"])


def test_deterministic_and_seeded():
    a = bake(_doc(), MAT)
    weather._cache.clear()
    b = bake(_doc(), MAT)
    for m in MAT:
        assert np.array_equal(a.maps[m], b.maps[m]), m
    c = bake(_doc(seed=2), MAT)
    assert not np.array_equal(a.maps["mask_rust"], c.maps["mask_rust"])


def test_age_increases_weathering():
    lo, hi = bake(_doc(age=0.3), MAT), bake(_doc(age=0.95), MAT)
    for m in ("mask_wear", "mask_dirt", "mask_streaks", "mask_rust"):
        assert hi.maps[m].mean() > lo.maps[m].mean(), m


def test_edge_wear_on_convex_edges():
    r = bake(_doc(rust=0.0, dirt=0.0), ("mask_wear", "curvature"))
    wear, curv = r.maps["mask_wear"], r.maps["curvature"]
    convex = curv > 0.6                     # 0.5 is flat
    flat = np.abs(curv - 0.5) < 0.02
    assert convex.any() and flat.any()
    assert wear[convex].mean() > 3 * wear[flat].mean()


def test_streaks_follow_gravity():
    def above_below(g):
        s = weather.simulate(_doc(gravity=g, streaks=1.0))["streak"]
        # Columns under panel "p" (x 0.15..0.35 m at 512 px/m): above vs below it.
        return s[20:70, 90:165].mean(), s[185:235, 90:165].mean()
    up, down = above_below(90)              # gravity points down the image
    assert down > 3 * up
    up, down = above_below(270)
    assert up > 3 * down


def test_tiling_is_periodic():
    r = bake(_doc(tiling=True), ("basecolor", "mask_rust", "mask_dirt"))
    for m in ("basecolor", "mask_rust", "mask_dirt"):
        a = r.maps[m]
        # Opposite edges continue each other: the jump across the seam is like any neighbor step.
        seam = np.abs(a[:, 0] - a[:, -1]).mean()
        inner = np.abs(np.diff(a, axis=1)).mean()
        assert seam < 4 * inner + 1e-3, m
        seam = np.abs(a[0] - a[-1]).mean()
        inner = np.abs(np.diff(a, axis=0)).mean()
        assert seam < 4 * inner + 1e-3, m


def test_paint_chips_and_metal():
    r = bake(_doc(paint=True, rust=0.0), ("metallic", "mask_wear", "basecolor"))
    met, wear = r.maps["metallic"], r.maps["mask_wear"]
    assert met[wear > 0.9].mean() > 0.6     # chips show bare metal
    assert met[wear < 0.01].mean() < 0.3    # painted elsewhere


def test_wires_keep_their_material():
    d = _doc()
    d.materials["wire"].update(color=[1, 0, 0], metallic=0.0)
    r = bake(d, ("basecolor", "wires", "mask_rust"))
    on = r.maps["wires"] > 0.99
    assert on.any()
    assert r.maps["mask_rust"][on].max() < 0.02
    assert (r.maps["basecolor"][on][:, 0] > r.maps["basecolor"][on][:, 1]).all()


def test_fast_bake_skips_weathering():
    d = _doc()
    r = bake(d, MAT, weathering=False)
    assert r.maps["mask_rust"].max() == 0
    assert r.maps["basecolor"].shape == (256, 256, 3)


def test_export_sizes_and_orm_channels():
    d = _doc()
    r = bake(d, ("ao", "roughness", "metallic", "orm", "basecolor"))
    o = r.maps["orm"]
    assert o.shape == (256, 256, 3)
    assert np.allclose(o[..., 0], r.maps["ao"])
    assert np.allclose(o[..., 1], r.maps["roughness"])
    assert np.allclose(o[..., 2], r.maps["metallic"])
    assert r.encode("orm", 16).dtype == np.uint16 and r.encode("orm", 16).shape == (256, 256, 3)
    bc = r.encode("basecolor")
    assert bc.dtype == np.uint8 and bc.shape == (256, 256, 3)
    # sRGB encoded: mid linear values come out brighter.
    lin = r.maps["basecolor"]
    mid = (lin > 0.15) & (lin < 0.25)
    assert bc[mid].mean() > 255 * lin[mid].mean()


@pytest.mark.parametrize("tiling,rect", [(False, (0.1, 0.1, 0.3, 0.3)), (True, (0.4, 0.0, 0.5, 0.15))])
def test_detail_matches_full_bake_with_weathering(tiling, rect):
    d = _doc(tiling=tiling, paint=True)
    full = bake(d, DETAIL_MAPS + ("spill",))
    norm = {k: full.info[k] for k in ("height_min", "height_max", "emissive_scale", "curvature_scale")}
    reg = bake_region(d, *rect, 512, DETAIL_MAPS, norm)
    x0, y0 = reg.info["region"][:2]
    i0, j0 = round(x0 * 512), round(y0 * 512)
    h, w = reg.maps["height"].shape
    sl = (slice(j0, j0 + h), slice(i0, i0 + w))
    for m in ("height", "normal", "basecolor", "rm"):
        assert np.allclose(reg.maps[m], full.maps[m][sl], atol=1e-4), m
