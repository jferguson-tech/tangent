"""Leaks (core/fluids.py), brush strokes (core/brush.py), heat and soot, wire
wear, and the staged weathering cache."""
import numpy as np
import pytest

from core import weather, fluids, brush
from core.bake import bake, bake_region, DETAIL_MAPS, MASK_MAPS
from core.document import Document
from core.panel import Panel, Bevel, Detail
from core.wire import Wire, End

PPM = 256          # px per meter in these tests


def _doc(age=0.0, tiling=False, **w):
    d = Document(canvas_w=1.0, canvas_h=1.0, texel_density=PPM, tiling=tiling)
    d.panels = [Panel(id="top", x=0.1, y=0.1, w=0.8, h=0.3, depth=0.01, bevel=Bevel(width=0.004)),
                Panel(id="low", x=0.1, y=0.45, w=0.8, h=0.45, depth=0.01, bevel=Bevel(width=0.004))]
    d.weathering.update(enabled=True, age=age, **w)
    return d


def _with(doc, **kw):
    d = doc.to_dict()
    d.update(kw)
    return Document.from_dict(d)


def px(v):
    return int(v * PPM)


# ---- settings -------------------------------------------------------------------------
def test_new_settings_are_sanitized():
    w = weather.weathering_from({"heat": 9, "wire_wear": -1, "auto_leaks": 400})
    assert w["heat"] == 1 and w["wire_wear"] == 0 and w["auto_leaks"] == weather.MAX_AUTO_LEAKS
    leaks = fluids.leaks_from([{"liquid": "lava", "amount": 5, "x": 0.2, "y": 0.3}, "junk"])
    assert len(leaks) == 1 and leaks[0].liquid == "water" and leaks[0].amount == 1
    s = brush.strokes_from([{"channel": "all", "mode": "add", "points": [[0, 0]]},       # "all" only erases
                            {"channel": "soot", "points": []},                           # no points: dropped
                            {"channel": "dirt", "mode": "erase", "radius": 99, "points": [[1, "x"], [0.5, 0.5]]}])
    assert [x["channel"] for x in s] == ["rust", "dirt"]
    assert s[1]["radius"] == 1.0 and s[1]["points"] == [[0.5, 0.5]]
    assert set(weather.PRESETS["leaking"]) >= set(weather.SLIDERS)
    assert len(MASK_MAPS) == 9


def test_document_round_trip_and_active():
    d = _doc()
    assert not weather.active(d)                       # enabled but nothing to do
    d2 = _with(d, leaks=[{"liquid": "oil", "x": 0.5, "y": 0.2}],
               strokes=[{"channel": "rust", "points": [[0.5, 0.5]]}])
    assert weather.active(d2)
    d3 = Document.from_dict(d2.to_dict())
    assert d3.leaks[0].liquid == "oil" and d3.strokes == d2.strokes


# ---- leaks ------------------------------------------------------------------------------
def test_leak_runs_down_in_gravity_direction():
    for angle, below in ((90, True), (270, False)):
        d = _with(_doc(), leaks=[{"liquid": "water", "amount": 0.8, "x": 0.5, "y": 0.5}])
        d.wire_sim["gravity_angle"] = angle
        r = bake(d, ("mask_wet", "mask_residue", "basecolor")).maps
        trail = np.maximum(r["mask_wet"], r["mask_residue"])
        down = trail[px(0.55):px(0.85), px(0.45):px(0.55)].mean()
        up = trail[px(0.15):px(0.45), px(0.45):px(0.55)].mean()
        assert (down > 4 * up + 0.01) if below else (up > 4 * down + 0.01), angle


def test_leak_follows_its_panel():
    leak = {"liquid": "oil", "amount": 0.8, "panel": "top", "anchor": "c", "dx": 0, "dy": 0}
    d = _with(_doc(), leaks=[leak])
    a = bake(d, ("mask_fluid",)).maps["mask_fluid"]
    d.find("top").x -= 0.2                              # the panel moves left
    b = bake(d, ("mask_fluid",)).maps["mask_fluid"]
    cols = lambda m: np.flatnonzero(m.sum(axis=0) > 0.5)
    assert abs((cols(a).mean() - cols(b).mean()) - px(0.2)) < 4


def test_oil_is_dark_and_glossy():
    d = _with(_doc(), leaks=[{"liquid": "oil", "amount": 1.0, "x": 0.5, "y": 0.15}])
    r = bake(d, ("mask_fluid", "basecolor", "roughness")).maps
    on = r["mask_fluid"] > 0.8
    off = r["mask_fluid"] == 0
    assert on.sum() > 20
    assert r["basecolor"][on].mean() < 0.5 * r["basecolor"][off].mean()
    assert r["roughness"][on].mean() < r["roughness"][off].mean()


def test_auto_leaks_are_seeded():
    d = _doc(auto_leaks=3)
    for p in d.panels:
        p.details = [Detail(kind="bolt", anchor="corners", ox=0.03, oy=0.03, w=0.015, h=0.015)]
    s1 = fluids.auto_sites(d, 3, 1)
    assert len(s1) == 3 and s1 == fluids.auto_sites(d, 3, 1)
    assert s1 != fluids.auto_sites(d, 3, 2)
    assert bake(d, ("mask_wet",)).maps["mask_wet"].max() > 0.5


# ---- brush --------------------------------------------------------------------------------
def test_brush_paints_at_age_zero():
    s = {"channel": "rust", "mode": "add", "radius": 0.05, "strength": 1, "hardness": 0.6,
         "points": [[0.3, 0.6], [0.7, 0.6]]}
    r = bake(_with(_doc(), strokes=[s]), ("mask_rust", "basecolor")).maps["mask_rust"]
    assert r[px(0.6), px(0.4):px(0.6)].min() > 0.9            # on the stroke
    assert r[px(0.75):, :].max() == 0 and r[:px(0.5), :].max() == 0


def test_erase_removes_simulated_effect_and_order_matters():
    base = _doc(age=1.0, rust=1.0, dirt=1.0)
    full = bake(base, ("mask_rust", "mask_dirt")).maps
    band = (slice(px(0.4), px(0.5)), slice(px(0.2), px(0.8)))
    assert full["mask_rust"][band].max() > 0.5 and full["mask_dirt"][band].max() > 0.3
    erase = {"channel": "all", "mode": "erase", "radius": 0.08, "strength": 1, "hardness": 0.9,
             "points": [[0.15, 0.45], [0.85, 0.45]]}
    r = bake(_with(base, strokes=[erase]), ("mask_rust", "mask_dirt")).maps
    assert r["mask_rust"][band].max() < 0.05 and r["mask_dirt"][band].max() < 0.05
    # Painting after erasing puts rust back; before erasing it is erased too.
    paint = {"channel": "rust", "mode": "add", "radius": 0.03, "strength": 1, "hardness": 0.6,
             "points": [[0.4, 0.45], [0.6, 0.45]]}
    after = bake(_with(base, strokes=[erase, paint]), ("mask_rust",)).maps["mask_rust"]
    before = bake(_with(base, strokes=[paint, erase]), ("mask_rust",)).maps["mask_rust"]
    assert after[px(0.45), px(0.45):px(0.55)].min() > 0.9
    assert before[px(0.45), px(0.45):px(0.55)].max() < 0.05


def test_stroke_wraps_on_tiling_canvas():
    s = {"channel": "soot", "mode": "add", "radius": 0.04, "strength": 1, "hardness": 0.8,
         "points": [[0.98, 0.5]]}
    r = bake(_with(_doc(tiling=True), strokes=[s]), ("mask_soot",)).maps["mask_soot"]
    assert r[px(0.5), 2] > 0.8 and r[px(0.5), -2] > 0.8      # both sides of the seam


def test_stroke_does_not_rerun_simulations(monkeypatch):
    d = _doc(age=0.8)
    bake(d, ("basecolor",))
    calls = []
    for name in ("_base", "_terrain_fields", "_heat", "_wires", "_leaks"):
        orig = getattr(weather, name)
        monkeypatch.setattr(weather, name, lambda *a, _o=orig, _n=name: calls.append(_n) or _o(*a))
    d2 = _with(d, strokes=[{"channel": "dirt", "points": [[0.5, 0.5]]}])
    bake(d2, ("basecolor",))
    assert calls == []


# ---- heat, soot, wires -----------------------------------------------------------------------
def test_vent_heats_and_soot_rises_against_gravity():
    d = _doc(age=1.0, heat=1.0)
    d.find("low").details = [Detail(kind="vent", shape="rect", anchor="c", ox=0, oy=0, w=0.08, h=0.04,
                                    depth=0.004, mode="inset")]
    r = bake(d, ("mask_heat", "mask_soot")).maps
    cx, cy = px(0.5), px(0.675)                         # vent center
    assert r["mask_heat"][cy, cx - 15:cx + 15].mean() > 0.4
    assert r["mask_heat"][px(0.95), cx] < 0.1
    above = r["mask_soot"][cy - px(0.15):cy - px(0.03), cx - 5:cx + 5].mean()
    below = r["mask_soot"][cy + px(0.03):cy + px(0.15), cx - 5:cx + 5].mean()
    assert above > 2 * below + 0.02


def test_wire_drips_streak_below_its_low_point():
    def streaks(ww):
        d = _doc(age=1.0, streaks=0.0, wire_wear=ww)
        d.panels = [Panel(id="p", x=0.0, y=0.0, w=1.0, h=1.0, depth=0.005, bevel=Bevel(width=0.003))]
        d.wires = [Wire(mode="sim", a=End(x=0.2, y=0.2), b=End(x=0.8, y=0.2), slack=1.3, radius=0.006)]
        return bake(d, ("mask_streaks",)).maps["mask_streaks"]
    with_drips, without = streaks(1.0), streaks(0.0)
    # The sagging wire's lowest point is at x = 0.5, a little below y = 0.2 + sag.
    col = slice(px(0.47), px(0.53))
    region = (slice(px(0.45), px(0.9)), col)
    assert with_drips[region].mean() > without[region].mean() + 0.05
    side = (slice(px(0.45), px(0.9)), slice(px(0.05), px(0.12)))
    assert with_drips[region].mean() > 3 * with_drips[side].mean() + 0.02


# ---- consistency ---------------------------------------------------------------------------
@pytest.mark.parametrize("tiling", [False, True])
def test_detail_matches_full_bake_with_effects(tiling):
    d = _with(_doc(age=0.7, heat=0.8, tiling=tiling),
              leaks=[{"liquid": "coolant", "amount": 0.7, "x": 0.5, "y": 0.2}],
              strokes=[{"channel": "rust", "points": [[0.2, 0.3], [0.4, 0.35]], "radius": 0.03}])
    full = bake(d, DETAIL_MAPS + ("spill",))
    norm = {k: full.info[k] for k in ("height_min", "height_max", "emissive_scale", "curvature_scale")}
    reg = bake_region(d, 0.15, 0.15, 0.6, 0.5, PPM, DETAIL_MAPS, norm)
    x0, y0 = reg.info["region"][:2]
    i0, j0 = round(x0 * PPM), round(y0 * PPM)
    h, w = reg.maps["height"].shape
    sl = (slice(j0, j0 + h), slice(i0, i0 + w))
    for m in ("height", "normal", "basecolor", "rm"):
        assert np.allclose(reg.maps[m], full.maps[m][sl], atol=1e-4), m


# ---- editable strokes -----------------------------------------------------------------------
def _rust_stroke(**kw):
    s = {"id": "s1", "channel": "rust", "mode": "add", "radius": 0.04, "strength": 1, "hardness": 0.6,
         "points": [[0.4, 0.65], [0.6, 0.65]]}
    s.update(kw)
    return s


def test_stroke_settings_are_sanitized():
    s = brush.stroke_from({"points": [[0.1, 0.2]], "panel": "low", "origin": [0.1, "x"]})
    assert s["id"] and s["visible"] is True and s["panel"] is None and s["origin"] is None
    s = brush.stroke_from({"points": [[0.1, 0.2]], "panel": "low", "origin": [0.1, 0.45], "visible": False})
    assert s["panel"] == "low" and s["origin"] == [0.1, 0.45] and s["visible"] is False


def test_hidden_stroke_is_ignored():
    on = bake(_with(_doc(), strokes=[_rust_stroke()]), ("mask_rust",)).maps["mask_rust"]
    off = bake(_with(_doc(), strokes=[_rust_stroke(visible=False)]), ("mask_rust",)).maps["mask_rust"]
    assert on.max() > 0.9 and off.max() == 0


def test_stroke_follows_its_panel():
    s = _rust_stroke(panel="low", origin=[0.1, 0.45])
    d = _with(_doc(), strokes=[s])
    a = bake(d, ("mask_rust",)).maps["mask_rust"]
    d.find("low").x += 0.05
    d.find("low").y += 0.1
    b = bake(d, ("mask_rust",)).maps["mask_rust"]
    rows = lambda m: np.flatnonzero(m.max(axis=1) > 0.5)
    cols = lambda m: np.flatnonzero(m.max(axis=0) > 0.5)
    assert abs((rows(b).mean() - rows(a).mean()) - px(0.1)) < 2
    assert abs((cols(b).mean() - cols(a).mean()) - px(0.05)) < 2
    # A stroke whose panel is gone stays where it was painted.
    d.panels = [p for p in d.panels if p.id != "low"]
    c = bake(d, ("mask_rust",)).maps["mask_rust"]
    assert abs(rows(c).mean() - rows(a).mean()) < 2
