import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from core import layout, route
from core.bake import bake
from core.document import Document
from core.panel import Panel, Bevel
from core.wire import Wire, End, resolved_path, sag_curve, polyline_length, is_stale, resample

ROOT = Path(__file__).resolve().parents[1]


def _doc(**kw):
    d = Document(canvas_w=1.0, canvas_h=1.0, texel_density=256, **kw)
    d.panels = [Panel(id="base", x=0.05, y=0.05, w=0.9, h=0.9, depth=0.01, corners=[0] * 4,
                      bevel=Bevel(width=0.01))]
    return d


def test_js_wire_simulation():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js not installed")
    out = subprocess.run([node, "--test", str(ROOT / "tests" / "js" / "wiresim.test.mjs")],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]


def test_wire_roundtrip_and_document():
    d = _doc()
    d.wires = [Wire(mode="route", a=End(panel="base", anchor="tl", dx=0.1, dy=0.1), b=End(x=0.8, y=0.8),
                    bundle=3, clip_spacing=0.1)]
    d.wire_sim["gravity_angle"] = 45.0
    again = Document.from_dict(json.loads(json.dumps(d.to_dict())))
    assert again.to_dict() == d.to_dict()
    assert again.wire_sim["gravity_angle"] == 45.0
    assert Document.from_dict({}).wire_sim["gravity_angle"] == 90.0     # default: down the image


def test_endpoint_follows_its_panel():
    d = _doc()
    end = End(panel="base", anchor="br", dx=-0.05, dy=-0.05)
    assert end.resolve(d) == pytest.approx((0.9, 0.9))
    d.panels[0].resize(0.5, 0.5)                   # top-left anchored
    assert end.resolve(d) == pytest.approx((0.5, 0.5))
    end.x, end.y = 0.3, 0.3
    d.panels = []                                  # panel gone: falls back to the free point
    assert end.resolve(d) == (0.3, 0.3)


def test_sim_wire_keeps_shape_and_moves_ends_with_panel():
    d = _doc()
    w = Wire(mode="sim", a=End(panel="base", anchor="tl", dx=0.1, dy=0.1),
             b=End(panel="base", anchor="tr", dx=-0.1, dy=0.1), slack=1.2)
    pts = sag_curve(w.a.resolve(d), w.b.resolve(d), 1.2)
    w.points = pts.tolist()
    w.settled = True
    assert not is_stale(w, d)
    d.panels[0].x += 0.02                          # move the panel
    assert is_stale(w, d)
    P = resolved_path(w, d)
    assert P[0] == pytest.approx(w.a.resolve(d)) and P[-1] == pytest.approx(w.b.resolve(d))
    assert polyline_length(P) == pytest.approx(polyline_length(pts), rel=1e-6)


def test_wire_adds_a_tube_to_the_height_field():
    d = _doc()
    base = bake(d, maps=("height",)).maps["height"].copy()
    d.wires = [Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.006, connectors=False,
                    profile="round")]
    r = bake(d, maps=("height", "wires"))
    H, M = r.maps["height"], r.maps["wires"]
    row = 128                                       # y = 0.5 m
    rise = H[row, 128] - base[row, 128]
    assert rise == pytest.approx(0.012, abs=0.0015)      # a 6 mm radius tube is 12 mm tall
    assert M[row, 128] == 1 and M[row - 10, 128] == 0    # mask covers the tube only
    assert H[row - 10, 128] == pytest.approx(base[row - 10, 128])


def test_connectors_clips_ribbon_and_bundle():
    d = _doc()
    d.wires = [Wire(mode="route", a=End(x=0.2, y=0.3), b=End(x=0.8, y=0.3), radius=0.004,
                    connectors=True, clip_spacing=0.15)]
    plain = Wire(mode="route", a=End(x=0.2, y=0.3), b=End(x=0.8, y=0.3), radius=0.004, connectors=False)
    Hc = bake(d, maps=("height",)).maps["height"]
    d.wires = [plain]
    Hp = bake(d, maps=("height",)).maps["height"]
    assert Hc[:, 50:56].max() > Hp[:, 50:56].max() + 0.001       # connector boss near x = 0.2
    assert (Hc > Hp + 0.0005).sum() > 40                         # clips add bumps along the run
    d.wires = [Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.004,
                    connectors=False, bundle=3, bundle_spacing=0.016)]
    M = bake(d, maps=("wires",)).maps["wires"]
    col = M[100:156, 128] > 0.5
    runs = np.count_nonzero(np.diff(col.astype(int)) == 1) + int(col[0])
    assert runs == 3, "three separate strands"
    d.wires = [Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.004,
                    connectors=False, profile="ribbon")]
    M = bake(d, maps=("wires",)).maps["wires"]
    width = (M[:, 128] > 0.5).sum() / 256
    assert width == pytest.approx(0.02, abs=0.006)                # ribbon is ~5 radii wide


def test_wires_stack_where_they_cross():
    d = _doc()
    h = Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.005, connectors=False, profile="round")
    v = Wire(mode="route", a=End(x=0.5, y=0.2), b=End(x=0.5, y=0.8), radius=0.005, connectors=False, profile="round")
    d.wires = [h, v]
    H = bake(d, maps=("height",)).maps["height"]
    top_cross = H[128, 128]
    top_alone = H[128, 80]
    assert top_cross == pytest.approx(top_alone + 0.010, abs=0.002)   # second tube rides on the first


def _axis_aligned(P, tol=1e-9):
    P = np.asarray(P)
    d = np.diff(P, axis=0)
    return bool(np.all((np.abs(d[:, 0]) < tol) | (np.abs(d[:, 1]) < tol)))


def test_route_is_manhattan_and_avoids_raised_blocks():
    d = _doc()
    d.panels.append(Panel(id="boss", x=0.46, y=0.46, w=0.08, h=0.08, depth=0.02))   # a raised boss
    w = Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), corner_radius=0.0)
    d.wires = [w]
    route.route_all(d)
    P = np.asarray(w.points)
    assert _axis_aligned(P), "right angles only"
    assert P[0] == pytest.approx([0.2, 0.5]) and P[-1] == pytest.approx([0.8, 0.5])
    Q = resample(P, 0.002)
    over = (Q[:, 0] > 0.465) & (Q[:, 0] < 0.535) & (Q[:, 1] > 0.465) & (Q[:, 1] < 0.535)
    assert not over.any(), "goes around the raised boss"


def test_route_45_and_rounded_corners():
    d = _doc()
    w = Wire(mode="route", a=End(x=0.2, y=0.2), b=End(x=0.8, y=0.8), allow45=True, corner_radius=0.0)
    d.wires = [w]
    route.route_all(d)
    dirs = np.diff(np.asarray(w.points), axis=0)
    diag = np.abs(np.abs(dirs[:, 0]) - np.abs(dirs[:, 1])) < 1e-9
    assert diag.any(), "uses a 45 degree run"
    w2 = Wire(mode="route", a=End(x=0.2, y=0.2), b=End(x=0.8, y=0.8), corner_radius=0.03)
    d.wires = [w2]
    route.route_all(d)
    assert not _axis_aligned(w2.points), "corners are rounded"
    assert len(w2.points) > 6


def test_later_routes_keep_clear_of_earlier_ones():
    d = _doc()
    a = Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.004, corner_radius=0)
    b = Wire(mode="route", a=End(x=0.2, y=0.5), b=End(x=0.8, y=0.5), radius=0.004, corner_radius=0,
             clearance=0.01)
    d.wires = [a, b]
    route.route_all(d)
    pb = resample(b.points, 0.005)
    mid_b = pb[(pb[:, 0] > 0.35) & (pb[:, 0] < 0.65)]
    assert len(mid_b) and np.all(np.abs(mid_b[:, 1] - 0.5) >= 0.015), "runs parallel, not on top"


def test_route_cache_and_determinism():
    d = _doc()
    d.wires = [Wire(mode="route", a=End(x=0.2, y=0.3), b=End(x=0.7, y=0.8))]
    route.route_all(d)
    first = [list(p) for p in d.wires[0].points]
    d2 = Document.from_dict(d.to_dict())
    d2.wires[0].points = []
    route.route_all(d2)
    assert d2.wires[0].points == first
    # Export resolution does not change the route.
    hi = Document.from_dict(d.to_dict())
    hi.texel_density = 1024
    hi.wires[0].points = []
    route.route_all(hi)
    assert hi.wires[0].points == first


def test_seamless_route_wraps_across_the_edge():
    d = _doc(tiling=True)
    d.panels = []
    w = Wire(mode="route", a=End(x=0.05, y=0.5), b=End(x=0.95, y=0.5), corner_radius=0)
    d.wires = [w]
    route.route_all(d)
    assert polyline_length(w.points) < 0.2, "goes the short way, across the seam"


def test_layout_wires_density_locked_and_modes():
    d = Document()
    d.panels = layout.generate(d, {"seed": 3})
    none = layout.generate_wires(d, d.panels, {"seed": 3, "wire_density": 0})
    assert none == []
    many = layout.generate_wires(d, d.panels, {"seed": 3, "wire_density": 0.5})
    assert len(many) > 5 and {w.mode for w in many} == {"sim", "route"}
    ids = {p.id for p in d.panels}
    assert all(w.a.panel in ids and w.b.panel in ids for w in many)
    keep = [Wire(name="mine", locked=True)]
    kept = layout.generate_wires(d, d.panels, {"seed": 3}, keep)
    assert kept[0].name == "mine"
    d.wires = many
    layout.place_new_sim_wires(d)
    assert all(len(w.points) > 2 for w in d.wires if w.mode == "sim")


def test_layout_endpoint_keeps_locked_wires(tmp_path):
    import app as app_module
    c = app_module.app.test_client()
    doc = c.get("/api/meta").get_json()["default_document"]
    assert doc["wires"], "the default document has a few wires"
    doc["wires"][0]["locked"] = True
    out = c.post("/api/layout", json={"doc": doc, "params": {"seed": 77}}).get_json()["doc"]
    kept = out["wires"][0]
    assert kept["id"] == doc["wires"][0]["id"]
    panel_ids = {p["id"] for p in out["panels"]}
    for end in ("a", "b"):
        assert kept[end]["panel"] is None or kept[end]["panel"] in panel_ids


def test_pathtracer_uses_wire_material():
    pytest.importorskip("numba")
    from render.scene import build_scene, tonemap
    from render.cpu import CPURenderer
    d = _doc()
    d.wires = [Wire(mode="route", a=End(x=0.1, y=0.5), b=End(x=0.9, y=0.5), radius=0.03, connectors=False)]
    R = CPURenderer()
    R.warmup()
    cols = []
    for alb in ([0.9, 0.05, 0.05], [0.05, 0.05, 0.9]):
        s = {"width": 64, "height": 64, "bloom": {"pathtrace": False}, "light": {"preset": "studio"},
             "camera": {"pitch": 89, "yaw": 0, "distance": 0.8},
             "wire_material": {"albedo": alb, "metallic": 0.0, "roughness": 0.6}}
        sc = build_scene(d, s)
        assert (sc["Vm"] >= 1.5).any()
        acc = np.zeros((64, 64, 3))
        for _ in range(16):
            R.render_pass(sc, acc)
        cols.append(acc[30:34, 20:44].mean(axis=(0, 1)))
    assert cols[0][0] > cols[0][2] and cols[1][2] > cols[1][0], "the wire takes the wire material's color"


def test_default_wire_is_a_14mm_hose():
    w = Wire()
    assert w.profile == "hose" and w.radius == pytest.approx(0.014)
    w2 = Wire.from_dict({})
    assert w2.profile == "hose" and w2.radius == pytest.approx(0.014)


def test_shuffle_wires_keeps_panels_and_locked_wires():
    import app as app_module
    c = app_module.app.test_client()
    doc = c.get("/api/meta").get_json()["default_document"]
    doc["wires"][0]["locked"] = True
    outs = []
    for seed in (11, 12):
        params = dict(doc["layout"], wire_seed=seed)
        outs.append(c.post("/api/layout/wires", json={"doc": doc, "params": params}).get_json()["doc"])
    for out in outs:
        assert out["panels"] == doc["panels"], "panels untouched"
        assert out["wires"][0]["id"] == doc["wires"][0]["id"], "locked wire kept"
        assert len(out["wires"]) > 1
        sims = [w for w in out["wires"] if w["mode"] == "sim"]
        assert all(len(w["points"]) > 2 for w in sims)
    ids = lambda o: [w["id"] for w in o["wires"][1:]]
    geom = lambda o: [(w["a"], w["b"], w["profile"]) for w in o["wires"][1:]]
    assert geom(outs[0]) != geom(outs[1]), "a different wire seed gives different wires"
    # A mix of kinds over a few shuffles.
    kinds = set()
    for seed in range(20, 30):
        out = c.post("/api/layout/wires", json={"doc": doc, "params": dict(doc["layout"], wire_seed=seed, wire_density=0.4)}).get_json()["doc"]
        kinds |= {(w["mode"], w["profile"]) for w in out["wires"]}
        kinds |= {("bundle",) for w in out["wires"] if w["bundle"] > 1}
    assert {m for m, *_ in kinds} >= {"sim", "route", "bundle"}
    assert {k[1] for k in kinds if len(k) == 2} == {"round", "ribbon", "hose"}


def test_shuffle_wires_with_zero_density_still_makes_wires():
    import app as app_module
    c = app_module.app.test_client()
    doc = c.get("/api/meta").get_json()["default_document"]
    out = c.post("/api/layout/wires", json={"doc": doc, "params": dict(doc["layout"], wire_seed=5, wire_density=0)}).get_json()["doc"]
    assert len(out["wires"]) > 0


def test_layout_wires_spread_out_when_most_panels_are_inset():
    """Regression: with mostly inset panels, every wire piled onto one pair of panels."""
    from collections import Counter
    params = {"seed": 95984, "inset_chance": 0.78, "nest_chance": 1.0, "stop_chance": 1.0,
              "wire_density": 0.83, "wire_sim_share": 0.0}
    d = Document()
    d.panels = layout.generate(d, params)
    ws = layout.generate_wires(d, d.panels, params)
    pairs = Counter(frozenset((w.a.panel, w.b.panel)) for w in ws)
    ends = Counter(e.panel for w in ws for e in (w.a, w.b))
    assert len(ws) >= 8
    assert len(pairs) >= 6, "wires use many different panel pairs"
    assert max(pairs.values()) <= layout.MAX_WIRES_PER_PAIR
    assert max(ends.values()) <= layout.MAX_WIRES_PER_PANEL
    inset_ids = {p.id for p in d.panels if p.mode == "inset"}
    assert any(e in inset_ids for e in ends), "inset panels take wire ends too"


def test_layout_makes_fewer_wires_rather_than_piling_up():
    d = Document()
    d.panels = [Panel(id="a", x=0.1, y=0.1, w=0.4, h=0.4), Panel(id="b", x=1.0, y=0.1, w=0.4, h=0.4),
                Panel(id="tiny", x=1.6, y=1.6, w=0.05, h=0.05)]
    ws = layout.generate_wires(d, d.panels, {"seed": 1, "wire_density": 5.0})
    assert len(ws) == layout.MAX_WIRES_PER_PAIR          # one usable pair, capped
    assert all({w.a.panel, w.b.panel} == {"a", "b"} for w in ws)
