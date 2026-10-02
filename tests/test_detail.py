"""Zoomed-in detail bakes must match the same pixels of a full bake exactly."""
import numpy as np
import pytest

from core import layout
from core.bake import bake, bake_region
from core.document import Document
from core.panel import Panel, Bevel, Detail, Emission
from core.wire import Wire, End


def _doc(tiling=False):
    d = Document(canvas_w=1.0, canvas_h=1.0, texel_density=1024, tiling=tiling)
    d.panels = layout.generate(d, {"seed": 7, "min_size": 0.12, "max_size": 0.4, "emissive_density": 0.6})
    d.panels.append(Panel(id="edge", x=0.85, y=0.4, w=0.3, h=0.2, depth=0.015, bevel=Bevel(width=0.01),
                          details=[Detail(anchor="c", ox=0, oy=0, w=0.04, h=0.02, shape="rect",
                                          emission=Emission(True, [1, 0.5, 0], 3))]))
    d.wires = [Wire(mode="route", a=End(x=0.1, y=0.3), b=End(x=0.9, y=0.7), radius=0.006),
               Wire(mode="sim", a=End(x=0.2, y=0.8), b=End(x=0.7, y=0.85), slack=1.2)]
    return d


MAPS = ("normal", "height", "ao", "curvature", "id", "emissive", "wires")


def _full(doc):
    return bake(doc, maps=MAPS + ("spill",))


@pytest.mark.parametrize("tiling,rect", [
    (False, (0.30, 0.25, 0.55, 0.45)),    # interior
    (False, (0.0, 0.0, 0.2, 0.15)),       # top-left corner, clamped at the canvas
    (False, (0.8, 0.85, 1.0, 1.0)),       # bottom-right corner
    (True, (0.82, 0.35, 1.0, 0.65)),      # touches the seam: margin wraps to the left side
    (True, (0.0, 0.0, 0.12, 0.1)),        # corner of a tiling canvas
])
def test_detail_matches_full_bake(tiling, rect):
    doc = _doc(tiling)
    full = _full(doc)
    fi = full.info
    norm = {k: fi[k] for k in ("height_min", "height_max", "emissive_scale", "curvature_scale")}
    reg = bake_region(doc, *rect, 1024, MAPS, norm)
    x0, y0, x1, y1 = reg.info["region"]
    i0, j0 = round(x0 * 1024), round(y0 * 1024)
    h, w = reg.maps["height"].shape
    sl = (slice(j0, j0 + h), slice(i0, i0 + w))
    assert (w, h) == (reg.info["width"], reg.info["height"])
    assert np.allclose(reg.maps["height"], full.maps["height"][sl], atol=1e-6)
    assert np.allclose(reg.maps["normal"], full.maps["normal"][sl], atol=1e-4)
    assert np.allclose(reg.maps["ao"], full.maps["ao"][sl], atol=1e-4)
    assert np.allclose(reg.maps["curvature"], full.maps["curvature"][sl], atol=1e-4)
    assert np.allclose(reg.maps["id"], full.maps["id"][sl])
    assert np.allclose(reg.maps["emissive"], full.maps["emissive"][sl], atol=1e-5)
    assert np.allclose(reg.maps["wires"], full.maps["wires"][sl], atol=1e-6)
    # Encoded with the overview's scaling, so PNG values match too.
    for m in MAPS:
        diff = np.abs(reg.encode(m).astype(int) - full.encode(m)[sl].astype(int))
        assert diff.max() <= 1, m          # at most one rounding step apart


def test_detail_is_sharper_than_the_overview():
    doc = _doc()
    over = bake(doc, maps=("normal",), max_res=256)              # what the 2D view had before
    reg = bake_region(doc, 0.3, 0.25, 0.55, 0.45, 1024, ("normal",))
    edge_px = lambda n: int((np.abs(n[..., 0]) > 0.2).sum())
    i0, j0 = 0.3 * 256, 0.25 * 256
    patch = over.maps["normal"][int(j0):int(j0) + 51, int(i0):int(i0) + 64]
    # Same area at 4x the density: about 4x as many pixels across each bevel.
    assert edge_px(reg.maps["normal"]) > 8 * edge_px(patch)


def test_detail_endpoint_and_limits():
    import app as app_module
    c = app_module.app.test_client()
    d = c.get("/api/meta").get_json()["default_document"]
    d["texel_density"] = 2048
    full = c.post("/api/bake", json={"doc": d, "maps": ["normal", "height", "curvature", "emissive"], "max_res": 512}).get_json()
    norm = {k: full["info"][k] for k in ("height_min", "height_max", "emissive_scale", "curvature_scale")}
    r = c.post("/api/bake", json={"doc": d, "region": {"x0": 0.5, "y0": 0.5, "x1": 0.9, "y1": 0.8, "px_per_m": 2048},
                                  "norm": norm})
    assert r.status_code == 200
    j = r.get_json()
    assert set(j["maps"]) == set(MAPS)
    assert (j["info"]["width"], j["info"]["height"]) == (820, 615)
    assert j["info"]["height_min"] == norm["height_min"]
    # Density above the document's is clamped to the document's.
    r = c.post("/api/bake", json={"doc": d, "region": {"x0": 0.5, "y0": 0.5, "x1": 0.6, "y1": 0.6, "px_per_m": 9999}})
    assert r.get_json()["info"]["px_per_m"] == 2048
    # Too large, outside the canvas, and malformed requests are refused cleanly.
    assert c.post("/api/bake", json={"doc": d, "region": {"x0": 0, "y0": 0, "x1": 2, "y1": 2, "px_per_m": 2048}}).status_code == 400
    assert c.post("/api/bake", json={"doc": d, "region": {"x0": 3, "y0": 3, "x1": 4, "y1": 4}}).status_code == 400
    assert c.post("/api/bake", json={"doc": d, "region": {"x0": "a"}}).status_code == 400
