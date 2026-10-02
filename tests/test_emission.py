import numpy as np
import pytest

from core import layout
from core.bake import bake
from core.document import Document
from core.panel import Panel, Bevel, Detail, Emission, Groove
from core.units import EMISSION_UNIT
from generators import Grid, PanelGenerator
from render.scene import emitter_tables


def _emission(doc):
    nx, ny = doc.resolution()
    return PanelGenerator().height(doc, Grid(doc.canvas_w, doc.canvas_h, nx, ny, doc.tiling), emission=True)[2]


def _lum(E):
    return E @ np.array([0.2126, 0.7152, 0.0722])


def test_emission_defaults_off_and_roundtrip():
    p = Panel()
    assert not p.has_emission() and p.emission.radiance() is None
    p.details.append(Detail(emission=Emission(True, [1, 0.5, 0], 3.0)))
    q = Panel.from_dict(p.to_dict())
    assert q.has_emission() and q.details[0].emission.strength == 3.0
    white = Emission(True, [1, 1, 1], 2.0).radiance()
    assert np.allclose(white, 2.0 * EMISSION_UNIT)


def test_no_emission_gives_black_map():
    doc = Document(canvas_w=0.5, canvas_h=0.5, texel_density=128, panels=[Panel(x=0.1, y=0.1, w=0.3, h=0.3)])
    assert _emission(doc).max() == 0


def test_light_strip_emits_its_area():
    det = Detail(kind="light", shape="rect", anchor="c", ox=0, oy=0, w=0.2, h=0.02,
                 emission=Emission(True, [1, 1, 1], 1.0))
    doc = Document(canvas_w=0.5, canvas_h=0.5, texel_density=512,
                   panels=[Panel(x=0.05, y=0.05, w=0.4, h=0.4, details=[det])])
    E = _emission(doc)
    px_area = (1 / 512) ** 2
    assert E[..., 0].sum() * px_area == pytest.approx(0.2 * 0.02 * EMISSION_UNIT, rel=0.03)


def test_later_panels_hide_emission_beneath():
    lit = Panel(x=0.05, y=0.05, w=0.4, h=0.4, bevel=Bevel(width=0.01),
                emission=Emission(True, [1, 1, 1], 1.0))
    cover = Panel(x=0.25, y=0.05, w=0.2, h=0.4)
    doc = Document(canvas_w=0.5, canvas_h=0.5, texel_density=256, panels=[lit, cover])
    E = _lum(_emission(doc))
    assert E[64, 40] > 0                  # x = 0.16 m: lit face
    assert E[64, 100] == 0                # x = 0.39 m: covered by the plain panel


def test_face_emission_skips_bevel_and_diffuser_brightens_center():
    p = Panel(x=0.0, y=0.0, w=0.5, h=0.5, corners=[0] * 4, bevel=Bevel(width=0.03),
              emission=Emission(True, [1, 1, 1], 1.0, diffuser=1.0))
    doc = Document(canvas_w=0.5, canvas_h=0.5, texel_density=256, panels=[p])
    row = _lum(_emission(doc))[64]
    assert row[3] == 0                    # inside the 3 cm bevel: dark
    assert row[64] > 2 * row[12] > 0      # diffuser: center much brighter than near the edge


def test_groove_glow_is_confined_to_the_groove():
    p = Panel(x=0.0, y=0.0, w=0.5, h=0.5, corners=[0] * 4, bevel=Bevel(width=0.01),
              groove=Groove(offset=0.04, width=0.01, depth=0.002, emission=Emission(True, [0, 1, 1], 2.0)))
    doc = Document(canvas_w=0.5, canvas_h=0.5, texel_density=512, panels=[p])
    row = _lum(_emission(doc))[128]
    lit = np.flatnonzero(row > 0) / 512.0
    assert lit.min() >= 0.04 - 0.002 and lit[lit < 0.25].max() <= 0.05 + 0.002


def test_bake_emissive_and_spill_maps():
    det = Detail(kind="light", shape="rect", anchor="c", ox=0, oy=0, w=0.1, h=0.01,
                 emission=Emission(True, [0.2, 0.8, 1.0], 4.0))
    doc = Document(canvas_w=1.0, canvas_h=1.0, texel_density=128,
                   panels=[Panel(x=0.1, y=0.1, w=0.8, h=0.8, details=[det])])
    r = bake(doc, maps=("emissive", "spill"))
    assert r.info["has_emission"] and r.info["emissive_scale"] > 0
    enc = r.encode("emissive")
    assert enc.dtype == np.uint8 and enc.max() == 255
    S = _lum(r.maps["spill"])
    assert S[64, 64] > S[64, 64 + 20] > S[64, 64 + 50] >= 0   # spill falls off with distance
    assert _lum(r.maps["emissive"])[64, 64 + 20] == 0          # emission itself is sharp


def test_layout_lights_follow_density_without_moving_geometry():
    doc = Document()
    dark = layout.generate(doc, {"seed": 11, "emissive_density": 0})
    lit = layout.generate(doc, {"seed": 11, "emissive_density": 0.6})
    assert not any(p.has_emission() for p in dark)
    assert any(p.has_emission() for p in lit)
    assert [(p.x, p.y, p.w, p.h) for p in dark] == [(p.x, p.y, p.w, p.h) for p in lit]
    amber = layout.generate(doc, {"seed": 11, "emissive_density": 1, "light_palette": "amber"})
    cols = {tuple(d.emission.color) for p in amber for d in p.details if d.emission.enabled}
    assert cols == {tuple(layout.LIGHT_COLORS["amber"])}


def test_emitter_tables():
    Em = np.zeros((16, 32, 3))
    Em[4, 5] = 1.0
    Em[10, 20] = 3.0
    E, epd, cdf, idx = emitter_tables(Em, 2.0, 1.0)
    assert list(idx) == [4 * 32 + 5, 10 * 32 + 20]
    assert cdf[0] == 0 and cdf[-1] == 1 and cdf[1] == pytest.approx(0.25)
    assert (epd * (2.0 / 32) * (1.0 / 16)).sum() == pytest.approx(1.0)
    _, _, cdf0, idx0 = emitter_tables(np.zeros((4, 4, 3)), 1, 1)
    assert len(idx0) == 1 and len(cdf0) == 2


# ---- pathtracer (needs Numba) ---------------------------------------------------
def _render_block(nee, rr, spp, view_emissive=True):
    pt = pytest.importorskip("render.cpu_pathtracer")
    from render.hdri import EnvMap
    n = 64
    W = 1.0
    # A 6 cm block with 45-degree sides. The sides glow and light the floor.
    yy, xx = np.mgrid[0:n, 0:n]
    dist_edge = np.minimum.reduce([xx - 20, 43 - xx, yy - 20, 43 - yy]).astype(float)
    Hm = np.clip((dist_edge + 0.5) * (W / n), 0, 0.06)
    Em = np.zeros((n, n, 3))
    Em[(Hm > 0) & (Hm < 0.06)] = 3.0 if view_emissive else 0.0
    Nm = np.zeros((n, n, 3))
    Nm[..., 2] = 1
    Mt = np.zeros((n, n, 5))
    Mt[..., :3] = 0.8
    Mt[..., 4] = 1.0
    texel = W / n
    sc = np.array([-W / 2, W / 2, -W / 2, W / 2, 0, 0.06, W, W, texel, 0, 1, 0, 1, 0, 0.02 * texel])
    cam = np.array([0, 0, 3, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0.17, 1.0])
    mat = np.array([0.8, 0.8, 0.8, 0.0, 1.0, 0.03, 0.03, 0.03, 0.6])
    env = EnvMap(np.zeros((8, 16, 3), np.float32))
    E, EPD, ECDF, EIDX = emitter_tables(Em, W, W)
    emp = np.array([1.0, 1.0, 1.0 if nee else 0.0])
    opt = np.array([1.0, 0.0, 0.02 if rr else 0.0])
    acc = np.zeros((32, 32, 3))
    for _ in range(spp):
        pt.render_pass(acc, Hm, Nm, Mt, sc, cam, mat, np.zeros((0, 14)), env.img, env.blur,
                       env.pdf, env.marg, env.cond, np.array([0.0, 0, 0, 0]),
                       E, EPD, ECDF, EIDX, emp, opt)
    return acc / spp


def test_emitters_light_the_floor_unbiased():
    full = _render_block(True, True, 256)
    no_rr = _render_block(True, False, 256)
    bsdf = _render_block(False, False, 2048)
    floor = np.zeros((32, 32), bool)
    floor[:6, :] = floor[-6:, :] = floor[:, :6] = floor[:, -6:] = True   # ring of floor around the block
    a, b, c = (img[floor].mean() for img in (full, no_rr, bsdf))
    assert a > 0.003                                   # the glow reaches the floor
    assert a == pytest.approx(b, rel=0.08)             # roulette keeps the estimate unbiased
    assert a == pytest.approx(c, rel=0.08)             # emitter sampling agrees with BSDF-only


def test_no_emitters_no_light():
    img = _render_block(True, True, 8, view_emissive=False)
    assert img.max() == 0


def test_bloom_conserves_energy_and_falls_off():
    from core.filters import bloom
    img = np.zeros((200, 320, 3))
    img[100, 160] = 400.0
    out = bloom(img, threshold=1.0, intensity=0.5, radius=1.0)
    assert (out - img).sum() / (400.0 * 3 * 0.5) == pytest.approx(1.0, rel=0.03)
    row = out[100, :, 0]
    assert row[165] > row[175] > row[200] >= 0
    dim = np.full((50, 60, 3), 0.5)
    assert np.array_equal(bloom(dim, threshold=1.0), dim)          # below threshold: untouched
    assert bloom(img, intensity=0.0) is img


def test_pathtrace_frames_use_bloom_setting():
    from render.scene import tonemap, merge_settings
    acc = np.zeros((40, 60, 3))
    acc[20, 30] = 200.0
    on = tonemap(acc, 1, 0.0, merge_settings({})["bloom"])
    off = tonemap(acc, 1, 0.0, merge_settings({"bloom": {"pathtrace": False}})["bloom"])
    assert on[20, 34].sum() > off[20, 34].sum() == 0
    assert merge_settings({})["bloom"]["pathtrace"] is True
