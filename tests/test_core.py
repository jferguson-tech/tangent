import struct
import zlib

import numpy as np
import pytest

from core import layout, profiles, sdf
from core.bake import bake, normals_from_height
from core.document import Document
from core.panel import Panel, Bevel, Detail, Groove
from core.png import encode_png
from generators import Grid, PanelGenerator


def _height(doc, nx=None, ny=None):
    nx = nx or doc.resolution()[0]
    ny = ny or doc.resolution()[1]
    H = PanelGenerator().height(doc, Grid(doc.canvas_w, doc.canvas_h, nx, ny, doc.tiling))[0]
    return H


def _bevel_px(row, depth):
    """Count pixels on the rising bevel from the left edge of a height row."""
    start = np.argmax(row > 1e-9)
    top = np.argmax(row >= depth * 0.999)
    return top - start


def test_bevel_width_is_constant_when_resizing():
    doc = Document(canvas_w=2.0, canvas_h=1.0, texel_density=512)
    p = Panel(x=0.1, y=0.1, w=0.4, h=0.6, depth=0.01, corners=[0, 0, 0, 0],
              bevel=Bevel(width=0.02, profile="linear"))
    doc.panels = [p]
    row = _height(doc)[int(0.4 * 512)]
    before = _bevel_px(row, p.depth)
    p.resize(1.6, 0.8)
    row = _height(doc)[int(0.4 * 512)]
    after = _bevel_px(row, p.depth)
    assert before == after
    assert abs(before - 0.02 * 512) <= 1.5


def test_resize_only_changes_rectangle():
    p = Panel(x=1, y=1, w=0.5, h=0.5, corners=[0.01, 0.02, 0.03, 0.04],
              bevel=Bevel(width=0.007, profile="ogee"),
              details=[Detail(w=0.02, ox=0.03, oy=0.03)])
    before = (p.bevel.width, p.bevel.profile, list(p.corners), p.details[0].w, p.details[0].ox)
    p.resize(1.0, 0.25, anchor="br")
    assert (p.x + p.w, p.y + p.h) == pytest.approx((1.5, 1.5))
    assert (p.bevel.width, p.bevel.profile, list(p.corners), p.details[0].w, p.details[0].ox) == before
    p.resize(0.5, 0.5, anchor="c")
    assert (p.x + p.w / 2, p.y + p.h / 2) == pytest.approx((1.0, 1.375))


def test_detail_anchor_tracks_panel_edge_on_resize():
    p = Panel(x=0, y=0, w=0.5, h=0.5, details=[Detail(anchor="br", ox=0.03, oy=0.04)])
    (cx, cy), = p.details[0].centers(*p.rect)
    assert (cx, cy) == pytest.approx((0.47, 0.46))
    p.resize(1.0, 2.0)
    (cx, cy), = p.details[0].centers(*p.rect)
    assert (cx, cy) == pytest.approx((0.97, 1.96))


def test_round_box_sdf_is_exact():
    hx, hy, r = 0.5, 0.3, 0.1
    d = sdf.sd_round_box(np.array([0.0, 0.5, 0.7]), np.array([0.0, 0.0, 0.0]), hx, hy, [r] * 4)
    assert d == pytest.approx([-0.3, 0.0, 0.2])
    # Outside a rounded corner the distance is to the arc.
    c = np.array([hx - r, hy - r])
    p = c + np.array([0.3, 0.4])
    d = sdf.sd_round_box(np.array([p[0]]), np.array([p[1]]), hx, hy, [r] * 4)
    assert d[0] == pytest.approx(0.5 - r)


def test_chamfer_box_cuts_corner_at_45_degrees():
    hx = hy = 0.5
    c = 0.1
    # Midpoint of the chamfer edge lies on the outline.
    mid = np.array([hx - c / 2]), np.array([hy - c / 2])
    assert sdf.sd_chamfer_box(*mid, hx, hy, [c] * 4)[0] == pytest.approx(0.0, abs=1e-9)
    # The original corner is now outside.
    assert sdf.sd_chamfer_box(np.array([hx]), np.array([hy]), hx, hy, [c] * 4)[0] > 0


@pytest.mark.parametrize("name", list(profiles.PROFILES))
def test_profiles_span_zero_to_one(name):
    t = np.linspace(0, 1, 101)
    v = profiles.evaluate(name, t, [[0, 0], [0.5, 0.8], [1, 1]])
    assert v[0] == pytest.approx(0.0, abs=1e-6)
    assert v[-1] == pytest.approx(1.0, abs=1e-6)
    assert np.all(v >= -1e-9) and np.all(v <= 1 + 1e-9)


def test_normal_convention_opengl_and_directx():
    H = np.zeros((32, 32), np.float32)
    H += np.arange(32)[:, None] * 0.001   # rises toward the bottom of the image
    n = normals_from_height(H, 0.01, 0.01, 1.0, False)
    # Height falls toward +V (up the texture), so the surface faces +V: green > 0.5 in OpenGL.
    assert n[16, 16, 1] > 0.05
    H2 = np.zeros((32, 32), np.float32) + np.arange(32)[None, :] * 0.001  # rises to the right
    n2 = normals_from_height(H2, 0.01, 0.01, 1.0, False)
    assert n2[16, 16, 0] < -0.05
    doc = Document(canvas_w=1, canvas_h=1, texel_density=64,
                   panels=[Panel(x=0.2, y=0.2, w=0.6, h=0.6, bevel=Bevel(width=0.1))])
    r = bake(doc, maps=("normal",))
    gl = r.encode("normal", 8, "gl").astype(int)
    dx = r.encode("normal", 8, "dx").astype(int)
    assert np.all(np.abs(gl[..., 1] + dx[..., 1] - 255) <= 1)
    assert np.array_equal(gl[..., 0], dx[..., 0])


def test_seamless_tiling_wraps_across_the_edge():
    doc = Document(canvas_w=1.0, canvas_h=1.0, texel_density=128, tiling=True)
    doc.panels = [Panel(x=0.8, y=0.3, w=0.4, h=0.4, depth=0.01, bevel=Bevel(width=0.03))]
    H = _height(doc)
    row = H[64]
    assert row[:20].max() > 0.009        # the part past the right edge lands on the left
    assert abs(row[0] - row[-1]) < 1e-3  # continuous across the seam
    doc.tiling = False
    H2 = _height(doc)
    assert H2[64, :20].max() == 0.0      # clipped when not tiling


def test_bake_maps_shapes_and_ranges():
    doc = Document(canvas_w=1, canvas_h=0.5, texel_density=128)
    doc.panels = layout.generate(doc, {"seed": 5, "min_size": 0.1, "max_size": 0.3})
    r = bake(doc, maps=("normal", "height", "ao", "curvature", "id"), ss=2)
    assert r.maps["normal"].shape == (64, 128, 3)
    assert np.allclose(np.linalg.norm(r.maps["normal"], axis=-1), 1, atol=1e-4)
    assert 0 <= r.maps["ao"].min() and r.maps["ao"].max() <= 1
    assert r.info["height_max"] > 0
    assert r.encode("height", 16).dtype == np.uint16


def test_layout_is_deterministic_and_keeps_locked_panels():
    doc = Document()
    a = layout.generate(doc, {"seed": 42})
    b = layout.generate(doc, {"seed": 42})
    assert [(p.x, p.y, p.w, p.h) for p in a] == [(p.x, p.y, p.w, p.h) for p in b]
    keep = Panel(x=0.5, y=0.5, w=0.6, h=0.6, locked=True, name="keep")
    doc.panels = [keep]
    out = layout.generate(doc, {"seed": 7})
    assert out[0] is keep
    for p in out[1:]:
        overlap = not (p.x + p.w <= keep.x or keep.x + keep.w <= p.x
                       or p.y + p.h <= keep.y or keep.y + keep.h <= p.y)
        assert not overlap


def test_layout_panels_stay_inside_canvas_with_border():
    doc = Document(canvas_w=2, canvas_h=1)
    for seed in range(5):
        for p in layout.generate(doc, {"seed": seed, "border": 0.02}):
            assert p.x >= 0.02 - 1e-9 and p.y >= 0.02 - 1e-9
            assert p.x + p.w <= 2 - 0.02 + 1e-9 and p.y + p.h <= 1 - 0.02 + 1e-9


def test_layout_symmetry_mirrors():
    doc = Document(canvas_w=2, canvas_h=2)
    ps = layout.generate(doc, {"seed": 3, "symmetry": "x", "nest_chance": 0})
    rects = {(round(p.x, 6), round(p.y, 6), round(p.w, 6), round(p.h, 6)) for p in ps}
    for x, y, w, h in rects:
        assert (round(2 - x - w, 6), y, w, h) in rects


def test_alias_warning_depends_on_texel_density():
    p = Panel(w=0.5, h=0.5, bevel=Bevel(width=0.003))
    assert any("alias" in w for w in p.warnings(512))      # 1.5 px
    assert not any("alias" in w for w in p.warnings(1024))  # 3 px
    p.groove = Groove(width=0.002, depth=0.001)
    assert any("Groove" in w for w in p.warnings(512))


def test_document_roundtrip_and_sanitizing():
    doc = Document(texel_density=1024, tiling=True)
    doc.panels = layout.generate(doc, {"seed": 2})
    d = doc.to_dict()
    again = Document.from_dict(d).to_dict()
    assert again == d
    bad = Document.from_dict({"texel_density": "x", "canvas_w": 1e9,
                              "panels": [{"mode": "evil", "bevel": {"profile": "nope"}}]})
    assert bad.texel_density == 512
    assert max(bad.resolution()) <= 8192
    assert bad.panels[0].mode == "raise" and bad.panels[0].bevel.profile == "linear"


def _decode_png(data):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, chunks = 8, {}
    idat = b""
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        crc = struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]
        assert crc == zlib.crc32(tag + body) & 0xFFFFFFFF
        if tag == b"IDAT":
            idat += body
        chunks[tag] = body
        pos += 12 + n
    w, h, depth, ctype = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    ch = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    bpp = ch * depth // 8
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, w * bpp + 1)
    assert np.all(raw[:, 0] == 1)
    rows = raw[:, 1:].copy()
    for i in range(bpp, rows.shape[1]):
        rows[:, i] = rows[:, i] + rows[:, i - bpp]
    if depth == 16:
        return rows.view(">u2").reshape(h, w, ch).astype(np.uint16)
    return rows.reshape(h, w, ch)


@pytest.mark.parametrize("dtype,ch", [(np.uint8, 1), (np.uint8, 3), (np.uint16, 3), (np.uint16, 1), (np.uint8, 4)])
def test_png_roundtrip(dtype, ch):
    rng = np.random.default_rng(0)
    hi = 65535 if dtype == np.uint16 else 255
    a = rng.integers(0, hi + 1, size=(17, 23, ch)).astype(dtype)
    b = _decode_png(encode_png(a))
    assert np.array_equal(a, b)
