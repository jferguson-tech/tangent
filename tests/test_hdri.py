import io

import numpy as np
import pytest

from render import hdri


# ---- helpers: a reference RGBE encoder (flat and new-style RLE) -------------
def _to_rgbe(img):
    v = img.max(axis=-1)
    mant, exp = np.frexp(v)
    scale = np.where(v > 1e-32, mant * 256.0 / np.where(v > 0, v, 1), 0)
    rgbe = np.zeros(img.shape[:2] + (4,), np.uint8)
    rgbe[..., :3] = np.clip(np.floor(img * scale[..., None]), 0, 255)
    rgbe[..., 3] = np.where(v > 1e-32, exp + 128, 0)
    return rgbe


def _rle_channel(row):
    out = bytearray()
    i, n = 0, len(row)
    while i < n:
        j = i
        while j < n and j - i < 127 and row[j] == row[i]:
            j += 1
        if j - i >= 3:
            out += bytes([128 + (j - i), row[i]])
            i = j
            continue
        j = i
        while j < n and j - i < 128 and not (j + 2 < n and row[j] == row[j + 1] == row[j + 2]):
            j += 1
        out += bytes([j - i]) + bytes(row[i:j])
        i = j
    return bytes(out)


def _encode_hdr(img, rle=True):
    h, w = img.shape[:2]
    rgbe = _to_rgbe(img)
    out = io.BytesIO()
    out.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode())
    for y in range(h):
        if rle:
            out.write(bytes([2, 2, w >> 8, w & 255]))
            for c in range(4):
                out.write(_rle_channel(rgbe[y, :, c].tolist()))
        else:
            out.write(rgbe[y].tobytes())
    return out.getvalue()


@pytest.mark.parametrize("rle", [True, False])
def test_read_hdr_roundtrip(rle):
    rng = np.random.default_rng(1)
    img = (rng.random((12, 40, 3)) ** 4 * 50).astype(np.float32)
    img[3, :] = 2.0               # long runs exercise the run-length path
    img[5, 7] = 0.0
    back = hdri.read_hdr(_encode_hdr(img, rle))
    assert back.shape == img.shape
    # RGBE shares one exponent per pixel: error is relative to the brightest channel.
    tol = img.max(axis=-1, keepdims=True) / 128 + 1e-6
    assert np.all(np.abs(back - img) <= tol)
    assert back[5, 7].max() == 0.0


@pytest.mark.parametrize("data", [b"not an hdr", b"#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n\0\0\0\0",
                                  b"#?RADIANCE\n\n+Y 2 +X 2\n" + bytes(16),
                                  b"#?RADIANCE\n\n-Y 4 +X 16\n" + bytes([2, 2, 0, 16, 200])])
def test_read_hdr_rejects_bad_files(data):
    with pytest.raises(ValueError):
        hdri.read_hdr(data)


def _lum(img):
    return img @ np.array([0.2126, 0.7152, 0.0722])


def test_builtins_are_finite_and_oriented():
    for key in hdri.BUILTINS:
        img = hdri.load_image(key)
        assert img.shape == (hdri.GEN_H, hdri.GEN_W, 3)
        assert np.isfinite(img).all() and img.min() >= 0
    hangar = _lum(hdri.load_image("dark_hangar"))
    h = hangar.shape[0]
    assert hangar[: h // 2].mean() > 5 * hangar[h // 2:].mean()   # lights are overhead
    # Studio key softbox centered at azimuth 40 deg, elevation 35 deg.
    studio = _lum(hdri.load_image("studio"))
    H, W = studio.shape
    v = int((90 - 35) / 180 * H)
    u = int((0.5 + np.deg2rad(40) / (2 * np.pi)) * W)
    assert studio[v, u] > 4.0


def test_dark_hangar_stays_dark():
    em = hdri.EnvMap(hdri.load_image("dark_hangar"))
    assert 0.01 < em.mean < 0.08


def test_envmap_sampling_tables():
    img = np.zeros((32, 64, 3), np.float32)
    img[4:6, 10:12] = 100.0
    img += 0.01
    em = hdri.EnvMap(img)
    h, w = em.pdf.shape
    assert em.marg[0] == 0 and em.marg[-1] == 1 and np.all(np.diff(em.marg) >= 0)
    assert np.all(np.diff(em.cond, axis=1) >= 0) and np.allclose(em.cond[:, -1], 1)
    assert em.pdf.sum() / (h * w) == pytest.approx(1.0)
    # The solid-angle pdf integrates to 1 over the sphere.
    theta = (np.arange(h) + 0.5) / h * np.pi
    dw = (np.pi / h) * (2 * np.pi / w) * np.sin(theta)[:, None]
    pdf_sa = em.pdf / (2 * np.pi ** 2 * np.sin(theta)[:, None])
    assert (pdf_sa * dw).sum() == pytest.approx(1.0, rel=1e-6)


def test_black_environment_disables_sampling():
    em = hdri.EnvMap(np.zeros((8, 16, 3), np.float32))
    assert not em.enabled


def test_gradient_matches_old_preset_gradient():
    g = hdri.gradient_image([[0.1, 0.1, 0.1], [0.02, 0.02, 0.02], [0, 0, 0]], 32, 64)
    assert g[0, 0, 0] == pytest.approx(0.1, abs=0.01)     # zenith = top color
    assert g[-1, 0, 0] == pytest.approx(0.0, abs=1e-6)    # nadir = bottom color


def test_prefilter_levels_shapes_and_energy():
    img = np.full((64, 128, 3), 0.5, np.float32)
    levels, irr = hdri.prefilter_for_gl(img)
    assert [lv.shape[:2] for lv in levels] == [(h, w) for _, h, w in hdri.GL_LEVELS]
    for lv in levels + [irr]:
        assert np.allclose(lv, 0.5, atol=1e-3)   # convolving a constant keeps it constant


# ---- pathtracer furnace tests (need Numba) --------------------------------
def _furnace(env_img, metallic, rough, nee, spp):
    pt = pytest.importorskip("render.cpu_pathtracer")
    Hm = np.zeros((16, 16))
    Nm = np.zeros((16, 16, 3))
    Nm[..., 2] = 1
    Mt = np.zeros((16, 16, 5))
    Mt[..., :3] = 1.0
    Mt[..., 3] = metallic
    Mt[..., 4] = rough
    sc = np.array([-50, 50, -50, 50, 0, 0, 100, 100, 100 / 16, 0, 0, 0, 1, 0, 1e-4], np.float64)
    cam = np.array([0, 0, 2, 0, 0, -1, 1, 0, 0, 0, 1, 0, 0.3, 1.0])
    mat = np.array([1, 1, 1, metallic, rough, 0.03, 0.03, 0.03, 0.6], np.float64)
    em = hdri.EnvMap(env_img)
    ep = np.array([1.0, 0.3, 0.0, nee])
    opt = np.array([1.0, 0.0, 0.0])
    acc = np.zeros((24, 24, 3))
    for _ in range(spp):
        pt.render_pass(acc, Hm, Nm, Mt, sc, cam, mat, np.zeros((0, 14)), em.img, em.blur,
                       em.pdf, em.marg, em.cond, ep, np.zeros((4, 4, 3)), np.zeros((4, 4)),
                       np.array([0.0, 1.0]), np.zeros(1, np.int64), np.array([0.0, 1.0, 0.0]), opt)
    return (acc / spp).mean()


def test_white_furnace_conserves_energy():
    white = np.ones((32, 64, 3), np.float32)
    assert _furnace(white, 0.0, 1.0, 1.0, 64) == pytest.approx(0.97, abs=0.03)
    assert _furnace(white, 1.0, 0.3, 1.0, 64) == pytest.approx(1.0, abs=0.03)


def test_env_sampling_is_unbiased():
    sun = np.full((32, 64, 3), 0.2, np.float32)
    sun[5:7, 20:22] = 300.0
    with_mis = _furnace(sun, 0.0, 1.0, 1.0, 256)
    bsdf_only = _furnace(sun, 0.0, 1.0, 0.0, 1024)
    assert with_mis == pytest.approx(bsdf_only, rel=0.08)


# ---- endpoints ---------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(hdri, "HDRI_DIR", tmp_path)
    monkeypatch.setattr(hdri, "CACHE_DIR", tmp_path / ".cache")
    import app as app_module
    return app_module.app.test_client()


def test_env_endpoints_and_upload(client, tmp_path):
    ids = [e["id"] for e in client.get("/api/env").get_json()["environments"]]
    assert ids[:4] == ["dark_hangar", "studio", "neon_night", "overcast"] and "gradient" in ids
    data = _encode_hdr(np.full((8, 16, 3), 0.25, np.float32))
    r = client.post("/api/env/upload", data={"file": (io.BytesIO(data), "../../My Sky!.hdr")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    env_id = r.get_json()["id"]
    assert env_id == "file:My_Sky.hdr" and (tmp_path / "My_Sky.hdr").is_file()
    assert env_id in [e["id"] for e in client.get("/api/env").get_json()["environments"]]
    assert client.get(f"/api/env/thumb?id={env_id}").status_code == 200
    gl = client.get(f"/api/env/gl?id={env_id}").get_json()
    assert len(gl["levels"]) == 5 and gl["irradiance"]["w"] == 32
    bad = client.post("/api/env/upload", data={"file": (io.BytesIO(b"nope"), "x.hdr")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    assert client.get("/api/env/thumb?id=file:missing.hdr").status_code == 404


def test_render_defaults_use_dark_hangar_with_visible_background(client):
    m = client.get("/api/meta").get_json()
    env = m["render_defaults"]["environment"]
    assert env["hdri"] == "dark_hangar" and env["background"] == "hdri"


def test_scene_builds_for_every_environment():
    from core.document import Document
    from core.panel import Panel
    from render.scene import build_scene
    doc = Document(canvas_w=0.5, canvas_h=0.5, panels=[Panel(x=0.1, y=0.1, w=0.3, h=0.3)])
    for env_id in list(hdri.BUILTINS) + ["gradient", "file:does-not-exist.hdr"]:
        sc = build_scene(doc, {"environment": {"hdri": env_id, "background": "blurred"}})
        assert sc["E"].shape[2] == 3 and sc["ep"][2] == 2 and sc["P"].shape == sc["E"].shape[:2]
