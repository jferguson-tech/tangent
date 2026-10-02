import io
import json
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def client():
    import app as app_module
    app_module.app.testing = True
    return app_module.app.test_client(), app_module


def test_meta(client):
    c, _ = client
    m = c.get("/api/meta").get_json()
    assert m["texel_presets"] == [256.0, 512.0, 1024.0, 2048.0]
    assert m["default_document"]["texel_density"] == 512.0
    assert m["default_document"]["tiling"] is False
    assert m["render_defaults"]["displacement"] is True
    assert m["render_defaults"]["light"]["preset"] == "scifi_spot"
    assert "custom" in m["profiles"]


def test_bake_and_layout(client):
    c, _ = client
    doc = c.get("/api/meta").get_json()["default_document"]
    r = c.post("/api/bake", json={"doc": doc, "maps": ["normal", "ao"], "max_res": 256}).get_json()
    assert r["maps"]["normal"].startswith("data:image/png;base64,")
    assert r["info"]["width"] == 256
    doc["panels"][0]["locked"] = True
    out = c.post("/api/layout", json={"doc": doc, "params": {"seed": 99}}).get_json()["doc"]
    assert out["panels"][0]["id"] == doc["panels"][0]["id"]


def test_export_zip(client):
    c, _ = client
    doc = {"canvas_w": 0.5, "canvas_h": 0.25, "texel_density": 256, "normal_convention": "dx",
           "panels": [{"x": 0.05, "y": 0.05, "w": 0.3, "h": 0.15}]}
    r = c.post("/api/export", json={"doc": doc, "maps": ["normal", "height", "id"], "bits": 16, "name": "t"})
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.data))
    names = set(z.namelist())
    assert {"t_normal_dx.png", "t_height.png", "t_id.png", "t.tangent.json", "t_info.txt"} <= names
    assert z.read("t_normal_dx.png")[24] == 16       # bit depth byte in IHDR
    assert z.read("t_id.png")[24] == 8
    assert "128 x 64 px" in z.read("t_info.txt").decode()


def test_render_progressive_frames(client):
    c, mod = client
    if mod.RENDER is None:
        pytest.skip(mod.RENDER_ERROR)
    doc = c.get("/api/meta").get_json()["default_document"]
    job = c.post("/api/render/start", json={"doc": doc, "settings": {"width": 96, "height": 64, "max_spp": 6}}).get_json()["job"]
    deadline = time.time() + 120
    info = None
    while time.time() < deadline:
        r = c.get("/api/render/frame?after=-1")
        if r.status_code == 200:
            info = json.loads(r.headers["X-Frame-Info"])
            if info["job"] == job and info["done"]:
                break
        time.sleep(0.2)
    assert info and info["done"] and info["spp"] == 6
    assert r.data[:4] == b"\x89PNG"


def test_editor_and_export_work_without_numba():
    """Block the numba import: the app must start and bake, the renderer must report unavailable."""
    code = (
        "import sys; sys.modules['numba'] = None\n"
        "import app\n"
        "assert app.RENDER is None and 'numba' in app.RENDER_ERROR.lower(), app.RENDER_ERROR\n"
        "c = app.app.test_client()\n"
        "doc = c.get('/api/meta').get_json()['default_document']\n"
        "assert c.post('/api/bake', json={'doc': doc, 'max_res': 128}).status_code == 200\n"
        "assert c.post('/api/export', json={'doc': doc, 'maps': ['normal'], 'ss': 1}).status_code == 200\n"
        "assert c.post('/api/render/start', json={'doc': doc}).status_code == 503\n"
        "print('ok')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("ok")
