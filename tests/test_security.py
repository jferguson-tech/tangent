import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import app as app_module
from render import hdri

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def c():
    app_module.app.testing = True
    return app_module.app.test_client()


DOC = {"doc": {"canvas_w": 0.5, "canvas_h": 0.5, "texel_density": 128, "panels": []}, "maps": ["normal"], "max_res": 64}


# ---- 2. real JSON only, same-origin writes, allowed hosts ---------------------------
@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded", None])
def test_non_json_bodies_are_refused(c, ctype):
    import json
    headers = {"Content-Type": ctype} if ctype else {}
    r = c.post("/api/bake", data=json.dumps(DOC), headers=headers)
    assert r.status_code == 415
    assert "application/json" in r.get_json()["error"]


def test_json_body_must_be_an_object(c):
    assert c.post("/api/bake", json=[1, 2]).status_code == 400
    assert c.post("/api/profile", data="{oops", content_type="application/json").status_code == 400


def test_real_json_works(c):
    assert c.post("/api/bake", json=DOC).status_code == 200


@pytest.mark.parametrize("host", ["localhost", "localhost:5000", "127.0.0.1:5000", "192.168.1.20:8080",
                                  "10.0.0.5", "[::1]:5000", "[fe80::1]:8080"])
def test_ip_and_local_hosts_are_allowed(c, host):
    assert c.get("/api/env", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example.com", "evil.example.com:5000", "attacker.localhost.evil.com"])
def test_other_host_names_are_refused(c, host):
    """A DNS-rebinding page reaches the server with its own domain as the Host."""
    r = c.get("/api/meta", headers={"Host": host})
    assert r.status_code == 403
    assert c.get("/", headers={"Host": host}).status_code == 403


def test_machine_name_and_allow_host_extras(c, monkeypatch):
    import socket
    name = socket.gethostname().lower()
    assert c.get("/api/env", headers={"Host": f"{name}:5000"}).status_code == 200
    monkeypatch.setattr(app_module.GUARD, "allowed", app_module.GUARD.allowed | {"tangent.lan"})
    assert c.get("/api/env", headers={"Host": "tangent.lan:5000"}).status_code == 200


def test_cross_site_writes_are_refused(c):
    h = {"Host": "192.168.1.20:5000"}
    ok = c.post("/api/bake", json=DOC, headers={**h, "Origin": "http://192.168.1.20:5000"})
    assert ok.status_code == 200
    for bad in ({"Origin": "https://evil.example.com"}, {"Origin": "http://192.168.1.20:6000"},
                {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}):
        assert c.post("/api/bake", json=DOC, headers={**h, **bad}).status_code == 403, bad
    # Uploads are multipart, so the origin check is what protects them.
    r = c.post("/api/env/upload", data={"file": (io.BytesIO(b"x"), "a.hdr")},
               headers={**h, "Origin": "https://evil.example.com"}, content_type="multipart/form-data")
    assert r.status_code == 403
    # Reads stay open to any allowed host; scripts without Origin can still post.
    assert c.get("/api/meta", headers={**h, "Origin": "https://evil.example.com"}).status_code == 200
    assert c.post("/api/bake", json=DOC, headers=h).status_code == 200


# ---- 3. debug only on a local bind -----------------------------------------------------
@pytest.mark.parametrize("host,refused", [("0.0.0.0", True), ("192.168.1.20", True), ("127.0.0.1", False)])
def test_debug_is_refused_on_network_hosts(host, refused):
    code = (
        "import sys, app\n"
        "app.app.run = lambda **kw: print('WOULD RUN', kw['host'], kw['debug'])\n"
        f"sys.argv = ['app.py', '--debug', '--host', '{host}']\n"
        "app.main()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    if refused:
        assert out.returncode != 0 and "Refusing to start" in out.stderr
    else:
        assert out.returncode == 0 and "WOULD RUN 127.0.0.1 True" in out.stdout


def test_network_host_without_debug_still_starts():
    code = (
        "import sys, app\n"
        "app.app.run = lambda **kw: print('WOULD RUN', kw['host'], kw['debug'])\n"
        "sys.argv = ['app.py', '--host', '0.0.0.0', '--allow-host', 'tangent.lan']\n"
        "app.main()\n"
        "print('tangent.lan' in app.GUARD.allowed)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and "WOULD RUN 0.0.0.0 False" in out.stdout and "True" in out.stdout


# ---- 4. upload folder limits -------------------------------------------------------------
def _hdr(n=8):
    import sys as _s
    _s.path.insert(0, str(ROOT / "tests"))
    from test_hdri import _encode_hdr
    return _encode_hdr(np.full((n, 2 * n, 3), 0.5, np.float32))


def test_upload_count_and_size_limits(c, tmp_path, monkeypatch):
    monkeypatch.setattr(hdri, "HDRI_DIR", tmp_path)
    monkeypatch.setattr(hdri, "CACHE_DIR", tmp_path / ".cache")
    monkeypatch.setattr(hdri, "MAX_UPLOAD_FILES", 2)
    data = _hdr()
    up = lambda name: c.post("/api/env/upload", data={"file": (io.BytesIO(data), name)},
                             content_type="multipart/form-data")
    assert up("one.hdr").status_code == 200
    assert up("two.hdr").status_code == 200
    r = up("three.hdr")
    assert r.status_code == 413 and "limit 2" in r.get_json()["error"]
    assert up("two.hdr").status_code == 200, "replacing an existing file is still allowed"
    monkeypatch.setattr(hdri, "MAX_UPLOAD_FILES", 50)
    monkeypatch.setattr(hdri, "MAX_UPLOAD_TOTAL", len(data) * 2 + 10)
    r = up("four.hdr")
    assert r.status_code == 413 and "exceed" in r.get_json()["error"]
    assert sorted(f.name for f in tmp_path.glob("*.hdr")) == ["one.hdr", "two.hdr"]
