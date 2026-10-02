"""Tangent: sci-fi panel normal map tool (Flask server).

Run with the project venv:
    _env\\Scripts\\python.exe app.py                  (this computer: http://127.0.0.1:5000)
    _env\\Scripts\\python.exe app.py --host 0.0.0.0   (other devices on your network)

See the Security section of README.md before serving beyond localhost.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import zipfile

import numpy as np

from flask import Flask, Response, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from core import bake as bakemod, brush, fluids, layout, profiles, route, weather, wire as wiremod
from core.document import Document
from core.panel import ANCHORS, CORNER_STYLES, DETAIL_MODES, DETAIL_SHAPES, PANEL_MODES
from core.png import encode_png
from core.units import EMISSION_UNIT, MAX_RESOLUTION, MIN_BEVEL_PX, TEXEL_DENSITY_PRESETS
from render import hdri
from render.presets import DEFAULT_RENDER, LIGHT_PRESETS, MATERIAL_PRESETS, WIRE_MATERIAL_PRESETS
from render.scene import ENV_CACHE
from security import RequestGuard, is_local_bind

app = Flask(__name__)
GUARD = RequestGuard()
app.before_request(GUARD.check)


@app.errorhandler(HTTPException)
def _http_error(e):
    """API errors come back as JSON the page can show."""
    if request.path.startswith("/api/"):
        return jsonify({"error": e.description or e.name}), e.code
    return e
app.config["MAX_CONTENT_LENGTH"] = hdri.MAX_HDR_BYTES + 1024 * 1024
hdri.HDRI_DIR.mkdir(exist_ok=True)

# The pathtracer is optional: everything else works without Numba.
try:
    from render.cpu import CPURenderer
    from render.manager import RenderManager
    RENDER = RenderManager(CPURenderer())
    RENDER_ERROR = None
except Exception as exc:  # ImportError when Numba is missing, or LLVM issues
    RENDER = None
    RENDER_ERROR = f"Pathtracer unavailable: {exc}"

DETAIL_PRESETS = {
    "bolts": {"kind": "bolt", "shape": "circle", "anchor": "corners", "ox": 0.03, "oy": 0.03,
              "w": 0.016, "h": 0.016, "depth": 0.003, "mode": "raise",
              "bevel": {"width": 0.0056, "profile": "round"}},
    "hex_bolts": {"kind": "hex bolt", "shape": "hex", "anchor": "corners", "ox": 0.03, "oy": 0.03,
                  "w": 0.018, "h": 0.018, "depth": 0.003, "mode": "raise",
                  "bevel": {"width": 0.003, "profile": "linear"}},
    "vent": {"kind": "vent", "shape": "rect", "anchor": "c", "ox": 0, "oy": 0, "w": 0.12,
             "h": 0.012, "radius": 0.006, "count": 5, "spacing": 0.026, "depth": 0.004,
             "mode": "inset", "bevel": {"width": 0.003, "profile": "linear"}},
    "cutout": {"kind": "cutout", "shape": "rect", "anchor": "tr", "ox": 0.08, "oy": 0.06,
               "w": 0.08, "h": 0.05, "radius": 0.006, "depth": 0.004, "mode": "inset",
               "bevel": {"width": 0.004, "profile": "linear"}},
    "groove": {"kind": "groove", "shape": "rect", "anchor": "t", "ox": 0, "oy": 0.045,
               "w": 0.25, "h": 0.006, "radius": 0.003, "depth": 0.002, "mode": "inset",
               "bevel": {"width": 0.002, "profile": "round"}},
    "light": {"kind": "light strip", "shape": "rect", "anchor": "b", "ox": 0, "oy": 0.04,
              "w": 0.2, "h": 0.014, "radius": 0.007, "depth": 0.002, "mode": "raise",
              "bevel": {"width": 0.004, "profile": "smooth"},
              "emission": {"enabled": True, "color": [0.35, 0.85, 1.0], "strength": 4.0}},
    "indicator": {"kind": "indicator", "shape": "circle", "anchor": "tr", "ox": 0.07, "oy": 0.03,
                  "w": 0.008, "h": 0.008, "rotation": 90, "count": 3, "spacing": 0.016,
                  "depth": 0.0015, "mode": "raise", "bevel": {"width": 0.003, "profile": "round"},
                  "emission": {"enabled": True, "color": [1.0, 0.62, 0.18], "strength": 6.0}},
}

# Strength presets for the emission UI (relative units, see core.units.EMISSION_UNIT).
EMISSION_PRESETS = {"Indicator": 6.0, "Strip": 4.0, "Light panel": 1.6}


def _json_body():
    """The request's JSON object. Only a real application/json body is
    accepted: browsers cannot send that cross-site without a CORS preflight,
    which this server never grants."""
    if not request.is_json:
        abort(415, description="Send the request body as application/json.")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        abort(400, description="The request body must be a JSON object.")
    return body


def _doc_from_request():
    body = _json_body()
    return Document.from_dict(body.get("doc") or {}), body


def _data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def _default_document():
    doc = Document()
    params = dict(layout.DEFAULT_PARAMS)
    doc.panels = layout.generate(doc, params)
    doc.wires = layout.generate_wires(doc, doc.panels, params)
    layout.place_new_sim_wires(doc)
    route.route_all(doc)
    doc.layout = params
    return doc


@app.get("/")
def index():
    return render_template("index.html")


FAVICON = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" rx="3" '
           'fill="#12161c"/><path d="M3 12 8 4l5 8z" fill="none" stroke="#39c6e6" stroke-width="1.6"/></svg>')


@app.get("/favicon.ico")
def favicon():
    return Response(FAVICON, mimetype="image/svg+xml")


@app.get("/api/meta")
def meta():
    return jsonify({
        "texel_presets": TEXEL_DENSITY_PRESETS,
        "min_bevel_px": MIN_BEVEL_PX,
        "max_resolution": MAX_RESOLUTION,
        "profiles": {name: profiles.sample(name) for name in profiles.PROFILES},
        "panel_modes": PANEL_MODES, "corner_styles": CORNER_STYLES,
        "detail_shapes": DETAIL_SHAPES, "detail_modes": DETAIL_MODES, "anchors": ANCHORS,
        "detail_presets": DETAIL_PRESETS,
        "layout_defaults": layout.DEFAULT_PARAMS,
        "light_presets": LIGHT_PRESETS,
        "material_presets": MATERIAL_PRESETS,
        "wire_material_presets": WIRE_MATERIAL_PRESETS,
        "weathering_presets": weather.PRESETS,
        "default_materials": weather.DEFAULT_MATERIALS,
        "default_weathering": weather.DEFAULT_WEATHERING,
        "max_auto_leaks": weather.MAX_AUTO_LEAKS,
        "liquids": {k: {"label": v["label"], "color": v["color"]} for k, v in fluids.LIQUIDS.items()},
        "brush_channels": brush.CHANNEL_LABELS,
        "mask_maps": list(bakemod.MASK_MAPS),
        "render_defaults": DEFAULT_RENDER,
        "environments": hdri.list_environments(),
        "backgrounds": {"hdri": "HDRI", "black": "Black", "blurred": "Blurred HDRI"},
        "emission_presets": EMISSION_PRESETS,
        "light_colors": layout.LIGHT_COLORS,
        "light_palettes": ["mixed", *layout.LIGHT_COLORS],
        "wire_profiles": list(wiremod.WIRE_PROFILES),
        "wire_anchors": list(wiremod.END_ANCHORS),
        "pathtracer": {"available": RENDER is not None, "error": RENDER_ERROR},
        "default_document": _default_document().to_dict(),
    })


@app.post("/api/profile")
def profile_curve():
    body = _json_body()
    return jsonify(profiles.sample(body.get("profile", "linear"), 48, body.get("points")))


@app.post("/api/bake")
def bake():
    doc, body = _doc_from_request()
    if body.get("region"):
        return bake_detail(doc, body)
    maps = [m for m in body.get("maps", ["normal"]) if m in bakemod.PREVIEW_MAPS] or ["normal"]
    max_res = int(body.get("max_res") or 1024)
    ss = int(body.get("ss") or 1)
    res = bakemod.bake(doc, maps=maps, max_res=max_res, ss=ss, weathering=not body.get("fast"))
    out = {}
    for m in maps:
        if m == "height_raw" and m in res.maps:
            hr = np.ascontiguousarray(res.maps[m], np.float16)   # one channel, meters
            out[m] = {"w": int(hr.shape[1]), "h": int(hr.shape[0]),
                      "data": base64.b64encode(hr.tobytes()).decode("ascii")}
        elif m in res.maps:
            out[m] = _data_url(encode_png(res.encode(m, 8, "gl"), level=1))
    # Centerlines of every visible wire (routed and simulated), for the overlay.
    paths = {}
    for w in doc.wires:
        if w.visible:
            P = wiremod.resolved_path(w, doc)
            paths[w.id] = np.round(P, 5).tolist()
    return jsonify({"maps": out, "info": res.info, "warnings": doc.warnings(), "wire_paths": paths})


def bake_detail(doc, body):
    """A rectangle of the canvas at full density, for the zoomed-in 2D view."""
    reg = body["region"]
    norm = body.get("norm") if isinstance(body.get("norm"), dict) else None
    maps = [m for m in body.get("maps", bakemod.DETAIL_MAPS) if m in bakemod.DETAIL_MAPS] or ["normal"]
    try:
        x0, y0, x1, y1 = (float(reg[k]) for k in ("x0", "y0", "x1", "y1"))
        ppm = float(reg.get("px_per_m") or doc.texel_density)
        if not (x1 > x0 and y1 > y0 and 1 <= ppm <= 16384):
            raise ValueError("bad region")
        res = bakemod.bake_region(doc, x0, y0, x1, y1, min(ppm, doc.texel_density), maps, norm)
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400
    out = {m: _data_url(encode_png(res.encode(m, 8, "gl"), level=1)) for m in maps if m in res.maps}
    return jsonify({"maps": out, "info": res.info})


@app.post("/api/layout")
def auto_layout():
    doc, body = _doc_from_request()
    params = layout.params_from(body.get("params"))
    # Locked wires survive; ends on panels that are about to be regenerated
    # become free points where they are now.
    keep = [w for w in doc.wires if w.locked]
    new_panels = layout.generate(doc, params)
    ids = {q.id for q in new_panels}
    for w in keep:
        for end in (w.a, w.b):
            end.x, end.y = end.resolve(doc)
            if end.panel and end.panel not in ids:
                end.panel = None
    doc.panels = new_panels
    doc.wires = layout.generate_wires(doc, [q for q in new_panels if not q.locked], params, keep)
    layout.place_new_sim_wires(doc)
    route.route_all(doc)
    doc.layout = params
    return jsonify({"doc": doc.to_dict()})


@app.post("/api/layout/wires")
def shuffle_wires():
    """Regenerate only the wires (all kinds) on the current panels. Locked wires stay."""
    doc, body = _doc_from_request()
    params = layout.params_from(body.get("params"))
    if params["wire_density"] <= 0:          # an explicit wire shuffle always makes some
        params["wire_density"] = layout.WIRE_PARAMS["wire_density"]
    keep = [w for w in doc.wires if w.locked]
    doc.wires = layout.generate_wires(doc, doc.panels, params, keep)
    layout.place_new_sim_wires(doc)
    route.route_all(doc)
    doc.layout = dict(doc.layout or {}, wire_seed=params["wire_seed"])
    return jsonify({"doc": doc.to_dict()})


@app.post("/api/export")
def export():
    doc, body = _doc_from_request()
    maps = [m for m in body.get("maps", ["normal"]) if m in bakemod.ALL_MAPS] or ["normal"]
    bits = 16 if int(body.get("bits", 8)) == 16 else 8
    nx, ny = doc.resolution()
    ss = int(body.get("ss") or (2 if max(nx, ny) <= 2048 else 1))
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(body.get("name") or "panel"))[:64] or "panel"
    res = bakemod.bake(doc, maps=maps, ss=ss)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for m in maps:
            suffix = f"normal_{doc.normal_convention}" if m == "normal" else m
            b = 8 if m in ("id", "emissive") else bits
            z.writestr(f"{name}_{suffix}.png", encode_png(res.encode(m, b, doc.normal_convention)))
        z.writestr(f"{name}.tangent.json", json.dumps(doc.to_dict(), indent=1))
        info = res.info
        z.writestr(f"{name}_info.txt", "\n".join([
            f"Resolution: {info['width']} x {info['height']} px",
            f"Canvas: {doc.canvas_w:g} m x {doc.canvas_h:g} m",
            f"Texel density: {doc.texel_density:g} px/m",
            f"Normal convention: {'OpenGL (Y+)' if doc.normal_convention == 'gl' else 'DirectX (Y-)'}",
            f"Height range: {info['height_min'] * 1000:.3f} mm to {info['height_max'] * 1000:.3f} mm "
            f"(black to white in the height map)",
            f"Supersampling: {ss}x",
            f"Bit depth: {bits} (ID mask and emissive are 8-bit)",
            *([f"Emissive: sRGB color, white = {info['emissive_scale']:.4g} linear radiance "
               f"= strength {info['emissive_scale'] / EMISSION_UNIT:.4g}. "
               f"Use that value as the emissive intensity in your engine."]
              if info.get("has_emission") else
              (["Emissive: no emitters in this document (map is black)."] if "emissive" in maps else [])),
        ]) + "\n")
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True,
                     download_name=f"{name}_maps.zip")


# ---- environments (HDRI) -------------------------------------------------
def _env_args():
    env_id = request.args.get("id") or hdri.DEFAULT_ENV
    preset = LIGHT_PRESETS.get(request.args.get("preset"), LIGHT_PRESETS["scifi_spot"])
    return env_id, preset["env"]


def _f16(img):
    rgba = np.concatenate([img, np.ones(img.shape[:2] + (1,), img.dtype)], axis=-1)
    return {"w": int(img.shape[1]), "h": int(img.shape[0]),
            "data": base64.b64encode(np.ascontiguousarray(rgba, np.float16).tobytes()).decode("ascii")}


@app.get("/api/env")
def env_list():
    return jsonify({"environments": hdri.list_environments(), "default": hdri.DEFAULT_ENV})


@app.get("/api/env/thumb")
def env_thumb():
    env_id, gradient = _env_args()
    try:
        img = ENV_CACHE.image(env_id, gradient)
    except (KeyError, ValueError, OSError):
        return jsonify({"error": f"unknown environment {env_id}"}), 404
    resp = Response(encode_png(hdri.thumbnail(img)), mimetype="image/png")
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/api/env/gl")
def env_gl():
    """Prefiltered roughness levels and irradiance for the editor's WebGL lighting."""
    env_id, gradient = _env_args()
    try:
        levels, irr = ENV_CACHE.gl(env_id, gradient)
    except (KeyError, ValueError, OSError):
        return jsonify({"error": f"unknown environment {env_id}"}), 404
    return jsonify({"id": env_id,
                    "levels": [dict(_f16(lv), rough=r) for lv, (r, _, _) in zip(levels, hdri.GL_LEVELS)],
                    "irradiance": _f16(irr)})


@app.post("/api/env/upload")
def env_upload():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file"}), 400
    try:
        env_id = hdri.save_upload(f.filename, f.read())
    except hdri.UploadLimit as e:
        return jsonify({"error": f"Upload refused: {e}"}), 413
    except ValueError as e:
        return jsonify({"error": f"not a usable .hdr file: {e}"}), 400
    return jsonify({"id": env_id, "environments": hdri.list_environments()})


# ---- pathtraced preview ---------------------------------------------------
def _no_renderer():
    return jsonify({"error": RENDER_ERROR or "Pathtracer unavailable"}), 503


@app.post("/api/render/start")
def render_start():
    if RENDER is None:
        return _no_renderer()
    doc, body = _doc_from_request()
    job = RENDER.start(doc, body.get("settings") or {})
    return jsonify({"job": job})


@app.post("/api/render/stop")
def render_stop():
    if RENDER is None:
        return _no_renderer()
    RENDER.stop()
    return jsonify({"ok": True})


@app.get("/api/render/status")
def render_status():
    if RENDER is None:
        return jsonify({"state": "unavailable", "error": RENDER_ERROR})
    return jsonify(RENDER.status())


@app.get("/api/render/frame")
def render_frame():
    if RENDER is None:
        return _no_renderer()
    after = int(request.args.get("after", -1))
    f = RENDER.frame(after)
    if f is None:
        return Response(status=204)
    version, data, info = f
    resp = Response(data, mimetype="image/png")
    resp.headers["X-Frame-Version"] = str(version)
    resp.headers["X-Frame-Info"] = json.dumps(info)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def main():
    ap = argparse.ArgumentParser(description="Tangent normal map tool")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true",
                    help="Flask debug mode (local only: its debugger can run code)")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra host name to accept (IP addresses and this machine's name always work)")
    a = ap.parse_args()
    if a.debug and not is_local_bind(a.host):
        sys.exit("Refusing to start: --debug enables an interactive debugger that can run code, "
                 f"so it is only allowed with a local --host (127.0.0.1 or localhost), not {a.host}.")
    GUARD.allowed |= {h.strip().lower() for h in a.allow_host if h.strip()}
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
