"""Environment maps (HDRI) for image-based lighting. Pure NumPy, no Numba.

Conventions: equirectangular, world z up. Row 0 is the zenith. Column u maps
to azimuth phi = (u - 0.5) * 2pi, measured from +x toward +y, so
direction = (sin(theta) cos(phi), sin(theta) sin(phi), cos(theta)).

Sources:
  * built-in procedural HDRIs, generated on first use and cached as .npy
  * Radiance .hdr files dropped into the `hdri/` folder or uploaded
  * "gradient": the light preset's old sky/horizon/ground gradient
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HDRI_DIR = ROOT / "hdri"
CACHE_DIR = HDRI_DIR / ".cache"

RENDER_W, RENDER_H = 1024, 512       # resolution used by the pathtracer
BLUR_W, BLUR_H = 64, 32              # "blurred" background option
GEN_W, GEN_H = 2048, 1024            # procedural generation resolution
MAX_HDR_BYTES = 128 * 1024 * 1024


# ---------------------------------------------------------------- .hdr I/O
def read_hdr(src) -> np.ndarray:
    """Read a Radiance RGBE (.hdr) file from a path or bytes. Returns float32 (H, W, 3)."""
    data = src if isinstance(src, (bytes, bytearray)) else Path(src).read_bytes()
    if not (data.startswith(b"#?RADIANCE") or data.startswith(b"#?RGBE")):
        raise ValueError("not a Radiance .hdr file")
    pos = 0
    fmt_ok = False
    while True:
        end = data.find(b"\n", pos)
        if end < 0:
            raise ValueError("truncated header")
        line = data[pos:end].strip()
        pos = end + 1
        if not line:
            break
        if line.startswith(b"FORMAT="):
            fmt_ok = line == b"FORMAT=32-bit_rle_rgbe"
            if not fmt_ok:
                raise ValueError(f"unsupported format {line.decode(errors='replace')}")
    end = data.find(b"\n", pos)
    m = re.fullmatch(rb"-Y (\d+) \+X (\d+)", data[pos:end].strip())
    if not m:
        raise ValueError("unsupported orientation (only '-Y H +X W' is supported)")
    h, w = int(m.group(1)), int(m.group(2))
    if not (0 < w <= 32768 and 0 < h <= 16384):
        raise ValueError("unreasonable image size")
    pos = end + 1
    buf = np.frombuffer(data, np.uint8)
    out = np.empty((h, w, 4), np.uint8)
    for y in range(h):
        if (8 <= w < 0x8000 and pos + 4 <= len(data) and data[pos] == 2 and data[pos + 1] == 2
                and data[pos + 2] < 128 and (data[pos + 2] << 8 | data[pos + 3]) == w):
            pos += 4
            for c in range(4):
                x = 0
                while x < w:
                    if pos >= len(data):
                        raise ValueError("truncated scanline")
                    n = data[pos]
                    pos += 1
                    if n > 128:
                        n -= 128
                        if x + n > w:
                            raise ValueError("bad run length")
                        out[y, x:x + n, c] = data[pos]
                        pos += 1
                    else:
                        if n == 0 or x + n > w:
                            raise ValueError("bad literal length")
                        out[y, x:x + n, c] = buf[pos:pos + n]
                        pos += n
                    x += n
        else:  # flat scanline
            n = w * 4
            if pos + n > len(data):
                raise ValueError("truncated flat scanline")
            out[y] = buf[pos:pos + n].reshape(w, 4)
            pos += n
    e = out[..., 3].astype(np.int32)
    scale = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    rgb = (out[..., :3].astype(np.float32) + 0.5) * scale[..., None]
    return np.where(e[..., None] > 0, rgb, 0.0).astype(np.float32)


# ---------------------------------------------------------------- helpers
def _dirs(h, w):
    """Direction per pixel center, plus theta and phi grids."""
    theta = (np.arange(h) + 0.5) / h * np.pi
    phi = (np.arange(w) + 0.5) / w * 2 * np.pi - np.pi
    T, P = np.meshgrid(theta, phi, indexing="ij")
    st = np.sin(T)
    return np.stack([st * np.cos(P), st * np.sin(P), np.cos(T)], -1), T, P


def _downsample(img, h, w):
    H, W = img.shape[:2]
    if H % h == 0 and W % w == 0:
        return img.reshape(h, H // h, w, W // w, 3).mean(axis=(1, 3))
    ys = (np.arange(h) + 0.5) * H / h - 0.5
    xs = (np.arange(w) + 0.5) * W / w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, H - 1)
    x0 = np.floor(xs).astype(int)
    ty = (ys - y0)[:, None, None]
    tx = (xs - x0)[None, :, None]
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = (x0 + 1) % W
    x0 %= W
    a = img[y0][:, x0] * (1 - tx) + img[y0][:, x1] * tx
    b = img[y1][:, x0] * (1 - tx) + img[y1][:, x1] * tx
    return a * (1 - ty) + b * ty


def _resize(img, h, w):
    """Area-average down to (h, w), averaging in 2x steps for large reductions."""
    while img.shape[0] >= 2 * h and img.shape[1] >= 2 * w and img.shape[0] % 2 == 0 and img.shape[1] % 2 == 0:
        img = img.reshape(img.shape[0] // 2, 2, img.shape[1] // 2, 2, 3).mean(axis=(1, 3))
    if img.shape[:2] != (h, w):
        img = _downsample(img, h, w)
    return img


def _smooth_rect(a, lo, hi, soft):
    """1 inside [lo, hi], soft edges of width `soft`."""
    return np.clip((a - lo) / soft + 0.5, 0, 1) * np.clip((hi - a) / soft + 0.5, 0, 1)


def _box_lights(D, specs, out):
    """Add rectangular emitters defined in angular space around a center direction."""
    for az, el, half_w, half_h, rgb, soft in specs:
        a, e = np.deg2rad(az), np.deg2rad(el)
        c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        t = np.array([-np.sin(a), np.cos(a), 0.0])
        b = np.cross(c, t)
        dc = D @ c
        front = dc > 0
        x = np.degrees(np.arctan2(D @ t, np.maximum(dc, 1e-6)))
        y = np.degrees(np.arctan2(D @ b, np.maximum(dc, 1e-6)))
        m = _smooth_rect(x, -half_w, half_w, soft) * _smooth_rect(y, -half_h, half_h, soft) * front
        out += m[..., None] * np.array(rgb, np.float32)


# ---------------------------------------------------------------- procedural HDRIs
def gen_dark_hangar(h=GEN_H, w=GEN_W):
    """Interior of a dim hangar: cool ceiling light strips, dark ribbed walls,
    amber and cyan indicator lights, and a faint blue bay door."""
    D, T, P = _dirs(h, w)
    img = np.zeros((h, w, 3), np.float32)
    hx, hy, top, eye = 30.0, 18.0, 9.0, 1.6     # half extents, ceiling height, eye height
    with np.errstate(divide="ignore", invalid="ignore"):
        tx = np.where(np.abs(D[..., 0]) > 1e-6, hx / np.abs(D[..., 0]), np.inf)
        ty = np.where(np.abs(D[..., 1]) > 1e-6, hy / np.abs(D[..., 1]), np.inf)
        tz = np.where(D[..., 2] > 1e-6, (top - eye) / D[..., 2],
                      np.where(D[..., 2] < -1e-6, eye / -D[..., 2], np.inf))
    t = np.minimum(np.minimum(tx, ty), tz)
    p = D * t[..., None]
    z = p[..., 2] + eye
    ceil = (tz <= np.minimum(tx, ty)) & (D[..., 2] > 0)
    floor = (tz <= np.minimum(tx, ty)) & (D[..., 2] < 0)
    wall = ~(ceil | floor)
    wall_x = wall & (tx <= ty)

    # Ceiling: dark, with narrow light strips running along x every 6 m.
    cy = p[..., 1]
    strip = _smooth_rect(np.abs(((cy + 3.0) % 6.0) - 3.0), -1, 0.11, 0.03)
    strip *= _smooth_rect(p[..., 0], -24, 24, 0.3)
    gaps = _smooth_rect(np.abs(((p[..., 0] + 2.5) % 5.0) - 2.5), 0.9, 9, 0.05)  # segment breaks
    ceil_col = np.array([0.008, 0.009, 0.011], np.float32)
    strip_col = np.array([2.8, 3.0, 3.4], np.float32)
    img += (ceil[..., None] * (ceil_col + (strip * gaps)[..., None] * strip_col))

    # Walls: ribbed panels, horizontal band of light at the top, indicator lights.
    u = np.where(wall_x, p[..., 1], p[..., 0])
    rib = 0.6 + 0.4 * _smooth_rect(np.abs(((u + 1.5) % 3.0) - 1.5), 0.08, 9, 0.03)
    band = 0.6 + 0.4 * _smooth_rect(np.abs(((z + 1.25) % 2.5) - 1.25), 0.05, 9, 0.03)
    wall_col = np.array([0.010, 0.011, 0.013], np.float32) * (rib * band)[..., None]
    top_glow = _smooth_rect(z, top - 0.55, top - 0.45, 0.03)[..., None] * np.array([0.25, 0.3, 0.38], np.float32)
    amber = _smooth_rect(np.abs(((u + 4.5) % 9.0) - 4.5), -1, 0.12, 0.03) * _smooth_rect(z, 2.55, 2.75, 0.03)
    cyan = _smooth_rect(np.abs(((u + 2.0) % 7.0) - 3.5), -1, 0.5, 0.05) * _smooth_rect(z, 0.45, 0.52, 0.02)
    ind = amber[..., None] * np.array([3.0, 1.3, 0.25], np.float32) + cyan[..., None] * np.array([0.2, 1.6, 2.2], np.float32)
    # Bay door at the -y wall: a large dim blue opening.
    door = (wall & ~wall_x & (D[..., 1] < 0)) * _smooth_rect(p[..., 0], -9, 9, 0.2) * _smooth_rect(z, 0.0, 6.5, 0.2)
    door_col = np.array([0.02, 0.03, 0.05], np.float32) * (0.6 + 0.4 * np.clip(z / 6.5, 0, 1))[..., None]
    img += wall[..., None] * (wall_col + top_glow + ind) * (1 - door[..., None])
    img += door[..., None] * door_col

    # Floor: very dark, faint painted guide lines.
    f = np.abs(((p[..., 0] + 5.0) % 10.0) - 5.0)
    lines = _smooth_rect(f, -1, 0.06, 0.02)
    img += floor[..., None] * (np.array([0.006, 0.006, 0.007], np.float32)
                               + lines[..., None] * np.array([0.05, 0.04, 0.01], np.float32))
    return img


def gen_studio(h=GEN_H, w=GEN_W):
    """Neutral photo studio: gray backdrop, a large key softbox, fill and top strip."""
    D, T, P = _dirs(h, w)
    zc = D[..., 2]
    sky = 0.06 + 0.05 * np.clip(zc, 0, 1)
    ground = 0.035 + 0.01 * np.clip(-zc, 0, 1)
    img = np.where(zc[..., None] >= 0, sky[..., None], ground[..., None]) * np.ones(3, np.float32)
    img = img.astype(np.float32)
    _box_lights(D, [
        (40.0, 35.0, 22.0, 14.0, (6.0, 5.9, 5.7), 3.0),     # key softbox
        (215.0, 25.0, 18.0, 12.0, (1.8, 1.85, 2.0), 4.0),   # fill
        (120.0, 70.0, 40.0, 4.0, (3.5, 3.5, 3.5), 1.5),     # top strip
        (300.0, 10.0, 4.0, 20.0, (2.5, 2.5, 2.6), 1.0),     # vertical rim strip
    ], img)
    return img


def gen_neon_night(h=GEN_H, w=GEN_W):
    """Night city: near-black sky, a band of lit windows, magenta and cyan neon tubes."""
    D, T, P = _dirs(h, w)
    el = np.degrees(np.pi / 2 - T)
    az = np.degrees(P)
    sky = np.array([0.002, 0.003, 0.008], np.float32) + np.clip(1 - el / 30, 0, 1)[..., None] * np.array([0.012, 0.006, 0.02], np.float32)
    img = np.where((el >= 0)[..., None], sky, np.array([0.002, 0.002, 0.003], np.float32)).astype(np.float32)
    rng = np.random.default_rng(7)
    # Skyline: buildings with lit windows between 0 and ~14 degrees elevation.
    n_b = 60
    edges = np.sort(rng.uniform(-180, 180, n_b))
    heights = rng.uniform(4, 14, n_b)
    idx = np.searchsorted(edges, az) % n_b
    bh = heights[idx]
    building = (el >= 0) & (el < bh)
    img[building] = np.array([0.003, 0.003, 0.005], np.float32)
    wx = (np.abs(((az * 6.0) % 1.0) - 0.5) < 0.22)
    wy = (np.abs(((el * 2.5) % 1.0) - 0.5) < 0.2)
    cell = (np.floor(az * 6.0).astype(int) * 7919 + np.floor(el * 2.5).astype(int) * 104729) % 97
    lit = building & wx & wy & (cell < 28)
    warm = (cell % 3 == 0)[..., None]
    img += lit[..., None] * np.where(warm, np.array([1.6, 1.1, 0.5], np.float32), np.array([0.6, 0.9, 1.4], np.float32))
    _box_lights(D, [
        (60.0, 22.0, 30.0, 0.5, (8.0, 0.6, 5.0), 0.3),      # magenta tube
        (-30.0, 30.0, 0.5, 14.0, (0.4, 5.0, 7.0), 0.3),     # cyan vertical tube
        (160.0, 16.0, 18.0, 0.4, (0.5, 6.0, 7.5), 0.3),     # cyan tube
        (240.0, 34.0, 10.0, 0.6, (7.0, 1.2, 4.0), 0.3),     # pink sign
        (110.0, 8.0, 6.0, 3.0, (3.0, 1.8, 0.4), 0.8),       # amber billboard
    ], img)
    return img


def gen_overcast(h=GEN_H, w=GEN_W):
    """Soft overcast sky (CIE overcast falloff) over neutral ground."""
    D, T, P = _dirs(h, w)
    zc = D[..., 2]
    lum = 1.1 * (1 + 2 * np.clip(zc, 0, 1)) / 3
    sky = lum[..., None] * np.array([0.92, 0.96, 1.0], np.float32)
    horizon = np.exp(-np.abs(zc) * 30)[..., None] * np.array([0.05, 0.05, 0.05], np.float32)
    ground = np.array([0.11, 0.1, 0.09], np.float32) * (0.8 + 0.2 * np.clip(-zc, 0, 1))[..., None]
    return np.where((zc >= 0)[..., None], sky + horizon, ground + horizon).astype(np.float32)


BUILTINS = {
    "dark_hangar": ("Dark hangar", gen_dark_hangar),
    "studio": ("Studio", gen_studio),
    "neon_night": ("Neon night", gen_neon_night),
    "overcast": ("Overcast sky", gen_overcast),
}
DEFAULT_ENV = "dark_hangar"


def gradient_image(env3, h=32, w=64):
    """The light preset's 3-color gradient as an environment image."""
    top, hor, bot = (np.array(c, np.float32) for c in env3)
    D, _, _ = _dirs(h, w)
    zc = D[..., 2:3]
    up = hor + (top - hor) * np.clip(zc, 0, 1)
    down = hor + (bot - hor) * np.clip(-zc * 4, 0, 1)
    return np.where(zc >= 0, up, down).astype(np.float32)


# ---------------------------------------------------------------- listing and loading
def _safe_name(name):
    name = os.path.basename(str(name).replace("\\", "/"))
    stem = name[:-4] if name.lower().endswith(".hdr") else name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")[:80] or "upload"
    return stem + ".hdr"


def list_environments():
    out = [{"id": k, "label": v[0], "kind": "builtin"} for k, v in BUILTINS.items()]
    if HDRI_DIR.is_dir():
        for f in sorted(HDRI_DIR.glob("*.hdr")):
            out.append({"id": "file:" + f.name, "label": f.stem, "kind": "file"})
    out.append({"id": "gradient", "label": "Light preset gradient", "kind": "gradient"})
    return out


def _file_for(env_id):
    if not env_id.startswith("file:"):
        return None
    name = _safe_name(env_id[5:])
    path = HDRI_DIR / name
    return path if path.is_file() else None


def _cache_key(env_id):
    if env_id in BUILTINS:
        return f"builtin_{env_id}_v3"
    path = _file_for(env_id)
    if path is None:
        return None
    st = path.stat()
    return f"file_{path.stem}_{st.st_size}_{int(st.st_mtime)}"


def load_image(env_id):
    """Full-resolution float32 image for a builtin or file id, cached on disk as .npy."""
    key = _cache_key(env_id)
    if key is None:
        raise KeyError(f"unknown environment {env_id!r}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{key}.npy"
    if cache.is_file():
        try:
            return np.load(cache)
        except (OSError, ValueError):
            pass
    if env_id in BUILTINS:
        img = BUILTINS[env_id][1]()
    else:
        img = read_hdr(_file_for(env_id))
    img = np.nan_to_num(np.maximum(img, 0), posinf=0).astype(np.float32)
    np.save(cache, img)
    return img


def save_upload(filename, data: bytes):
    """Validate and store an uploaded .hdr. Returns its environment id."""
    if len(data) > MAX_HDR_BYTES:
        raise ValueError("file too large")
    read_hdr(data)  # raises ValueError if invalid
    HDRI_DIR.mkdir(parents=True, exist_ok=True)
    name = _safe_name(filename)
    (HDRI_DIR / name).write_bytes(data)
    return "file:" + name


# ---------------------------------------------------------------- sampling data
class EnvMap:
    """Pathtracer-ready environment: radiance image, blurred background and
    piecewise-constant importance sampling tables (luminance x sin(theta))."""

    def __init__(self, img):
        img = img.astype(np.float64)
        if img.shape[0] > RENDER_H or img.shape[1] > RENDER_W:
            img = _resize(img, RENDER_H, RENDER_W)
        self.img = np.ascontiguousarray(img)
        h, w = self.img.shape[:2]
        self.blur = np.ascontiguousarray(_blur_background(self.img))
        lum = self.img @ np.array([0.2126, 0.7152, 0.0722])
        sin_t = np.sin((np.arange(h) + 0.5) / h * np.pi)
        f = lum * sin_t[:, None]
        total = f.sum()
        self.enabled = bool(total > 0)
        if not self.enabled:
            f = np.ones_like(f)
            total = f.sum()
        self.pdf = np.ascontiguousarray(f * (h * w) / total)           # density over the unit square
        rows = f.sum(axis=1)
        self.marg = np.concatenate([[0.0], np.cumsum(rows) / rows.sum()])
        cond = np.cumsum(f, axis=1)
        safe = np.where(rows[:, None] > 0, rows[:, None], 1.0)
        cond = np.where(rows[:, None] > 0, cond / safe, (np.arange(1, w + 1) / w)[None, :])
        self.cond = np.ascontiguousarray(np.concatenate([np.zeros((h, 1)), cond], axis=1))
        self.marg[-1] = 1.0
        self.cond[:, -1] = 1.0
        # Mean radiance over the sphere, useful for tests and exposure hints.
        self.mean = float((lum * sin_t[:, None]).sum() * (np.pi / h) * (2 * np.pi / w) / (4 * np.pi))


def _blur_background(img):
    small = _resize(img, BLUR_H * 2, BLUR_W * 2)
    small = _resize(small, BLUR_H, BLUR_W)
    k = np.array([1, 4, 6, 4, 1], np.float64) / 16
    for _ in range(2):
        small = sum(np.roll(small, i - 2, axis=1) * k[i] for i in range(5))
        p = np.pad(small, ((2, 2), (0, 0), (0, 0)), mode="edge")
        small = sum(p[i:i + BLUR_H] * k[i] for i in range(5))
    return small


# ---------------------------------------------------------------- editor prefiltering
GL_LEVELS = [  # (roughness, height, width); level 0 is the sharp image
    (0.0, 256, 512), (0.25, 64, 128), (0.5, 32, 64), (0.75, 32, 64), (1.0, 16, 32)]
GL_IRR = (16, 32)


def _convolve(src, out_h, out_w, exponent):
    hs, ws = src.shape[:2]
    Ds, Ts, _ = _dirs(hs, ws)
    sa = (np.sin(Ts) * (np.pi / hs) * (2 * np.pi / ws)).ravel().astype(np.float32)
    Ds = Ds.reshape(-1, 3).astype(np.float32)
    S = src.reshape(-1, 3).astype(np.float32)
    Do = _dirs(out_h, out_w)[0].reshape(-1, 3).astype(np.float32)
    out = np.empty((Do.shape[0], 3), np.float32)
    for i in range(0, Do.shape[0], 512):
        c = np.maximum(Do[i:i + 512] @ Ds.T, 0.0)
        wgt = np.power(c, exponent) * sa
        out[i:i + 512] = (wgt @ S) / np.maximum(wgt.sum(axis=1, keepdims=True), 1e-12)
    return out.reshape(out_h, out_w, 3)


def prefilter_for_gl(img):
    """Roughness levels (Phong-lobe approximation of GGX) plus cosine irradiance."""
    levels = []
    base = _resize(img.astype(np.float64), 64, 128)
    for rough, h, w in GL_LEVELS:
        if rough == 0:
            levels.append(_resize(img.astype(np.float64), h, w).astype(np.float32))
            continue
        a = rough * rough
        n = max(2.0 / (a * a) - 2.0, 1.0)
        levels.append(_convolve(base, h, w, n))
    irr = _convolve(_resize(img.astype(np.float64), 32, 64), GL_IRR[0], GL_IRR[1], 1.0)
    return levels, irr


class EnvCache:
    """Thread-safe cache of EnvMap and GL prefiltered data by environment id."""

    def __init__(self):
        self._lock = threading.Lock()
        self._maps = {}
        self._gl = {}

    def _key(self, env_id, gradient):
        if env_id == "gradient":
            return ("gradient", tuple(map(tuple, gradient)))
        return (env_id, _cache_key(env_id))

    def image(self, env_id, gradient):
        if env_id == "gradient":
            return gradient_image(gradient)
        return load_image(env_id)

    def envmap(self, env_id, gradient=None) -> EnvMap:
        if env_id != "gradient" and _cache_key(env_id) is None:
            env_id = DEFAULT_ENV
        key = self._key(env_id, gradient)
        with self._lock:
            if key in self._maps:
                return self._maps[key]
        em = EnvMap(self.image(env_id, gradient))
        with self._lock:
            self._maps[key] = em
            if len(self._maps) > 8:
                self._maps.pop(next(iter(self._maps)))
        return em

    def gl(self, env_id, gradient=None):
        if env_id != "gradient" and _cache_key(env_id) is None:
            env_id = DEFAULT_ENV
        key = self._key(env_id, gradient)
        with self._lock:
            if key in self._gl:
                return self._gl[key]
        disk = None
        if env_id != "gradient":
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            disk = CACHE_DIR / f"{key[1]}_gl.npz"
        data = None
        if disk is not None and disk.is_file():
            try:
                z = np.load(disk)
                data = ([z[f"l{i}"] for i in range(len(GL_LEVELS))], z["irr"])
            except (OSError, ValueError, KeyError):
                data = None
        if data is None:
            data = prefilter_for_gl(self.image(env_id, gradient))
            if disk is not None:
                np.savez(disk, irr=data[1], **{f"l{i}": lv for i, lv in enumerate(data[0])})
        with self._lock:
            self._gl[key] = data
        return data


def thumbnail(img, h=96, w=192):
    """Small tonemapped preview (uint8 RGB) with automatic exposure."""
    t = _resize(img.astype(np.float64), h, w)
    k = 1.5   # fixed exposure, so dark environments look dark
    x = t * k
    y = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
    return (np.power(y, 1 / 2.2) * 255 + 0.5).astype(np.uint8)
