"""Multi-threaded CPU pathtracer (Numba).

This is the only module in the project that imports Numba. It renders a
height-field panel lit by spotlights with a GGX metallic/roughness material.

Scene layout (all float64 arrays, meters):
  sc   [bx0, bx1, by0, by1, zmin, zmax, tex_w, tex_h, texel, wrap,
        displace, floor_z, normal_scale, variation, eps]
  cam  [px, py, pz, fx, fy, fz, rx, ry, rz, ux, uy, uz, tan_half_fov, aspect]
  mat  [unused x5, floor_r, floor_g, floor_b, floor_rough]
  Mt   (h, w, 5) surface material per texel: linear r, g, b, metallic, roughness
       (panels, paint, wires, weathering; see core/weather.py)
  lights (n, 14): [px, py, pz, dx, dy, dz, r, g, b, cos_inner, cos_outer, radius, 0, 0]
  E    (h, w, 3) equirect environment radiance, row 0 = zenith
  EB   (h', w', 3) blurred environment for the background option
  P, M, C  env importance sampling: pdf over the unit square, marginal and
       conditional CDFs (see render/hdri.py EnvMap)
  ep   [intensity, rotation_rad, background (0 hdri, 1 black, 2 blurred), env_nee]
  Em   (h, w, 3) emission radiance texture (same layout as the height map)
  EPD  (h, w) emitter sampling density per unit area (xy plane), per texel
  ECDF (n + 1,) cumulative emitter power over the n emissive texels
  EIDX (n,) flat texel index of each emissive texel
  emp  [enabled, tiles, emitter_nee]
  opt  [max_bounces, clamp, emitter shadow-ray roulette threshold (0 = off)]
World: x right, y up the texture (+V), z out of the surface.
"""
import math

import numpy as np
from numba import njit, prange

INV_PI = 1.0 / math.pi


# ---------------------------------------------------------------- textures
@njit(cache=True, fastmath=True, inline="always")
def _wrapi(i, n, wrap):
    if wrap:
        return i % n
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return i


@njit(cache=True, fastmath=True)
def _tex_coords(x, y, sc, nx, ny):
    # Texture pixel coordinates; row 0 is the top of the image (+y).
    fx = (x + 0.5 * sc[6]) / sc[6] * nx - 0.5
    fy = (0.5 * sc[7] - y) / sc[7] * ny - 0.5
    return fx, fy


@njit(cache=True, fastmath=True)
def _height(Hm, x, y, sc):
    ny, nx = Hm.shape
    fx, fy = _tex_coords(x, y, sc, nx, ny)
    wrap = sc[9] > 0.5
    x0 = math.floor(fx)
    y0 = math.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = _wrapi(int(x0), nx, wrap)
    i1 = _wrapi(int(x0) + 1, nx, wrap)
    j0 = _wrapi(int(y0), ny, wrap)
    j1 = _wrapi(int(y0) + 1, ny, wrap)
    a = Hm[j0, i0] * (1 - tx) + Hm[j0, i1] * tx
    b = Hm[j1, i0] * (1 - tx) + Hm[j1, i1] * tx
    return a * (1 - ty) + b * ty


@njit(cache=True, fastmath=True)
def _normal_map(Nm, x, y, sc):
    ny, nx = Nm.shape[0], Nm.shape[1]
    fx, fy = _tex_coords(x, y, sc, nx, ny)
    wrap = sc[9] > 0.5
    x0 = math.floor(fx)
    y0 = math.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = _wrapi(int(x0), nx, wrap)
    i1 = _wrapi(int(x0) + 1, nx, wrap)
    j0 = _wrapi(int(y0), ny, wrap)
    j1 = _wrapi(int(y0) + 1, ny, wrap)
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    nxv = Nm[j0, i0, 0] * w00 + Nm[j0, i1, 0] * w10 + Nm[j1, i0, 0] * w01 + Nm[j1, i1, 0] * w11
    nyv = Nm[j0, i0, 1] * w00 + Nm[j0, i1, 1] * w10 + Nm[j1, i0, 1] * w01 + Nm[j1, i1, 1] * w11
    nzv = Nm[j0, i0, 2] * w00 + Nm[j0, i1, 2] * w10 + Nm[j1, i0, 2] * w01 + Nm[j1, i1, 2] * w11
    s = sc[12]
    nxv *= s
    nyv *= s
    inv = 1.0 / math.sqrt(nxv * nxv + nyv * nyv + nzv * nzv + 1e-20)
    return nxv * inv, nyv * inv, nzv * inv


@njit(cache=True, fastmath=True)
def _material(Mt, x, y, sc):
    """Bilinear material at a surface point: (r, g, b, metallic, roughness)."""
    ny, nx = Mt.shape[0], Mt.shape[1]
    fx, fy = _tex_coords(x, y, sc, nx, ny)
    wrap = sc[9] > 0.5
    x0 = math.floor(fx)
    y0 = math.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = _wrapi(int(x0), nx, wrap)
    i1 = _wrapi(int(x0) + 1, nx, wrap)
    j0 = _wrapi(int(y0), ny, wrap)
    j1 = _wrapi(int(y0) + 1, ny, wrap)
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    r = Mt[j0, i0, 0] * w00 + Mt[j0, i1, 0] * w10 + Mt[j1, i0, 0] * w01 + Mt[j1, i1, 0] * w11
    g = Mt[j0, i0, 1] * w00 + Mt[j0, i1, 1] * w10 + Mt[j1, i0, 1] * w01 + Mt[j1, i1, 1] * w11
    b = Mt[j0, i0, 2] * w00 + Mt[j0, i1, 2] * w10 + Mt[j1, i0, 2] * w01 + Mt[j1, i1, 2] * w11
    m = Mt[j0, i0, 3] * w00 + Mt[j0, i1, 3] * w10 + Mt[j1, i0, 3] * w01 + Mt[j1, i1, 3] * w11
    ro = Mt[j0, i0, 4] * w00 + Mt[j0, i1, 4] * w10 + Mt[j1, i0, 4] * w01 + Mt[j1, i1, 4] * w11
    return r, g, b, m, ro


# ---------------------------------------------------------------- geometry
@njit(cache=True, fastmath=True)
def _slab(o, d, lo, hi):
    if abs(d) < 1e-12:
        if o < lo or o > hi:
            return 1.0, -1.0
        return -1e30, 1e30
    t0 = (lo - o) / d
    t1 = (hi - o) / d
    if t0 > t1:
        t0, t1 = t1, t0
    return t0, t1


@njit(cache=True, fastmath=True)
def _intersect(ox, oy, oz, dx, dy, dz, Hm, sc, tmax):
    """Return (kind, t). kind 0 = miss, 1 = panel, 2 = floor."""
    bx0, bx1, by0, by1 = sc[0], sc[1], sc[2], sc[3]
    if sc[10] < 0.5:  # normal-only: flat plane at z = 0
        if dz >= -1e-12:
            return 0, 0.0
        t = -oz / dz
        if t <= 0 or t > tmax:
            return 0, 0.0
        x = ox + dx * t
        y = oy + dy * t
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            return 1, t
        return 2, t

    zmin, zmax = sc[4] - sc[14], sc[5] + sc[14]
    ax0, ax1 = _slab(ox, dx, bx0, bx1)
    ay0, ay1 = _slab(oy, dy, by0, by1)
    az0, az1 = _slab(oz, dz, zmin, zmax)
    t0 = max(max(ax0, ay0), max(az0, 0.0))
    t1 = min(min(ax1, ay1), min(az1, tmax))
    if t1 > t0:
        dxy = math.sqrt(dx * dx + dy * dy)
        step = 0.75 * sc[8] / max(dxy, 1e-9)
        t = t0
        above = (oz + dz * t) > _height(Hm, ox + dx * t, oy + dy * t, sc)
        n = 0
        while t < t1 and n < 4096:
            tn = min(t + step, t1)
            z = oz + dz * tn
            h = _height(Hm, ox + dx * tn, oy + dy * tn, sc)
            if z <= h:
                if above:
                    lo, hi = t, tn
                    for _ in range(8):
                        mid = 0.5 * (lo + hi)
                        if oz + dz * mid <= _height(Hm, ox + dx * mid, oy + dy * mid, sc):
                            hi = mid
                        else:
                            lo = mid
                    return 1, hi
            else:
                above = True
            t = tn
            n += 1
    # Floor outside the panel bounds, level with the panel base.
    if dz < -1e-12:
        t = (sc[11] - oz) / dz
        if 0 < t <= tmax:
            x = ox + dx * t
            y = oy + dy * t
            if not (bx0 <= x <= bx1 and by0 <= y <= by1):
                return 2, t
    return 0, 0.0


@njit(cache=True, fastmath=True)
def _geo_normal(Hm, x, y, sc):
    e = sc[8]
    hx = (_height(Hm, x + e, y, sc) - _height(Hm, x - e, y, sc)) / (2 * e)
    hy = (_height(Hm, x, y + e, sc) - _height(Hm, x, y - e, sc)) / (2 * e)
    inv = 1.0 / math.sqrt(hx * hx + hy * hy + 1.0)
    return -hx * inv, -hy * inv, inv


# ---------------------------------------------------------------- BRDF
@njit(cache=True, fastmath=True)
def _onb(nx, ny, nz):
    if nz < -0.9999999:
        return 0.0, -1.0, 0.0, -1.0, 0.0, 0.0
    a = 1.0 / (1.0 + nz)
    b = -nx * ny * a
    return 1.0 - nx * nx * a, b, -nx, b, 1.0 - ny * ny * a, -ny


@njit(cache=True, fastmath=True)
def _ggx_d(noh, a2):
    d = noh * noh * (a2 - 1.0) + 1.0
    return a2 / (math.pi * d * d + 1e-20)


@njit(cache=True, fastmath=True)
def _smith_g1(nox, a2):
    return 2.0 * nox / (nox + math.sqrt(a2 + (1.0 - a2) * nox * nox) + 1e-20)


@njit(cache=True, fastmath=True)
def _eval_brdf(n, wo, wi, base, metallic, rough):
    """Returns (f_r, f_g, f_b, pdf) for the lobe mixture used in sampling."""
    nol = n[0] * wi[0] + n[1] * wi[1] + n[2] * wi[2]
    nov = n[0] * wo[0] + n[1] * wo[1] + n[2] * wo[2]
    if nol <= 0 or nov <= 0:
        return 0.0, 0.0, 0.0, 0.0
    hx, hy, hz = wo[0] + wi[0], wo[1] + wi[1], wo[2] + wi[2]
    inv = 1.0 / math.sqrt(hx * hx + hy * hy + hz * hz + 1e-20)
    hx *= inv
    hy *= inv
    hz *= inv
    noh = max(n[0] * hx + n[1] * hy + n[2] * hz, 0.0)
    voh = max(wo[0] * hx + wo[1] * hy + wo[2] * hz, 1e-6)
    a = max(rough * rough, 0.002)
    a2 = a * a
    D = _ggx_d(noh, a2)
    G = _smith_g1(nov, a2) * _smith_g1(nol, a2)
    fw = (1.0 - voh) ** 5
    spec = D * G / (4.0 * nov * nol)
    kd = (1.0 - metallic) * INV_PI
    f0r = 0.04 * (1.0 - metallic) + base[0] * metallic
    f0g = 0.04 * (1.0 - metallic) + base[1] * metallic
    f0b = 0.04 * (1.0 - metallic) + base[2] * metallic
    Fr = f0r + (1.0 - f0r) * fw
    Fg = f0g + (1.0 - f0g) * fw
    Fb = f0b + (1.0 - f0b) * fw
    p_spec = 0.5 + 0.5 * metallic
    pdf = p_spec * D * noh / (4.0 * voh) + (1.0 - p_spec) * nol * INV_PI
    return (spec * Fr + (1.0 - Fr) * kd * base[0],
            spec * Fg + (1.0 - Fg) * kd * base[1],
            spec * Fb + (1.0 - Fb) * kd * base[2], pdf)


@njit(cache=True, fastmath=True)
def _sample_dir(n, wo, metallic, rough):
    tx, ty, tz, bx, by, bz = _onb(n[0], n[1], n[2])
    u1 = np.random.random()
    u2 = np.random.random()
    phi = 2.0 * math.pi * u2
    if np.random.random() < 0.5 + 0.5 * metallic:
        a = max(rough * rough, 0.002)
        cos_t = math.sqrt((1.0 - u1) / (1.0 + (a * a - 1.0) * u1))
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        lx, ly, lz = sin_t * math.cos(phi), sin_t * math.sin(phi), cos_t
        hx = tx * lx + bx * ly + n[0] * lz
        hy = ty * lx + by * ly + n[1] * lz
        hz = tz * lx + bz * ly + n[2] * lz
        voh = wo[0] * hx + wo[1] * hy + wo[2] * hz
        return 2 * voh * hx - wo[0], 2 * voh * hy - wo[1], 2 * voh * hz - wo[2]
    r = math.sqrt(u1)
    lx, ly, lz = r * math.cos(phi), r * math.sin(phi), math.sqrt(max(0.0, 1.0 - u1))
    return (tx * lx + bx * ly + n[0] * lz, ty * lx + by * ly + n[1] * lz,
            tz * lx + bz * ly + n[2] * lz)


# ---------------------------------------------------------------- environment
@njit(cache=True, fastmath=True)
def _env_uv(dx, dy, dz, rot):
    phi = math.atan2(dy, dx) - rot
    u = 0.5 + phi / (2.0 * math.pi)
    u -= math.floor(u)
    v = math.acos(min(max(dz, -1.0), 1.0)) / math.pi
    return u, v


@njit(cache=True, fastmath=True)
def _env_lookup(E, u, v):
    h, w = E.shape[0], E.shape[1]
    fx = u * w - 0.5
    fy = min(max(v * h - 0.5, 0.0), h - 1.0)
    x0 = math.floor(fx)
    y0 = int(math.floor(fy))
    tx = fx - x0
    ty = fy - y0
    i0 = int(x0) % w
    i1 = (i0 + 1) % w
    j1 = min(y0 + 1, h - 1)
    r = ((E[y0, i0, 0] * (1 - tx) + E[y0, i1, 0] * tx) * (1 - ty)
         + (E[j1, i0, 0] * (1 - tx) + E[j1, i1, 0] * tx) * ty)
    g = ((E[y0, i0, 1] * (1 - tx) + E[y0, i1, 1] * tx) * (1 - ty)
         + (E[j1, i0, 1] * (1 - tx) + E[j1, i1, 1] * tx) * ty)
    b = ((E[y0, i0, 2] * (1 - tx) + E[y0, i1, 2] * tx) * (1 - ty)
         + (E[j1, i0, 2] * (1 - tx) + E[j1, i1, 2] * tx) * ty)
    return r, g, b


@njit(cache=True, fastmath=True)
def _env_radiance(E, dx, dy, dz, ep):
    u, v = _env_uv(dx, dy, dz, ep[1])
    r, g, b = _env_lookup(E, u, v)
    k = ep[0]
    return r * k, g * k, b * k


@njit(cache=True, fastmath=True)
def _env_pdf(P, dx, dy, dz, ep):
    """Solid-angle pdf of _env_sample for a direction."""
    s = math.sqrt(max(0.0, 1.0 - dz * dz))
    if s < 1e-7:
        return 0.0
    h, w = P.shape[0], P.shape[1]
    u, v = _env_uv(dx, dy, dz, ep[1])
    i = min(int(v * h), h - 1)
    j = min(int(u * w), w - 1)
    return P[i, j] / (2.0 * math.pi * math.pi * s)


@njit(cache=True, fastmath=True)
def _env_sample(P, M, C, ep):
    """Importance-sample a direction proportional to luminance x sin(theta)."""
    h, w = P.shape[0], P.shape[1]
    u1 = np.random.random()
    u2 = np.random.random()
    i = np.searchsorted(M, u1, side="right") - 1
    i = min(max(i, 0), h - 1)
    dm = M[i + 1] - M[i]
    fv = (u1 - M[i]) / dm if dm > 0 else 0.5
    row = C[i]
    j = np.searchsorted(row, u2, side="right") - 1
    j = min(max(j, 0), w - 1)
    dc = row[j + 1] - row[j]
    fu = (u2 - row[j]) / dc if dc > 0 else 0.5
    v = (i + fv) / h
    u = (j + fu) / w
    theta = v * math.pi
    phi = (u - 0.5) * 2.0 * math.pi + ep[1]
    s = math.sin(theta)
    if s < 1e-7:
        return 0.0, 0.0, 1.0, 0.0
    pdf = P[i, j] / (2.0 * math.pi * math.pi * s)
    return s * math.cos(phi), s * math.sin(phi), math.cos(theta), pdf


# ---------------------------------------------------------------- emitters
@njit(cache=True, fastmath=True)
def _emission(Em, x, y, sc):
    ny, nx = Em.shape[0], Em.shape[1]
    fx, fy = _tex_coords(x, y, sc, nx, ny)
    wrap = sc[9] > 0.5
    if not wrap and (fx < -0.5 or fy < -0.5 or fx > nx - 0.5 or fy > ny - 0.5):
        return 0.0, 0.0, 0.0
    x0 = math.floor(fx)
    y0 = math.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = _wrapi(int(x0), nx, wrap)
    i1 = _wrapi(int(x0) + 1, nx, wrap)
    j0 = _wrapi(int(y0), ny, wrap)
    j1 = _wrapi(int(y0) + 1, ny, wrap)
    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty
    r = Em[j0, i0, 0] * w00 + Em[j0, i1, 0] * w10 + Em[j1, i0, 0] * w01 + Em[j1, i1, 0] * w11
    g = Em[j0, i0, 1] * w00 + Em[j0, i1, 1] * w10 + Em[j1, i0, 1] * w01 + Em[j1, i1, 1] * w11
    b = Em[j0, i0, 2] * w00 + Em[j0, i1, 2] * w10 + Em[j1, i0, 2] * w01 + Em[j1, i1, 2] * w11
    return r, g, b


@njit(cache=True, fastmath=True)
def _emit_texel(x, y, sc, nx, ny):
    """Texel index containing a world point (the cell used by emitter sampling)."""
    u = (x + 0.5 * sc[6]) / sc[6] * nx
    v = (0.5 * sc[7] - y) / sc[7] * ny
    j = int(math.floor(u)) % nx
    i = int(math.floor(v)) % ny
    return i, j


@njit(cache=True, fastmath=True)
def _emit_pdf_area(EPD, x, y, nz, sc, emp):
    """Area density (per unit surface area) of picking this point by emitter sampling."""
    ny, nx = EPD.shape[0], EPD.shape[1]
    i, j = _emit_texel(x, y, sc, nx, ny)
    t = emp[1]
    return EPD[i, j] * nz / (t * t)


@njit(cache=True, fastmath=True)
def _emit_sample(Hm, EPD, ECDF, EIDX, sc, emp):
    """Pick a point on an emitter proportional to emitted power.
    Returns (x, y, z, nx, ny, nz, area_pdf)."""
    ny, nx = EPD.shape[0], EPD.shape[1]
    k = np.searchsorted(ECDF, np.random.random(), side="right") - 1
    k = min(max(k, 0), EIDX.shape[0] - 1)
    flat = EIDX[k]
    i = flat // nx
    j = flat - i * nx
    x = (j + np.random.random()) / nx * sc[6] - 0.5 * sc[6]
    y = 0.5 * sc[7] - (i + np.random.random()) / ny * sc[7]
    t = int(emp[1])
    if t > 1:   # one of the repeated tiles, uniformly
        x += (int(np.random.random() * t) - (t - 1) // 2) * sc[6]
        y += (int(np.random.random() * t) - (t - 1) // 2) * sc[7]
    if sc[10] > 0.5:
        z = _height(Hm, x, y, sc)
        gx, gy, gz = _geo_normal(Hm, x, y, sc)
    else:
        z = 0.0
        gx, gy, gz = 0.0, 0.0, 1.0
    return x, y, z, gx, gy, gz, EPD[i, j] * gz / (t * t)


# ---------------------------------------------------------------- integrator
@njit(cache=True, fastmath=True)
def _trace(ox, oy, oz, dx, dy, dz, Hm, Nm, Mt, sc, mat, lights, E, EB, P, M, C, ep,
           Em, EPD, ECDF, EIDX, emp, opt):
    Lr = Lg = Lb = 0.0
    tr = tg = tb = 1.0
    eps = sc[14]
    max_b = int(opt[0])
    use_env_nee = ep[3] > 0.5
    use_emit = emp[0] > 0.5
    use_emit_nee = use_emit and emp[2] > 0.5
    rr_t = opt[2]
    last_pdf = -1.0          # pdf of the BSDF sample that produced this ray
    for bounce in range(max_b + 1):
        kind, t = _intersect(ox, oy, oz, dx, dy, dz, Hm, sc, 1e30)
        if kind == 0:
            w = 1.0
            if bounce == 0:
                mode = int(ep[2])
                if mode == 0:
                    er, eg, eb = _env_radiance(E, dx, dy, dz, ep)
                elif mode == 2:
                    er, eg, eb = _env_radiance(EB, dx, dy, dz, ep)
                else:
                    er = eg = eb = 0.0
            else:
                er, eg, eb = _env_radiance(E, dx, dy, dz, ep)
                if use_env_nee:
                    pl = _env_pdf(P, dx, dy, dz, ep)
                    w = last_pdf * last_pdf / (last_pdf * last_pdf + pl * pl + 1e-30)
            Lr += tr * er * w
            Lg += tg * eg * w
            Lb += tb * eb * w
            break
        px, py, pz = ox + dx * t, oy + dy * t, oz + dz * t
        if kind == 1:
            if sc[10] > 0.5:
                g = _geo_normal(Hm, px, py, sc)
            else:
                g = (0.0, 0.0, 1.0)
            n = _normal_map(Nm, px, py, sc)
            mr, mg, mb, metallic, rough = _material(Mt, px, py, sc)
            base = (mr, mg, mb)
            rough = min(max(rough, 0.02), 1.0)
            if use_emit:
                er, eg, eb = _emission(Em, px, py, sc)
                if er + eg + eb > 0:
                    w = 1.0
                    cos_l = -(g[0] * dx + g[1] * dy + g[2] * dz)
                    if cos_l <= 0:
                        w = 0.0
                    elif bounce > 0 and use_emit_nee:
                        pl = _emit_pdf_area(EPD, px, py, g[2], sc, emp) * t * t / cos_l
                        w = last_pdf * last_pdf / (last_pdf * last_pdf + pl * pl + 1e-30)
                    Lr += tr * er * w
                    Lg += tg * eg * w
                    Lb += tb * eb * w
        else:
            g = (0.0, 0.0, 1.0)
            n = g
            base = (mat[5], mat[6], mat[7])
            metallic = 0.0
            rough = mat[8]
        wo = (-dx, -dy, -dz)
        if n[0] * wo[0] + n[1] * wo[1] + n[2] * wo[2] <= 0.0:
            n = g
        # Offset along the geometric normal to avoid self-hits.
        sx, sy, sz = px + g[0] * eps, py + g[1] * eps, pz + g[2] * eps

        # Next-event estimation: sample each spotlight's disk.
        for li in range(lights.shape[0]):
            L = lights[li]
            lx, ly, lz = L[0] - px, L[1] - py, L[2] - pz
            d0 = math.sqrt(lx * lx + ly * ly + lz * lz)
            if L[11] > 0:
                ax, ay, az, bx, by, bz = _onb(lx / d0, ly / d0, lz / d0)
                rr = L[11] * math.sqrt(np.random.random())
                ph = 2 * math.pi * np.random.random()
                cu, cv = rr * math.cos(ph), rr * math.sin(ph)
                lx += ax * cu + bx * cv
                ly += ay * cu + by * cv
                lz += az * cu + bz * cv
            dist = math.sqrt(lx * lx + ly * ly + lz * lz)
            wix, wiy, wiz = lx / dist, ly / dist, lz / dist
            if g[0] * wix + g[1] * wiy + g[2] * wiz <= 0:
                continue
            cos_a = -(wix * L[3] + wiy * L[4] + wiz * L[5])
            if cos_a <= L[10]:
                continue
            s = min(max((cos_a - L[10]) / max(L[9] - L[10], 1e-6), 0.0), 1.0)
            s = s * s * (3 - 2 * s)
            wi = (wix, wiy, wiz)
            fr, fg, fb, _ = _eval_brdf(n, wo, wi, base, metallic, rough)
            nol = n[0] * wix + n[1] * wiy + n[2] * wiz
            if nol <= 0 or fr + fg + fb <= 0:
                continue
            occ, _t = _intersect(sx, sy, sz, wix, wiy, wiz, Hm, sc, dist)
            if occ != 0:
                continue
            w = s * nol / (dist * dist)
            Lr += tr * fr * L[6] * w
            Lg += tg * fg * L[7] * w
            Lb += tb * fb * L[8] * w

        # Next-event estimation: importance-sampled environment, MIS with the BSDF.
        if use_env_nee:
            ex, ey, ez, pl = _env_sample(P, M, C, ep)
            if pl > 0 and g[0] * ex + g[1] * ey + g[2] * ez > 0:
                wi = (ex, ey, ez)
                fr, fg, fb, pb = _eval_brdf(n, wo, wi, base, metallic, rough)
                nol = n[0] * ex + n[1] * ey + n[2] * ez
                if nol > 0 and fr + fg + fb > 0:
                    occ, _t = _intersect(sx, sy, sz, ex, ey, ez, Hm, sc, 1e30)
                    if occ == 0:
                        er, eg, eb = _env_radiance(E, ex, ey, ez, ep)
                        w = pl / (pl * pl + pb * pb) * nol
                        Lr += tr * fr * er * w
                        Lg += tg * fg * eg * w
                        Lb += tb * fb * eb * w

        # Next-event estimation: emissive texels sampled by power, MIS with the BSDF.
        if use_emit_nee:
            qx, qy, qz, qnx, qny, qnz, pa = _emit_sample(Hm, EPD, ECDF, EIDX, sc, emp)
            if pa > 0:
                # Aim slightly above the emitter so the march does not clip it.
                tx_, ty_, tz_ = qx + qnx * eps * 4, qy + qny * eps * 4, qz + qnz * eps * 4
                lx, ly, lz = tx_ - sx, ty_ - sy, tz_ - sz
                dist2 = lx * lx + ly * ly + lz * lz
                dist = math.sqrt(dist2)
                if dist > 1e-6:
                    wix, wiy, wiz = lx / dist, ly / dist, lz / dist
                    cos_l = -(qnx * wix + qny * wiy + qnz * wiz)
                    if cos_l > 1e-4 and g[0] * wix + g[1] * wiy + g[2] * wiz > 0:
                        wi = (wix, wiy, wiz)
                        fr, fg, fb, pb = _eval_brdf(n, wo, wi, base, metallic, rough)
                        nol = n[0] * wix + n[1] * wiy + n[2] * wiz
                        if nol > 0 and fr + fg + fb > 0:
                            er, eg, eb = _emission(Em, qx, qy, sc)
                            pl = pa * dist2 / cos_l
                            w = pl / (pl * pl + pb * pb) * nol
                            # Russian roulette on the shadow ray: weak (distant, grazing)
                            # emitters are tested with probability ~ their contribution,
                            # and boosted when they are, which keeps the estimate unbiased.
                            est = max(tr, max(tg, tb)) * max(fr * er, max(fg * eg, fb * eb)) * w
                            keep = min(1.0, est / rr_t) if rr_t > 0 else 1.0
                            if keep > 0 and np.random.random() < keep:
                                occ, _t = _intersect(sx, sy, sz, wix, wiy, wiz, Hm, sc, dist - eps * 2)
                                if occ == 0:
                                    w /= keep
                                    Lr += tr * fr * er * w
                                    Lg += tg * fg * eg * w
                                    Lb += tb * fb * eb * w

        if bounce == max_b:
            break
        nx_, ny_, nz_ = _sample_dir(n, wo, metallic, rough)
        wi = (nx_, ny_, nz_)
        if g[0] * nx_ + g[1] * ny_ + g[2] * nz_ <= 0:
            break
        fr, fg, fb, pdf = _eval_brdf(n, wo, wi, base, metallic, rough)
        if pdf <= 1e-8:
            break
        nol = n[0] * nx_ + n[1] * ny_ + n[2] * nz_
        tr *= fr * nol / pdf
        tg *= fg * nol / pdf
        tb *= fb * nol / pdf
        last_pdf = pdf
        if bounce >= 2:
            p = min(max(tr, max(tg, tb)), 0.95)
            if np.random.random() > p:
                break
            tr /= p
            tg /= p
            tb /= p
        ox, oy, oz = sx, sy, sz
        dx, dy, dz = nx_, ny_, nz_
    c = opt[1]
    m = max(Lr, max(Lg, Lb))
    if c > 0 and m > c:
        k = c / m
        Lr *= k
        Lg *= k
        Lb *= k
    return Lr, Lg, Lb


@njit(parallel=True, nogil=True, cache=True, fastmath=True)
def render_pass(accum, Hm, Nm, Mt, sc, cam, mat, lights, E, EB, P, M, C, ep,
                Em, EPD, ECDF, EIDX, emp, opt):
    """Add one sample per pixel into `accum` (h, w, 3)."""
    h, w = accum.shape[0], accum.shape[1]
    for j in prange(h):
        for i in range(w):
            u = (2.0 * (i + np.random.random()) / w - 1.0) * cam[12] * cam[13]
            v = (1.0 - 2.0 * (j + np.random.random()) / h) * cam[12]
            dx = cam[3] + u * cam[6] + v * cam[9]
            dy = cam[4] + u * cam[7] + v * cam[10]
            dz = cam[5] + u * cam[8] + v * cam[11]
            inv = 1.0 / math.sqrt(dx * dx + dy * dy + dz * dz)
            r, g, b = _trace(cam[0], cam[1], cam[2], dx * inv, dy * inv, dz * inv,
                             Hm, Nm, Mt, sc, mat, lights, E, EB, P, M, C, ep,
                             Em, EPD, ECDF, EIDX, emp, opt)
            accum[j, i, 0] += r
            accum[j, i, 1] += g
            accum[j, i, 2] += b
