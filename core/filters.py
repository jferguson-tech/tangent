"""Fast image filters (NumPy): Gaussian approximation from repeated box blurs."""
import numpy as np


def _box_axis(a, r, axis, wrap):
    """Box blur of radius r (window 2r+1) along one axis using a running sum."""
    if r < 1:
        return a
    n = a.shape[axis]
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(a, pad, mode="wrap" if wrap else "edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64 if a.dtype == np.float64 else np.float32)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    lo[axis] = slice(0, n)
    return ((c[tuple(hi)] - c[tuple(lo)]) * (1.0 / (2 * r + 1))).astype(a.dtype, copy=False)


def gauss_blur(img, sigma_px, wrap=False):
    """Approximate Gaussian blur (three box passes per axis). Works on (h, w) or (h, w, c)."""
    if sigma_px < 0.5:
        return img
    # Three boxes of width w give variance 3 * (w^2 - 1) / 12.
    r = max(1, int(round((np.sqrt(4 * sigma_px * sigma_px + 1) - 1) / 2)))
    out = img
    for _ in range(3):
        out = _box_axis(out, r, 0, wrap)
        out = _box_axis(out, r, 1, wrap)
    return out


def srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _half(a):
    hh, ww = a.shape[0] // 2, a.shape[1] // 2
    return a[: hh * 2, : ww * 2].reshape(hh, 2, ww, 2, a.shape[2]).mean(axis=(1, 3))


def _resize_bilinear(a, h, w):
    """Bilinear resize with pixel-center alignment (upsampling)."""
    H, W = a.shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * H / h - 0.5, 0, H - 1)
    xs = np.clip((np.arange(w) + 0.5) * W / w - 0.5, 0, W - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    ty = (ys - y0).astype(np.float32)[:, None, None]
    tx = (xs - x0).astype(np.float32)[None, :, None]
    top = a[y0][:, x0] * (1 - tx) + a[y0][:, x1] * tx
    bot = a[y1][:, x0] * (1 - tx) + a[y1][:, x1] * tx
    return top * (1 - ty) + bot * ty


# (fraction of image height as blur sigma, weight, mip level it is computed at)
BLOOM_BANDS = ((0.004, 0.4, 1), (0.012, 0.3, 1), (0.03, 0.2, 2), (0.07, 0.1, 3))


def bloom(hdr, threshold=1.0, intensity=0.5, radius=1.0):
    """Add a soft-threshold multi-radius glow to a linear HDR image (h, w, 3).

    Like game bloom, wide glows are blurred on smaller copies of the image
    and everything is summed at half resolution before one upsample."""
    if intensity <= 0:
        return hdr
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
    knee = max(threshold * 0.5, 1e-4)
    soft = np.clip(lum - threshold + knee, 0, 2 * knee)
    soft = soft * soft / (4 * knee)
    contrib = np.maximum(soft, lum - threshold) / np.maximum(lum, 1e-6)
    bright = (hdr * np.clip(contrib, 0, None)[..., None]).astype(np.float32)
    h, w = hdr.shape[:2]
    if h < 16 or w < 16:
        return hdr + intensity * gauss_blur(bright, 0.01 * h * radius)
    mips = [bright, _half(bright)]
    while len(mips) < 4 and min(mips[-1].shape[:2]) >= 8:
        mips.append(_half(mips[-1]))
    base = mips[1]
    glow = np.zeros_like(base)
    for frac, wgt, lvl in BLOOM_BANDS:
        lvl = min(lvl, len(mips) - 1)
        g = gauss_blur(mips[lvl], frac * h * radius / 2 ** lvl)
        if lvl > 1:
            g = _resize_bilinear(g, base.shape[0], base.shape[1])
        glow += wgt * g
    return hdr + intensity * _resize_bilinear(glow, h, w)
