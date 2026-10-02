"""Minimal PNG encoder (NumPy + zlib) supporting 8 and 16-bit gray/RGB/RGBA.

Pillow cannot write 16-bit RGB, which normal maps need, so this avoids the
dependency entirely. Rows use the Sub filter, which suits smooth maps.
"""
import struct
import zlib

import numpy as np

_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}


def _chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def encode_png(arr: np.ndarray, level: int = 6) -> bytes:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    if c not in _COLOR_TYPE:
        raise ValueError(f"unsupported channel count {c}")
    if arr.dtype == np.uint16:
        depth, raw = 16, arr.astype(">u2")
    elif arr.dtype == np.uint8:
        depth, raw = 8, arr
    else:
        raise ValueError("array must be uint8 or uint16")
    bpp = c * depth // 8
    rows = np.ascontiguousarray(raw).view(np.uint8).reshape(h, w * bpp)
    filt = rows.copy()
    filt[:, bpp:] = rows[:, bpp:] - rows[:, :-bpp]      # Sub filter, uint8 wraps
    data = np.empty((h, w * bpp + 1), np.uint8)
    data[:, 0] = 1
    data[:, 1:] = filt
    ihdr = struct.pack(">IIBBBBB", w, h, depth, _COLOR_TYPE[c], 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(data.tobytes(), level))
            + _chunk(b"IEND", b""))


def to_uint(arr01: np.ndarray, bits: int) -> np.ndarray:
    """Quantize a [0, 1] float array to uint8 or uint16 with rounding."""
    a = np.clip(arr01, 0.0, 1.0)
    if bits == 16:
        return (a * 65535.0 + 0.5).astype(np.uint16)
    return (a * 255.0 + 0.5).astype(np.uint8)
