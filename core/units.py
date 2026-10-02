"""World-unit helpers.

Everything in a document is stored in meters. Pixels are derived through the
texel density (pixels per meter), game-style.
"""

TEXEL_DENSITY_DEFAULT = 512.0
TEXEL_DENSITY_PRESETS = [256.0, 512.0, 1024.0, 2048.0]

# A bevel narrower than this many pixels aliases badly in the normal map.
MIN_BEVEL_PX = 2.0

MAX_RESOLUTION = 8192

# Emission strength 1 is about as bright as a white surface under the default
# spotlight. Strength x this = linear radiance used by the renderers.
EMISSION_UNIT = 2.0


def m_to_px(meters: float, density: float) -> float:
    return float(meters) * float(density)


def px_to_m(pixels: float, density: float) -> float:
    return float(pixels) / float(density)
