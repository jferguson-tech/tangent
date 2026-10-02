"""Lighting and material presets for the pathtraced preview (no Numba here).

Light positions are given relative to the canvas size S = max(width, height),
so a preset frames a 1 m panel and a 4 m panel the same way. `power` is the
irradiance at the aim point, so brightness doesn't depend on distance.
"""

LIGHT_PRESETS = {
    "scifi_spot": {
        "label": "Sci-fi spotlight",
        "env": [[0.010, 0.013, 0.022], [0.004, 0.005, 0.008], [0.0, 0.0, 0.0]],
        "lights": [
            # key: hard cool-white spot beside the panel, grazing angle
            {"azimuth": 200.0, "elevation": 24.0, "distance": 0.95, "aim": [0.04, 0.0],
             "color": [0.85, 0.93, 1.0], "power": 14.0, "cone": 40.0, "softness": 0.18,
             "radius": 0.006},
            # faint cyan rim from the opposite side
            {"azimuth": 35.0, "elevation": 12.0, "distance": 1.1, "aim": [0.0, 0.0],
             "color": [0.25, 0.85, 1.0], "power": 0.35, "cone": 55.0, "softness": 0.6,
             "radius": 0.02},
        ],
    },
    "hangar": {
        "label": "Hangar overhead",
        "env": [[0.05, 0.05, 0.055], [0.02, 0.02, 0.022], [0.0, 0.0, 0.0]],
        "lights": [
            {"azimuth": 250.0, "elevation": 70.0, "distance": 1.2, "aim": [0.0, 0.0],
             "color": [1.0, 0.95, 0.85], "power": 2.2, "cone": 60.0, "softness": 0.5,
             "radius": 0.06},
            {"azimuth": 60.0, "elevation": 30.0, "distance": 1.4, "aim": [0.0, 0.0],
             "color": [0.6, 0.7, 0.9], "power": 0.4, "cone": 70.0, "softness": 0.7,
             "radius": 0.08},
        ],
    },
    "emergency": {
        "label": "Emergency red",
        "env": [[0.006, 0.0, 0.0], [0.003, 0.0, 0.0], [0.0, 0.0, 0.0]],
        "lights": [
            {"azimuth": 160.0, "elevation": 20.0, "distance": 0.9, "aim": [0.0, 0.0],
             "color": [1.0, 0.12, 0.05], "power": 5.0, "cone": 32.0, "softness": 0.3,
             "radius": 0.01},
            {"azimuth": 330.0, "elevation": 15.0, "distance": 1.0, "aim": [0.0, 0.0],
             "color": [1.0, 0.45, 0.1], "power": 0.5, "cone": 50.0, "softness": 0.5,
             "radius": 0.02},
        ],
    },
    "studio": {
        "label": "Studio three-point",
        "env": [[0.12, 0.12, 0.13], [0.05, 0.05, 0.05], [0.0, 0.0, 0.0]],
        "lights": [
            {"azimuth": 225.0, "elevation": 40.0, "distance": 1.2, "aim": [0.0, 0.0],
             "color": [1.0, 0.98, 0.95], "power": 2.5, "cone": 60.0, "softness": 0.6,
             "radius": 0.05},
            {"azimuth": 315.0, "elevation": 35.0, "distance": 1.3, "aim": [0.0, 0.0],
             "color": [0.85, 0.9, 1.0], "power": 0.8, "cone": 70.0, "softness": 0.7,
             "radius": 0.08},
            {"azimuth": 80.0, "elevation": 15.0, "distance": 1.2, "aim": [0.0, 0.0],
             "color": [1.0, 1.0, 1.0], "power": 0.8, "cone": 50.0, "softness": 0.5,
             "radius": 0.03},
        ],
    },
}

MATERIAL_PRESETS = {
    "gunmetal": {"label": "Gunmetal", "albedo": [0.42, 0.44, 0.47], "metallic": 1.0, "roughness": 0.45},
    "bare_steel": {"label": "Bare steel", "albedo": [0.62, 0.62, 0.64], "metallic": 1.0, "roughness": 0.22},
    "painted": {"label": "Painted metal", "albedo": [0.55, 0.56, 0.58], "metallic": 0.0, "roughness": 0.42},
    "military": {"label": "Military green", "albedo": [0.22, 0.27, 0.18], "metallic": 0.0, "roughness": 0.55},
    "white_hull": {"label": "White hull", "albedo": [0.8, 0.8, 0.78], "metallic": 0.0, "roughness": 0.32},
    "copper": {"label": "Copper", "albedo": [0.95, 0.64, 0.54], "metallic": 1.0, "roughness": 0.3},
}

WIRE_MATERIAL_PRESETS = {
    "rubber": {"label": "Black rubber", "albedo": [0.035, 0.035, 0.04], "metallic": 0.0, "roughness": 0.5},
    "braided": {"label": "Braided steel", "albedo": [0.6, 0.6, 0.62], "metallic": 1.0, "roughness": 0.42},
    "red": {"label": "Red insulation", "albedo": [0.5, 0.04, 0.03], "metallic": 0.0, "roughness": 0.35},
    "yellow": {"label": "Yellow insulation", "albedo": [0.75, 0.55, 0.05], "metallic": 0.0, "roughness": 0.35},
    "orange_hose": {"label": "Orange hose", "albedo": [0.8, 0.25, 0.03], "metallic": 0.0, "roughness": 0.55},
    "copper": {"label": "Bare copper", "albedo": [0.95, 0.64, 0.54], "metallic": 1.0, "roughness": 0.3},
}

DEFAULT_RENDER = {
    "width": 640,
    "height": 400,
    "max_spp": 256,
    "bounces": 3,
    "displacement": True,
    "height_scale": 1.0,
    "tiles": 1,
    "exposure": 0.0,
    "clamp": 20.0,
    "camera": {"yaw": 18.0, "pitch": 42.0, "distance": 1.25, "fov": 40.0},
    "wire_material": {"preset": "rubber", "albedo": [0.035, 0.035, 0.04], "metallic": 0.0,
                      "roughness": 0.5},
    "material": {"preset": "gunmetal", "albedo": [0.42, 0.44, 0.47], "metallic": 1.0,
                 "roughness": 0.45, "variation": 0.15},
    "scene_light": 1.0,
    "bloom": {"pathtrace": True, "editor": True, "threshold": 1.5, "intensity": 0.2,
              "radius": 1.0},
    "editor": {"spill": True, "spill_gain": 1.0},   # realtime viewport only
    "environment": {"hdri": "dark_hangar", "intensity": 1.0, "rotation": 0.0,
                    "background": "hdri"},
    "light": {"preset": "scifi_spot", "azimuth": 200.0, "elevation": 24.0,
              "intensity": 1.0, "cone": 40.0, "color": [0.85, 0.93, 1.0]},
}
