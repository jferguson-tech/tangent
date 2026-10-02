# Tangent

[![tests](https://github.com/jferguson-tech/tangent/actions/workflows/tests.yml/badge.svg)](https://github.com/jferguson-tech/tangent/actions/workflows/tests.yml)

A browser-based normal map tool for sci-fi panels, served by Flask. Panels are
parametric: bevels, corners, grooves and details are stored in meters, so
resizing a panel never changes its edge treatment. A multi-threaded CPU
pathtracer renders a physically based preview under a sci-fi spotlight.
Weathering (edge wear, dirt, water streaks, rust, liquid leaks, heat and soot,
wire wear, and hand-painted strokes) is simulated on the server and exported
as base color, roughness and metallic maps plus per-effect masks.

![Output maps of a weathered panel: pathtraced preview, base color, normal, height, roughness, metallic, ambient occlusion, curvature, emissive, wire mask, panel ID and the nine weathering masks](docs/images/output-maps.png)

![The Tangent editor: tools, panel and wire lists and the wire simulation on the left, the lit 2D view of a weathered panel with leak markers above the pathtraced preview in the middle, and the Surface tab with the Leaking reactor weathering settings and leak list on the right](docs/images/editor-ui.jpg)

*The editor: tools, panels, wires and the wire simulation on the left, the lit
2D view and the pathtraced preview in the middle, and the Surface tab
(weathering, leaks, brush and materials) on the right.*

## Setup

Everything installs into the project venv `_env`. Nothing touches system Python.

```
python -m venv _env
_env\Scripts\python.exe -m pip install -r requirements.txt
_env\Scripts\python.exe -m pip install -r requirements-dev.txt   # tests only
```

## Run

```
_env\Scripts\python.exe app.py                    # this computer only: http://127.0.0.1:5000
_env\Scripts\python.exe app.py --host 0.0.0.0     # also other devices on your network
```

`run.bat` does the first one. `--port` picks another port (default 5000).

To use it from another device, start it with `--host 0.0.0.0`, find this PC's
address with `ipconfig` (the IPv4 address, such as `192.168.1.20`) and open
`http://192.168.1.20:5000` there. The first time, Windows asks whether Python
may accept connections: allow **Private networks** only. Read the Security
section below before serving on a network.

The first launch compiles the pathtracer, which takes a few seconds. Numba
caches the result for later runs.

## Tests

```
_env\Scripts\python.exe -m pytest
```

The wire simulation tests run under Node.js (`node --test tests/js/wiresim.test.mjs`);
pytest runs them too when Node is installed and skips them otherwise.

## Units

* Documents store every length in meters. The UI shows panel positions in cm
  and bevels, depths and details in mm.
* Texel density converts meters to pixels. The default is 512 px/m, with
  presets for 256, 1024 and 2048 px/m. Output resolution = canvas size × density.
* Every width field shows its pixel equivalent. A bevel, groove or detail
  bevel under 2 px is flagged because it will alias.

## Using it

| Action | How |
| --- | --- |
| Select, move | Click or drag a panel. Drag empty space to pan, wheel to zoom. |
| Resize | Drag a handle. Snaps to the grid; hold Alt to ignore it. |
| Draw a panel | `R`, then drag. It copies the selected panel's style. |
| Undo / redo | `Ctrl+Z` / `Ctrl+Y` |
| Duplicate, delete | `Ctrl+D`, `Delete` |
| Lock, hide | `L`, `H`. Locked panels survive auto-layout. |
| Stack order | `[` and `]` |
| Add a wire | `W`, then click start and end (Shift keeps the tool) |
| Paint weathering | `B`, then drag. Settings on the Surface tab. |
| Place a leak | `K`, then click; drag a leak to move it, `Delete` removes it |
| Nudge | Arrow keys, Shift for ×10 |
| Fit view | `F` |
| Orbit the render | Drag the render view, wheel to zoom, double-click to reset |

**Zoom detail.** The 2D view shows an overview of at most 1024 px across the
canvas. Zoom in further and the visible area is baked again at up to the full
texel density (never more than your screen can show) and drawn over the
overview, so higher densities are visible up close. The status bar shows the
patch size and density. Any edit drops the patch until the next bake, so it
never shows stale geometry. Exports always use the full density.

**Tiling.** Seamless mode is off by default, because a layout with a border or
seam already repeats cleanly. Turn it on when panels cross the canvas edge.
Their distance fields then wrap to the opposite side. Use *Tile ×3* in the
editor to check the seams.

**Environment (HDRI).** The pathtracer lights the panel with an equirectangular
HDRI plus the spotlights. Four procedural HDRIs ship built in: Dark hangar (the
default), Studio, Neon night and Overcast sky. They are generated on first use
and cached in `hdri/.cache/`. To add your own, drop Radiance `.hdr` files into
the `hdri/` folder or use *Upload .hdr* in the render settings. EXR is not
supported. Settings cover intensity, rotation, and the background (visible
HDRI by default, black, or blurred). The environment is importance-sampled and
combined with the material's own sampling, so bright strips and suns stay low
noise. With *Render lights* on, the editor view uses prefiltered versions of
the same HDRI.

**Lights (emission).** Details, inner grooves and panel faces can glow. Each
has a color (picker, swatches or color temperature) and a relative strength,
where 1 is about as bright as a white surface under the default spotlight.
Panel faces have a diffuser that fades the edges. Auto-layout adds light
strips, indicator dots, glowing grooves and light panels; set *Lights* to 0 to
turn that off. In the pathtraced view emitters really light the scene: their
light spills onto bevels, recesses, neighbors and the floor, shadowed by the
height field. In the realtime view emitters glow with bloom, and a baked,
blurred copy of the emission fakes the spill so nearby surfaces look lit.
*Scene light* in the render settings dims the spotlights and HDRI together.
Bloom is on by default in both views and has threshold, strength and radius
controls. The *Emissive* view mode shows the raw emission map.

**Wires.** Press `W`, click a start point and an end point (hold Shift to keep
adding). An end clicked on a panel attaches to it (nearest anchor plus an
offset), so it follows that panel when it moves or resizes; drag an end handle
to re-attach it. Each wire is either:

* *Simulated*: a rope that hangs under gravity, drapes over bevels, catches on
  details and, with *Collide* on, stacks on other wires. Use *Start / Pause /
  Resume / Step / Reset* in the left sidebar; while it runs you can drag a wire
  to pull it around. The gravity direction (default: down the image) and
  strength are set with the dial. The settled shape is saved in the document,
  so exports match what you saw. A wire whose panel moved is marked with an
  orange dot until it is re-simulated (or turn on *Auto settle*). Wires that
  have never been simulated, such as new auto-layout wires, settle on their own.
* *Routed*: a right-angle path (optionally with 45° runs) found by A* over the
  panels: it avoids raised details and bevel edges, prefers seams and grooves,
  minimises bends, and keeps clear of wires routed before it, so groups run in
  parallel. Bends are rounded to the bend radius.

Wires can be round tubes, ribbon cables or ribbed hoses, with connectors at the
ends, clips along the run, several parallel strands (a harness) and emission.
They have their own material in both previews (black rubber by default, see
*Wires* on the Surface tab) and an optional *Wire mask* export.
New wires default to a 14 mm ribbed hose. Auto-layout adds a few wires of all
kinds (*Wires*, *Simulated* and *Harnesses* settings), and *Shuffle wires*
re-rolls only the wires on the current panels, keeping locked wires.
The simulation runs in the browser (`static/js/wiresim.js`); the server only
bakes the saved shapes. A height field cannot hold empty space under a wire, so
a wire spanning a gap reads as resting on the surface below it.

**Surface and weathering.** The Surface tab sets the materials: bare metal,
an optional paint layer over it, the wires, and a small per-panel variation.
They drive both previews and the base color, roughness and metallic exports.
*Weathering* (off by default) ages the surface. Pick a preset (Clean, Field
use, Abandoned, Leaking reactor) or set *Age* and the strengths of each effect:

* *Edge wear* chips paint off convex edges and corners, or polishes bare metal.
* *Dirt* collects in crevices, corners and other occluded spots.
* *Streaks*: simulated water droplets run over the height field in the
  gravity direction (the same dial as the wire simulation). They slide along
  raised edges and leave vertical grime streaks below panels and details.
* *Rust* grows where water pools and in crevices, in patches. It pits the
  surface, blisters nearby paint, and stains the panels below with streaks
  carried by the same water.
* *Heat & soot*: vents and bright lights temper nearby bare metal (straw,
  bronze, purple, blue, closest to the source hottest) and scorch paint.
  Vents also leave soot that rises against gravity and spreads as it goes.
* *Wire wear*: wires rub the paint off the edges they rest on and get scuffed
  there, rubber fades and cracks with age, and rain running along a sagging
  wire drips from its lowest points, streaking the panels below.
* *Auto leaks* starts that many leaks at random bolts, vents and wire
  connectors (see below).

**Leaks.** With the Leak tool (`K`), click to place a leak; one placed on a
panel moves with it. Each leak pours water, oil or coolant (with a tint
color), and *Amount* sets how much has leaked. A shallow-liquid simulation
on 1 cm cells carries it down: liquid pools on ledges until it spills over,
follows seams and grooves, and wanders a little on the micro-roughness of the
surface. Water leaves a grimy trail with mineral rings at its edges (and
rusts what it runs over), oil a dark glossy stain, coolant a tinted crust,
and a still-running leak leaves its trail wet and glossy. Leaks don't depend
on *Age*.

**Brush.** With the Brush tool (`B`), paint rust, dirt, edge wear (chips),
grime streaks, oil, soot or heat tint onto the surface, or erase them
(*Erase* also removes what the simulation made, and *Everything* erases all
effects). Radius, strength and hardness for new strokes are on the Surface
tab and are remembered by the browser. A stroke painted on a panel is
attached to it (the topmost panel under most of the stroke) and moves with it;
one painted on empty canvas stays put.

Strokes stay editable. The *Strokes* list on the Surface tab shows them newest
first: click one to select it (it is outlined in the view) and change its
effect, mode, radius, strength or hardness, hide it with the dot button,
reorder it with the arrows (later strokes paint over earlier ones), or delete
it; `Delete` removes the selected stroke while the Brush tool is active, and
`Escape` deselects it. *Use as brush* copies a stroke's settings to the brush.
Strokes are stored as vectors in the document, so every change is undoable,
saved, and stays sharp at any texel density.

The simulations run on the server at up to 512 px/m (leaks at 100 px/m,
heat and soot at 128 px/m) and are cached in stages that depend only on their
own inputs, in parallel: the first bake of a heavily weathered 2 x 2 m canvas
takes about 3 seconds, and a brush stroke, a leak or a material change only
reruns what it affects. Fine detail (chip edges, pits, rust color) is noise
anchored to the canvas, so detail patches, exports and tiling canvases all
line up. *New seed* re-rolls the pattern. While you drag a panel the 2D view
shows the materials without weathering; it comes back when you let go. The
*Color* and *Rough* view modes show the base color and roughness maps.

**Export.** The export dialog writes a zip with the normal map (OpenGL or
DirectX), height, AO, curvature, a panel ID mask and an emissive map as PNG,
plus the document JSON and an info file. The base color (sRGB), roughness and
metallic maps are on by default; you can also export an ORM map (occlusion,
roughness and metallic in the R, G and B channels) and the weathering masks
as grayscale images: wear, dirt, streaks, rust, wet, fluid (oil and coolant
stains), residue (mineral rings and crust), soot and heat. The info file gives the
height range in millimeters and the emissive intensity to use in your engine,
because the emissive PNG is normalized so its brightest texel is white.

## Layout

```
app.py                  Flask routes; the pathtracer is optional
core/
  units.py              texel density, alias threshold
  sdf.py                rounded/chamfered box, circle, hexagon distance fields
  profiles.py           bevel profiles (linear, round, cove, smooth, ogee, stepped, custom)
  panel.py              Panel, Bevel, Groove, Detail (parametric, meters)
  document.py           canvas, density, tiling, panel stack, JSON
  layout.py             auto-layout (guillotine splits, symmetry, nesting, details, wires)
  wire.py               Wire model, endpoints, tube/ribbon/hose baking
  route.py              Manhattan A* wire routing (cached)
  bake.py               height -> normal, AO, curvature, ID, emissive, material and mask maps
  weather.py            materials and weathering (wear, dirt, water, rust, heat, soot, wire wear)
  fluids.py             leak sources and the shallow-liquid simulation
  brush.py              hand-painted weathering strokes
  filters.py            blur, bloom, sRGB helpers
  png.py                8/16-bit PNG writer
generators/
  base.py               Generator interface and pixel grid with wrap-around
  panels.py             panel stack compositor (future: water, ocean, flow)
render/
  presets.py            light and material presets (no Numba)
  hdri.py               .hdr reader, procedural HDRIs, sampling tables, editor prefiltering
  scene.py              builds pathtracer inputs from a document (no Numba)
  base.py               Renderer interface (future GPU backends)
  cpu.py, cpu_pathtracer.py   Numba CPU pathtracer, the only Numba code
  manager.py            background progressive render jobs
static/, templates/     editor UI (vanilla JS, WebGL2); js/wiresim.js is the wire simulation,
                        js/surface.js the Surface tab, js/weathertools.js the brush and leak tools
tests/
```

If Numba is missing or fails to load, the editor, WebGL preview and exporters
keep working. The render pane then says the pathtracer is unavailable.

## Security

Tangent has **no login**. Anyone who can reach its port can use every feature,
including exports, renders and HDRI uploads. By default it listens only on your
own machine (`127.0.0.1`).

**Serving to other devices** (`--host 0.0.0.0`) is meant for a network you trust,
such as your home LAN:

* Never expose the port to the internet (no port forwarding, no public cloud VM
  without a firewall or a password-protecting proxy in front).
* When Windows asks whether Python may accept connections, allow **Private**
  networks only, so the app is unreachable on cafe or hotel Wi-Fi.
* Open it on other devices by IP address (`http://192.168.x.x:5000`) or by this
  machine's name (`http://YOUR-PC-NAME:5000`). Other host names are refused
  unless you add them with `--allow-host NAME` or the `TANGENT_ALLOWED_HOSTS`
  environment variable (comma separated). This blocks DNS-rebinding attacks
  from web pages.
* `--debug` only works with a local host: Flask's debugger can run code, so the
  app refuses to start with `--debug` and a network address.
* Traffic is plain HTTP, so documents and renders are visible to others on the
  same network.
* All devices share one pathtracer: a render started on one device restarts the
  render on another.
* There is no memory limit: a very large export (8192 px, or high supersampling)
  can use many gigabytes of RAM, and any device on the network can start one.

**What the server guards against**

* Only real `application/json` bodies are accepted by the JSON endpoints, and
  requests that change anything must come from the app's own page (Origin and
  `Sec-Fetch-Site` checks). Other web pages open in your browser cannot drive
  the server.
* Uploaded HDRIs must be valid Radiance `.hdr` files (128 MB each). Their names
  are sanitized so they stay inside `hdri/`, and the folder is capped at 50
  files and 2 GB.
* Documents are validated and clamped field by field, and the page never
  inserts names or other text as HTML.

## License

MIT, see [LICENSE](LICENSE).
