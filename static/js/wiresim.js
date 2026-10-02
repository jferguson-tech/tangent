// Wire simulation: Verlet ropes on the panel height field (no DOM; runs in the
// browser and under Node for tests).
//
// Coordinates are canvas meters, image space (x right, y down); z is height
// above the panel base. Gravity lies in the canvas plane (a wall-mounted
// panel); a small extra force presses wires onto the surface. Wires collide
// with the height field and, optionally, with each other.

const G = 9.81;
const PRESS = 3.0;          // m/s^2 toward the surface
const DAMP = 0.992;         // velocity kept per substep
const FRICTION = 0.3;       // in-plane velocity lost while touching the surface
const SUBSTEPS = 8;     // many small steps stretch far less than many iterations
const ITERS = 4;
const SETTLE_SPEED = 2e-5;  // m per frame; slower than this counts as resting
const SETTLE_FRAMES = 40;

export function polylineLength(P) {
  let L = 0;
  for (let i = 1; i < P.length; i++) L += Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]);
  return L;
}

export function resample(P, n) {
  const s = [0];
  for (let i = 1; i < P.length; i++) s.push(s[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1]));
  const L = s[s.length - 1];
  const out = [];
  let j = 0;
  for (let k = 0; k < n; k++) {
    const t = (L * k) / (n - 1);
    while (j < s.length - 2 && s[j + 1] < t) j++;
    const seg = s[j + 1] - s[j];
    const u = seg > 0 ? (t - s[j]) / seg : 0;
    out.push([P[j][0] + (P[j + 1][0] - P[j][0]) * u, P[j][1] + (P[j + 1][1] - P[j][1]) * u]);
  }
  return out;
}

// Depth of a parabola over chord L whose arc length is `target` (bisection).
export function sagDepth(L, target) {
  if (target <= L * (1 + 1e-9)) return 0;
  const arc = (d) => {
    let s = 0, px = 0, py = 0;
    for (let k = 1; k <= 64; k++) {
      const t = k / 64, x = L * t, y = 4 * d * t * (1 - t);
      s += Math.hypot(x - px, y - py); px = x; py = y;
    }
    return s;
  };
  let lo = 0, hi = target;
  for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; if (arc(m) < target) lo = m; else hi = m; }
  return (lo + hi) / 2;
}

// Parabolic sag of the right length (mirrors core/wire.py sag_curve).
export function sagCurve(a, b, slack, gravityAngle = 90, n = 48) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const out = [];
  if (L < 1e-9) { for (let i = 0; i < n; i++) out.push([a[0], a[1]]); return out; }
  const depth = sagDepth(L, Math.max(slack, 1) * L);
  const ga = (gravityAngle * Math.PI) / 180;
  const g = [Math.cos(ga), Math.sin(ga)];
  const c = [(b[0] - a[0]) / L, (b[1] - a[1]) / L];
  const dot = g[0] * c[0] + g[1] * c[1];
  let p = [g[0] - c[0] * dot, g[1] - c[1] * dot];
  let pl = Math.hypot(p[0], p[1]);
  if (pl < 1e-6) { p = [-c[1], c[0]]; pl = 1; }
  p = [p[0] / pl, p[1] / pl];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1), s = 4 * depth * t * (1 - t);
    out.push([a[0] + (b[0] - a[0]) * t + p[0] * s, a[1] + (b[1] - a[1]) * t + p[1] * s]);
  }
  return out;
}

export class WireSim {
  constructor() {
    this.wires = [];
    this.field = null;
    this.params = { gravityAngle: 90, gravity: 1, collide: true };
    this.grab = null;
    this.quietFrames = 0;
    this.lastSpeed = Infinity;
  }

  // field: { w, h, data: Float32Array (row 0 = top), width, height (meters), wrap }
  setField(field) { this.field = field; }

  setParams(p) { Object.assign(this.params, p); this.quietFrames = 0; }

  sample(x, y) {
    const f = this.field;
    if (!f) return [0, 0, 0];
    const fx = (x / f.width) * f.w - 0.5, fy = (y / f.height) * f.h - 0.5;
    const x0 = Math.floor(fx), y0 = Math.floor(fy);
    const tx = fx - x0, ty = fy - y0;
    const W = f.w, H = f.h, d = f.data;
    let xa, xb, ya, yb;
    if (f.wrap) {
      xa = ((x0 % W) + W) % W; xb = (xa + 1) % W;
      ya = ((y0 % H) + H) % H; yb = (ya + 1) % H;
    } else {
      xa = x0 < 0 ? 0 : x0 >= W ? W - 1 : x0; xb = x0 + 1 < 0 ? 0 : x0 + 1 >= W ? W - 1 : x0 + 1;
      ya = y0 < 0 ? 0 : y0 >= H ? H - 1 : y0; yb = y0 + 1 < 0 ? 0 : y0 + 1 >= H ? H - 1 : y0 + 1;
    }
    const a = d[ya * W + xa], b = d[ya * W + xb], c = d[yb * W + xa], e = d[yb * W + xb];
    const h = (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + e * tx) * ty;
    const sx = f.width / W, sy = f.height / H;
    const gx = ((b - a) * (1 - ty) + (e - c) * ty) / sx;
    const gy = ((c - a) * (1 - tx) + (e - b) * tx) / sy;
    return [h, gx, gy];
  }

  // specs: [{ id, a: [x, y], b: [x, y], points: [[x, y]...], radius (in-plane), thick (z), slack }]
  setWires(specs) {
    this.wires = specs.map((s) => this._make(s));
    this.quietFrames = 0;
  }

  _make(s) {
    const chord = Math.hypot(s.b[0] - s.a[0], s.b[1] - s.a[1]);
    const L = Math.max(chord * Math.max(1, s.slack || 1), 1e-4);
    const r = Math.max(s.radius, 1e-4);
    const n = Math.max(12, Math.min(96, Math.round(L / (2.5 * r))));
    let start = s.points && s.points.length >= 2 ? s.points.map((p) => [...p]) : sagCurve(s.a, s.b, s.slack);
    // Keep the stored shape but move its ends onto the current endpoints.
    const P0 = start[0], P1 = start[start.length - 1];
    const da = [s.a[0] - P0[0], s.a[1] - P0[1]], db = [s.b[0] - P1[0], s.b[1] - P1[1]];
    const total = polylineLength(start) || 1;
    let acc = 0;
    start = start.map((p, i) => {
      if (i) acc += Math.hypot(p[0] - start[i - 1][0], p[1] - start[i - 1][1]);
      const w = acc / total;
      return [p[0] + da[0] * (1 - w) + db[0] * w, p[1] + da[1] * (1 - w) + db[1] * w];
    });
    const pts = resample(start, n);
    const w = {
      id: s.id, n, r, thick: Math.max(s.thick ?? s.radius, 1e-4), rest: L / (n - 1),
      a: [...s.a], b: [...s.b],
      x: new Float64Array(n), y: new Float64Array(n), z: new Float64Array(n),
      px: new Float64Array(n), py: new Float64Array(n), pz: new Float64Array(n),
      inv: new Float64Array(n).fill(1),
    };
    w.inv[0] = 0;
    w.inv[n - 1] = 0;
    for (let i = 0; i < n; i++) {
      const [x, y] = pts[i];
      const z = this.sample(x, y)[0] + w.thick;
      w.x[i] = w.px[i] = x;
      w.y[i] = w.py[i] = y;
      w.z[i] = w.pz[i] = z;
    }
    return w;
  }

  // Move an endpoint (e.g. a panel moved while simulating).
  setEnds(id, a, b) {
    const w = this.wires.find((q) => q.id === id);
    if (!w) return;
    w.a = [...a];
    w.b = [...b];
    this.quietFrames = 0;
  }

  // Grab the particle nearest (x, y) within maxDist meters. Returns true if one was grabbed.
  grabAt(x, y, maxDist) {
    let best = null, bd = maxDist;
    for (const w of this.wires) {
      for (let i = 1; i < w.n - 1; i++) {
        const d = Math.hypot(w.x[i] - x, w.y[i] - y);
        if (d < bd) { bd = d; best = { w, i }; }
      }
    }
    if (!best) return false;
    this.grab = { ...best, x, y };
    this.quietFrames = 0;
    return true;
  }

  dragTo(x, y) { if (this.grab) { this.grab.x = x; this.grab.y = y; this.quietFrames = 0; } }

  release() { this.grab = null; }

  get settled() { return this.quietFrames >= SETTLE_FRAMES; }

  step(dt = 1 / 60) {
    const h = dt / SUBSTEPS;
    const ga = (this.params.gravityAngle * Math.PI) / 180;
    const gm = G * this.params.gravity;
    const gx = Math.cos(ga) * gm * h * h, gy = Math.sin(ga) * gm * h * h, gz = -PRESS * h * h;
    const startX = this.wires.map((w) => Float64Array.from(w.x));
    const startY = this.wires.map((w) => Float64Array.from(w.y));
    for (let s = 0; s < SUBSTEPS; s++) {
      for (const w of this.wires) this._integrate(w, gx, gy, gz);
      for (let it = 0; it < ITERS; it++) for (const w of this.wires) this._constraints(w);
      if (this.params.collide && this.wires.length > 1) this._collideWires();
      for (const w of this.wires) this._collideField(w, true);
    }
    let speed = 0;
    this.wires.forEach((w, k) => {
      for (let i = 0; i < w.n; i++) speed = Math.max(speed, Math.hypot(w.x[i] - startX[k][i], w.y[i] - startY[k][i]));
    });
    this.lastSpeed = speed;
    this.quietFrames = speed < SETTLE_SPEED && !this.grab ? this.quietFrames + 1 : 0;
    return speed;
  }

  _integrate(w, gx, gy, gz) {
    for (let i = 0; i < w.n; i++) {
      if (w.inv[i] === 0) continue;
      const vx = (w.x[i] - w.px[i]) * DAMP, vy = (w.y[i] - w.py[i]) * DAMP, vz = (w.z[i] - w.pz[i]) * DAMP;
      w.px[i] = w.x[i]; w.py[i] = w.y[i]; w.pz[i] = w.z[i];
      w.x[i] += vx + gx;
      w.y[i] += vy + gy;
      w.z[i] += vz + gz;
    }
  }

  _pins(w) {
    const pin = (i, p) => {
      w.x[i] = p[0]; w.y[i] = p[1];
      w.z[i] = this.sample(p[0], p[1])[0] + w.thick;
    };
    pin(0, w.a);
    pin(w.n - 1, w.b);
    const g = this.grab;
    if (g && g.w === w) pin(g.i, [g.x, g.y]);
  }

  _constraints(w) {
    this._pins(w);
    const g = this.grab && this.grab.w === w ? this.grab.i : -1;
    const inv = (i) => (i === g ? 0 : w.inv[i]);
    // Stretch: neighbours keep their rest length.
    for (let i = 0; i < w.n - 1; i++) {
      const j = i + 1;
      const dx = w.x[j] - w.x[i], dy = w.y[j] - w.y[i], dz = w.z[j] - w.z[i];
      const d = Math.hypot(dx, dy, dz) || 1e-12;
      const wi = inv(i), wj = inv(j), ws = wi + wj;
      if (!ws) continue;
      const k = (d - w.rest) / (d * ws);
      w.x[i] += dx * k * wi; w.y[i] += dy * k * wi; w.z[i] += dz * k * wi;
      w.x[j] -= dx * k * wj; w.y[j] -= dy * k * wj; w.z[j] -= dz * k * wj;
    }
    // Long-range tethers: no particle may be farther from a pinned end than
    // the rope length between them. Cheap, and it stops visible stretching.
    for (let i = 1; i < w.n - 1; i++) {
      if (inv(i) === 0) continue;
      for (const [e, lim] of [[0, i * w.rest], [w.n - 1, (w.n - 1 - i) * w.rest]]) {
        const dx = w.x[i] - w.x[e], dy = w.y[i] - w.y[e], dz = w.z[i] - w.z[e];
        const d = Math.hypot(dx, dy, dz);
        if (d > lim && d > 0) {
          const k = lim / d;
          w.x[i] = w.x[e] + dx * k; w.y[i] = w.y[e] + dy * k; w.z[i] = w.z[e] + dz * k;
        }
      }
    }
    // Light bending stiffness: particles two apart may not fold together.
    const minD = w.rest * 1.7;
    for (let i = 0; i < w.n - 2; i++) {
      const j = i + 2;
      const dx = w.x[j] - w.x[i], dy = w.y[j] - w.y[i], dz = w.z[j] - w.z[i];
      const d = Math.hypot(dx, dy, dz) || 1e-12;
      if (d >= minD) continue;
      const wi = inv(i), wj = inv(j), ws = wi + wj;
      if (!ws) continue;
      const k = ((d - minD) / (d * ws)) * 0.5;
      w.x[i] += dx * k * wi; w.y[i] += dy * k * wi; w.z[i] += dz * k * wi;
      w.x[j] -= dx * k * wj; w.y[j] -= dy * k * wj; w.z[j] -= dz * k * wj;
    }
  }

  _collideField(w, friction) {
    for (let i = 0; i < w.n; i++) {
      if (w.inv[i] === 0) continue;
      const [hgt, hx, hy] = this.sample(w.x[i], w.y[i]);
      const pen = hgt + w.thick - w.z[i];
      if (pen <= 0) continue;
      // Push out along the surface normal: steep bevels push sideways.
      const nl = Math.hypot(hx, hy, 1);
      const nx = -hx / nl, ny = -hy / nl, nz = 1 / nl;
      const m = pen * nz;
      w.x[i] += nx * m; w.y[i] += ny * m; w.z[i] += nz * m;
      if (w.z[i] < hgt + w.thick) w.z[i] = hgt + w.thick;
      if (friction) {
        w.px[i] += (w.x[i] - w.px[i]) * FRICTION;
        w.py[i] += (w.y[i] - w.py[i]) * FRICTION;
      }
    }
  }

  _collideWires() {
    // Uniform grid built by counting sort (no per-step allocations once sized).
    const ws = this.wires;
    let total = 0, maxR = 0;
    for (const w of ws) { total += w.n; maxR = Math.max(maxR, w.r); }
    const cell = 2 * maxR, inv = 1 / cell;
    let size = 1;
    while (size < total * 2) size <<= 1;
    const mask = size - 1;
    if (!this._g || this._g.size !== size || this._g.total < total) {
      this._g = { size, total, start: new Int32Array(size + 1), items: new Int32Array(total * 2), cellOf: new Int32Array(total) };
    }
    const { start, items, cellOf } = this._g;
    const hash = (cx, cy) => (Math.imul(cx, 73856093) ^ Math.imul(cy, 19349663)) & mask;
    start.fill(0);
    let p = 0;
    for (let wi = 0; wi < ws.length; wi++) {
      const w = ws[wi];
      for (let i = 0; i < w.n; i++, p++) {
        const h = hash(Math.floor(w.x[i] * inv), Math.floor(w.y[i] * inv));
        cellOf[p] = h;
        start[h + 1]++;
      }
    }
    for (let h = 0; h < size; h++) start[h + 1] += start[h];
    const fill = Int32Array.from(start.subarray(0, size));
    p = 0;
    for (let wi = 0; wi < ws.length; wi++) {
      for (let i = 0; i < ws[wi].n; i++, p++) {
        const at = fill[cellOf[p]]++;
        items[at * 2] = wi;
        items[at * 2 + 1] = i;
      }
    }
    for (let wi = 0; wi < ws.length; wi++) {
      const w = ws[wi];
      for (let i = 0; i < w.n; i++) {
        const cx = Math.floor(w.x[i] * inv), cy = Math.floor(w.y[i] * inv);
        for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) {
          const h = hash(cx + ox, cy + oy);
          for (let q = start[h]; q < start[h + 1]; q++) {
            const wj = items[q * 2], j = items[q * 2 + 1];
            if (wj < wi || (wj === wi && j <= i + 3)) continue;
            const v = ws[wj];
            const dx = v.x[j] - w.x[i], dy = v.y[j] - w.y[i], dz = v.z[j] - w.z[i];
            const minD = w.r + v.r, d2 = dx * dx + dy * dy + dz * dz;
            if (d2 >= minD * minD) continue;
            const d = Math.sqrt(d2) || 1e-9;
            const a = w.inv[i], b = v.inv[j];
            if (!(a + b)) continue;
            // Separate mostly upward, so crossing wires stack instead of sliding
            // apart: the one already higher (or the later wire) goes on top.
            const up = Math.abs(dz) > 1e-7 ? Math.sign(dz) : 1;
            let ux = (0.3 * dx) / d, uy = (0.3 * dy) / d, uz = up;
            const ul = Math.hypot(ux, uy, uz);
            ux /= ul; uy /= ul; uz /= ul;
            const k = (minD - d) / (a + b);
            w.x[i] -= ux * k * a; w.y[i] -= uy * k * a; w.z[i] -= uz * k * a;
            v.x[j] += ux * k * b; v.y[j] += uy * k * b; v.z[j] += uz * k * b;
          }
        }
      }
    }
  }

  // Current shapes: { id: [[x, y], ...] } rounded to 0.01 mm.
  shapes() {
    const out = {};
    for (const w of this.wires) {
      const pts = [];
      for (let i = 0; i < w.n; i++) pts.push([Math.round(w.x[i] * 1e5) / 1e5, Math.round(w.y[i] * 1e5) / 1e5]);
      out[w.id] = pts;
    }
    return out;
  }

  // Simulated length of a wire (for tests and diagnostics).
  length(id) {
    const w = this.wires.find((q) => q.id === id);
    let L = 0;
    for (let i = 1; i < w.n; i++) L += Math.hypot(w.x[i] - w.x[i - 1], w.y[i] - w.y[i - 1], w.z[i] - w.z[i - 1]);
    return L;
  }
}
