// Wires in the editor: endpoint attachment, the wire tool, selection, endpoint
// dragging, grabbing wires while simulating, overlay drawing, and the
// simulation controller (start / pause / resume / step / reset).

import { clone, uid } from './util.js';
import { api } from './api.js';
import { WireSim, sagCurve } from './wiresim.js';

export const ANCHORS = ['tl', 't', 'tr', 'l', 'c', 'r', 'bl', 'b', 'br'];

export function anchorPoint(p, a) {
  const x = a.endsWith('l') ? p.x : a.endsWith('r') ? p.x + p.w : p.x + p.w / 2;
  const y = a.startsWith('t') ? p.y : a.startsWith('b') ? p.y + p.h : p.y + p.h / 2;
  return [x, y];
}

// Mirrors core/wire.py End.resolve.
export function resolveEnd(end, doc) {
  if (end.panel) {
    const p = doc.panels.find((q) => q.id === end.panel);
    if (p) {
      const [ax, ay] = anchorPoint(p, end.anchor || 'c');
      return [ax + (end.dx || 0), ay + (end.dy || 0)];
    }
  }
  return [end.x || 0, end.y || 0];
}

// An endpoint at (x, y): attached to `panel` (nearest anchor + offset) or free.
export function endAt(x, y, panel) {
  if (!panel) return { panel: null, anchor: 'c', dx: 0, dy: 0, x, y };
  let best = 'c', bd = Infinity;
  for (const a of ANCHORS) {
    const [ax, ay] = anchorPoint(panel, a);
    const d = Math.hypot(ax - x, ay - y);
    if (d < bd) { bd = d; best = a; }
  }
  const [ax, ay] = anchorPoint(panel, best);
  return { panel: panel.id, anchor: best, dx: x - ax, dy: y - ay, x, y };
}

export const halfWidth = (w) => w.radius * (w.profile === 'ribbon' ? 2.5 : 1);
export const strandSpacing = (w) => (w.bundle_spacing > 0 ? w.bundle_spacing : 2 * halfWidth(w) * 1.02);
export const totalHalfWidth = (w) => ((Math.max(1, w.bundle) - 1) * 0.5) * strandSpacing(w) + halfWidth(w);

export function defaultWire(doc, a, b, mode, style = {}) {
  const w = Object.assign({
    id: 'w' + uid('').slice(0, 8), name: `Wire ${doc.wires.length + 1}`, mode,
    profile: 'hose', radius: 0.014, slack: 1.15, corner_radius: 0.025, allow45: false,
    clearance: 0.01, connectors: true, clip_spacing: 0, bundle: 1, bundle_spacing: 0,
    emission: { enabled: false, color: [0.35, 0.85, 1.0], strength: 4.0, diffuser: 0 },
    points: [], settled: false, locked: false, visible: true,
  }, clone(style), { a, b, mode });
  w.id = 'w' + Math.random().toString(16).slice(2, 10);
  w.name = `Wire ${doc.wires.length + 1}`;
  w.points = [];
  w.settled = false;
  w.locked = false;
  w.visible = true;
  if (mode === 'sim') w.points = sagCurve(resolveEnd(a, doc), resolveEnd(b, doc), w.slack, doc.wire_sim.gravity_angle);
  return w;
}

export function isStale(w, doc) {
  if (w.mode !== 'sim') return false;
  if (!w.settled || !w.points || w.points.length < 2) return true;
  const a = resolveEnd(w.a, doc), b = resolveEnd(w.b, doc);
  const p0 = w.points[0], p1 = w.points[w.points.length - 1];
  return Math.max(Math.abs(a[0] - p0[0]), Math.abs(a[1] - p0[1]), Math.abs(b[0] - p1[0]), Math.abs(b[1] - p1[1])) > 1e-5;
}

function distToPolyline(P, x, y) {
  let best = Infinity;
  for (let i = 0; i < P.length - 1; i++) {
    const [ax, ay] = P[i], [bx, by] = P[i + 1];
    const dx = bx - ax, dy = by - ay;
    const L2 = dx * dx + dy * dy || 1e-18;
    const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L2));
    best = Math.min(best, Math.hypot(x - ax - t * dx, y - ay - t * dy));
  }
  return best;
}

function decodeF16(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const h = new Uint16Array(u8.buffer);
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const v = h[i], s = v & 0x8000 ? -1 : 1, e = (v >> 10) & 0x1f, m = v & 0x3ff;
    out[i] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// ---- simulation controller --------------------------------------------------------
export class SimController {
  constructor(store, { onFrame = () => {}, onState = () => {} } = {}) {
    this.store = store;
    this.sim = new WireSim();
    this.state = 'idle';          // idle | running | paused | settled
    this.onFrame = onFrame;
    this.onState = onState;
    this.signature = null;
    this.raf = 0;
    this.fieldStale = true;
    this.frames = 0;
    this.stepMs = 0;
  }

  get running() { return this.state === 'running'; }

  simWires(doc = this.store.doc) {
    return doc.wires.filter((w) => w.mode === 'sim' && w.visible && !w.locked);
  }

  // Wires whose geometry the simulation owns right now (hidden from bakes).
  activeIds() { return this.state === 'running' ? new Set(this.sim.wires.map((w) => w.id)) : new Set(); }

  _signature(doc) {
    return JSON.stringify(this.simWires(doc).map((w) => [w.id, w.radius, w.profile, w.slack, w.bundle, w.bundle_spacing]));
  }

  _spec(w, doc, points) {
    const thick = w.profile === 'ribbon' ? 0.4 * w.radius : w.radius;
    return { id: w.id, a: resolveEnd(w.a, doc), b: resolveEnd(w.b, doc), points, radius: totalHalfWidth(w), thick, slack: w.slack };
  }

  async _loadField() {
    const doc = clone(this.store.doc);
    for (const w of doc.wires) if (w.mode === 'sim') w.visible = false;   // sim wires rest on panels and routed wires
    const r = await api.bake(doc, ['height_raw'], 512);
    const f = r.maps.height_raw;
    this.sim.setField({ w: f.w, h: f.h, data: decodeF16(f.data), width: doc.canvas_w, height: doc.canvas_h, wrap: doc.tiling });
    this.fieldStale = false;
  }

  async _build() {
    const doc = this.store.doc;
    if (this.fieldStale) await this._loadField();
    const current = this.sim.shapes();
    this.sim.setWires(this.simWires(doc).map((w) => this._spec(w, doc, current[w.id] || w.points)));
    this._params();
    this.signature = this._signature(doc);
  }

  _params() {
    const s = this.store.doc.wire_sim;
    this.sim.setParams({ gravityAngle: s.gravity_angle, gravity: s.gravity, collide: s.collide });
  }

  _setState(s) { this.state = s; this.onState(s); }

  async start() {
    if (this.running) return;
    if (!this.simWires().length) { this._setState('idle'); return; }
    if (this.fieldStale || this.signature !== this._signature(this.store.doc) || !this.sim.wires.length) await this._build();
    this.syncEnds();
    this._params();
    this.sim.quietFrames = 0;
    this._setState('running');
    this.frames = 0;
    this._loop();
  }

  _loop() {
    cancelAnimationFrame(this.raf);
    const tick = () => {
      if (this.state !== 'running') return;
      const t0 = performance.now();
      this.sim.step(1 / 60);
      this.stepMs = this.stepMs * 0.9 + (performance.now() - t0) * 0.1;
      this.frames++;
      this.onFrame();
      if (this.sim.settled) {
        this.commit(true);
        this._setState('settled');
        return;
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  pause() {
    if (!this.running) return;
    cancelAnimationFrame(this.raf);
    this.sim.release();
    this.commit(this.sim.settled);
    this._setState('paused');
  }

  async step() {
    if (this.running) return;
    if (!this.simWires().length) return;
    if (this.fieldStale || this.signature !== this._signature(this.store.doc) || !this.sim.wires.length) await this._build();
    this.syncEnds();
    this._params();
    this.sim.step(1 / 60);
    this.commit(false);
    this._setState('paused');
  }

  // Back to the starting sag (the shape before any simulation).
  reset() {
    cancelAnimationFrame(this.raf);
    this.sim.setWires([]);
    this.signature = null;
    const doc = this.store.doc;
    this.store.mutate((d) => {
      for (const w of this.simWires(d)) {
        w.points = sagCurve(resolveEnd(w.a, d), resolveEnd(w.b, d), w.slack, d.wire_sim.gravity_angle);
        w.settled = false;
      }
    }, 'sim');
    this._setState('idle');
    return doc;
  }

  commit(settled) {
    const shapes = this.sim.shapes();
    if (!Object.keys(shapes).length) return;
    this.store.mutate((d) => {
      for (const w of d.wires) {
        if (shapes[w.id]) { w.points = shapes[w.id]; w.settled = !!settled; }
      }
    }, 'sim');
  }

  // Panels moved or wires edited while simulating.
  docChanged() {
    this.fieldStale = true;
    if (this.running) {
      if (this.signature !== this._signature(this.store.doc)) {
        cancelAnimationFrame(this.raf);
        this._setState('paused');
        this.start();
        return;
      }
      this._loadField().catch(() => {});
      this.syncEnds();
      this._params();
      this.sim.quietFrames = 0;
    }
  }

  syncEnds() {
    const doc = this.store.doc;
    for (const w of this.simWires(doc)) this.sim.setEnds(w.id, resolveEnd(w.a, doc), resolveEnd(w.b, doc));
  }

  liveShape(id) {
    if (this.state !== 'running') return null;
    const w = this.sim.wires.find((q) => q.id === id);
    if (!w) return null;
    const pts = [];
    for (let i = 0; i < w.n; i++) pts.push([w.x[i], w.y[i]]);
    return pts;
  }
}

// ---- editor layer ---------------------------------------------------------------------
export class WireLayer {
  constructor(editor, store, ctrl) {
    this.editor = editor;
    this.store = store;
    this.ctrl = ctrl;
    this.paths = {};          // id -> centerline from the last bake
    this.pending = null;      // wire tool: first endpoint
    this.cursor = null;
    this.drag = null;
    this.hover = null;
  }

  get doc() { return this.store.doc; }

  pathOf(w) {
    return this.ctrl.liveShape(w.id) || this.paths[w.id] || (w.points && w.points.length > 1 ? w.points : [resolveEnd(w.a, this.doc), resolveEnd(w.b, this.doc)]);
  }

  hit(x, y) {
    const tolPx = 6 / this.editor.view.scale;
    const ws = this.doc.wires;
    for (let i = ws.length - 1; i >= 0; i--) {
      const w = ws[i];
      if (!w.visible) continue;
      if (distToPolyline(this.pathOf(w), x, y) <= Math.max(totalHalfWidth(w), tolPx)) return w;
    }
    return null;
  }

  _endHandle(sx, sy) {
    const w = this.store.selectedWire;
    if (!w || w.locked) return null;
    for (const which of ['a', 'b']) {
      const [x, y] = this.editor.toScreen(...resolveEnd(w[which], this.doc));
      if (Math.hypot(sx - x, sy - y) <= 9) return which;
    }
    return null;
  }

  // Before panel handles: wire tool, grabbing a simulating wire, endpoint handles.
  downFirst(e, sx, sy, wx, wy) {
    const ed = this.editor;
    if (ed.tool === 'wire') {
      const x = ed._snap(wx, e), y = ed._snap(wy, e);
      const hit = ed.hitPanel(x, y);
      const end = endAt(x, y, hit ? hit.p : null);
      if (!this.pending) {
        this.pending = end;
        this.cursor = [x, y];
      } else {
        const mode = this.store.prefs.wireMode || 'sim';
        const style = this.store.selectedWire ? this.store.selectedWire : {};
        const { id, name, a, b, points, settled, locked, visible, mode: _m, ...keep } = style;
        const w = defaultWire(this.doc, this.pending, end, mode, keep);
        this.pending = null;
        this.store.mutate((d) => d.wires.push(w), 'editor');
        this.store.select(w.id);
        if (!e.shiftKey) ed.setTool('select');
      }
      ed.requestDraw();
      return true;
    }
    if (this.ctrl.running) {
      const tol = Math.max(8 / ed.view.scale, 0.01);
      if (this.ctrl.sim.grabAt(wx, wy, tol)) {
        this.drag = { kind: 'grab' };
        return true;
      }
    }
    const which = this._endHandle(sx, sy);
    if (which) {
      this.drag = { kind: 'end', which, id: this.store.selectedWire.id, moved: false };
      return true;
    }
    return false;
  }

  // After panel handles: selecting a wire by clicking on it.
  downSelect(e, sx, sy, wx, wy) {
    const w = this.hit(wx, wy);
    if (!w) return false;
    this.store.select(w.id, 'editor');
    this.editor.requestDraw();
    return true;
  }

  move(e, sx, sy, wx, wy) {
    const ed = this.editor;
    if (ed.tool === 'wire') {
      this.cursor = [ed._snap(wx, e), ed._snap(wy, e)];
      ed.requestDraw();
    }
    const d = this.drag;
    if (!d) {
      const hv = ed.tool === 'select' ? this.hit(wx, wy) : null;
      const id = hv ? hv.id : null;
      if (id !== this.hover) { this.hover = id; ed.requestDraw(); }
      return !!hv;
    }
    if (d.kind === 'grab') {
      this.ctrl.sim.dragTo(wx, wy);
      return true;
    }
    if (d.kind === 'end') {
      const w = this.doc.wires.find((q) => q.id === d.id);
      if (!w) return true;
      if (!d.moved) { this.store.checkpoint(); d.moved = true; }
      const x = ed._snap(wx, e), y = ed._snap(wy, e);
      const hit = ed.hitPanel(x, y);
      w[d.which] = endAt(x, y, hit ? hit.p : null);
      this.paths[w.id] = null;
      this.store.emit('live', 'editor');
      ed.requestDraw();
      return true;
    }
    return false;
  }

  up() {
    const d = this.drag;
    this.drag = null;
    if (!d) return false;
    if (d.kind === 'grab') this.ctrl.sim.release();
    if (d.kind === 'end' && d.moved) this.store.emit('doc', 'editor');
    return true;
  }

  cancel() { this.pending = null; this.editor.requestDraw(); }

  draw(ctx) {
    const ed = this.editor, s = ed.view.scale;
    const line = (P, width, color, dash = []) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.setLineDash(dash);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      P.forEach((p, i) => { const [x, y] = ed.toScreen(p[0], p[1]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
      ctx.setLineDash([]);
    };
    const sel = this.store.selected;
    const live = this.ctrl.activeIds();
    for (const w of this.doc.wires) {
      if (!w.visible) continue;
      const P = this.pathOf(w);
      if (P.length < 2) continue;
      if (live.has(w.id)) {
        // Live simulation: drawn as a shaded tube (the baked maps hide it).
        const hw = Math.max(totalHalfWidth(w) * s, 1);
        line(P, hw * 2 + 1, 'rgba(0,0,0,0.65)');
        line(P, hw * 2, '#1d2229');
        line(P, Math.max(hw * 0.7, 1), 'rgba(170,190,210,0.45)');
      }
      if (w.id === sel) line(P, 1.5, '#39c6e6');
      else if (w.id === this.hover) line(P, 1, 'rgba(255,255,255,0.75)');
      if (isStale(w, this.doc) && !live.has(w.id)) {
        const m = P[Math.floor(P.length / 2)];
        const [x, y] = ed.toScreen(m[0], m[1]);
        ctx.fillStyle = '#f0b03c';
        ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2); ctx.fill();
      }
    }
    const w = this.store.selectedWire;
    if (w && w.visible) {
      for (const which of ['a', 'b']) {
        const [x, y] = ed.toScreen(...resolveEnd(w[which], this.doc));
        ctx.fillStyle = w[which].panel ? '#39c6e6' : '#0b0e12';
        ctx.strokeStyle = '#39c6e6';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }
    if (ed.tool === 'wire' && this.pending && this.cursor) {
      const a = resolveEnd(this.pending, this.doc);
      const mode = this.store.prefs.wireMode || 'sim';
      const P = mode === 'sim' ? sagCurve(a, this.cursor, 1.15, this.doc.wire_sim.gravity_angle, 32) : [a, [this.cursor[0], a[1]], this.cursor];
      line(P, 1.5, '#39c6e6', [5, 4]);
    }
  }
}
