// Editor viewport: view transform, hit testing, panel move/resize/draw, overlays.

import { snap, uid, fmt, clamp } from './util.js';

const HANDLE = 7;            // half size of a resize handle (css px)
const MIN_SIZE = 0.01;       // smallest panel side (m)
const HANDLES = ['tl', 't', 'tr', 'r', 'br', 'b', 'bl', 'l'];
const CURSORS = { tl: 'nwse-resize', br: 'nwse-resize', tr: 'nesw-resize', bl: 'nesw-resize',
                  t: 'ns-resize', b: 'ns-resize', l: 'ew-resize', r: 'ew-resize' };

// Mirrors core/panel.py Detail.centers, for overlay markers only.
export function detailCenters(d, x0, y0, x1, y1) {
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  const a = d.anchor;
  let base;
  if (a === 'corners') {
    base = [[x0 + d.ox, y0 + d.oy], [x1 - d.ox, y0 + d.oy], [x1 - d.ox, y1 - d.oy], [x0 + d.ox, y1 - d.oy]];
  } else if (a === 'left-right') {
    base = [[x0 + d.ox, cy + d.oy], [x1 - d.ox, cy + d.oy]];
  } else if (a === 'top-bottom') {
    base = [[cx + d.ox, y0 + d.oy], [cx + d.ox, y1 - d.oy]];
  } else {
    const px = a.endsWith('l') ? x0 + d.ox : a.endsWith('r') ? x1 - d.ox : cx + d.ox;
    const py = a.startsWith('t') ? y0 + d.oy : a.startsWith('b') ? y1 - d.oy : cy + d.oy;
    base = [[px, py]];
  }
  const n = Math.max(1, d.count | 0);
  if (n === 1) return base;
  const r = (d.rotation || 0) * Math.PI / 180;
  const dx = -Math.sin(r), dy = Math.cos(r);
  const out = [];
  for (const [bx, by] of base) {
    for (let k = 0; k < n; k++) {
      const s = (k - (n - 1) / 2) * d.spacing;
      out.push([bx + dx * s, by + dy * s]);
    }
  }
  return out;
}

export class Editor {
  constructor({ viewport, overlay, glview, store, onCursor, onToolChange, newPanelStyle }) {
    this.viewport = viewport;
    this.overlay = overlay;
    this.ctx = overlay.getContext('2d');
    this.gl = glview;
    this.store = store;
    this.onCursor = onCursor || (() => {});
    this.onToolChange = onToolChange || (() => {});
    this.newPanelStyle = newPanelStyle;
    this.view = { scale: 200, ox: 20, oy: 20 };
    this.tool = 'select';
    this.mode = 'lit';
    this.tiles = 1;
    this.light = null;           // lighting for the GL view
    this.lightGizmo = [];        // [{x, y}] in canvas meters
    this.hrange = [0, 0];
    this.emis = { scale: 0, spillScale: 0, spill: true, spillGain: 1 };
    this.bloom = { enabled: true, threshold: 1.5, intensity: 0.2, radius: 1 };
    this.drag = null;
    this.hover = null;
    this.space = false;
    this.fitted = false;
    this._raf = 0;

    new ResizeObserver(() => this.resize()).observe(viewport);
    overlay.addEventListener('pointerdown', (e) => this._down(e));
    overlay.addEventListener('pointermove', (e) => this._move(e));
    overlay.addEventListener('pointerup', (e) => this._up(e));
    overlay.addEventListener('pointercancel', (e) => this._up(e));
    overlay.addEventListener('pointerleave', () => { this.hover = null; this.onCursor(null); this.requestDraw(); });
    overlay.addEventListener('wheel', (e) => this._wheel(e), { passive: false });
    overlay.addEventListener('contextmenu', (e) => e.preventDefault());
  }

  get doc() { return this.store.doc; }

  setTool(t) {
    this.tool = t;
    if (t !== 'wire' && this.wires) this.wires.cancel();
    this.overlay.style.cursor = t === 'draw' || t === 'wire' ? 'crosshair' : 'default';
    this.onToolChange(t);
  }

  resize() {
    const r = this.viewport.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.cssW = r.width; this.cssH = r.height; this.dpr = dpr;
    this.gl.resize(r.width, r.height, dpr);
    this.overlay.width = Math.round(r.width * dpr);
    this.overlay.height = Math.round(r.height * dpr);
    if (!this.fitted && r.width > 10) { this.fit(); this.fitted = true; }
    this.requestDraw();
  }

  fit() {
    const W = this.doc.canvas_w * this.tiles, H = this.doc.canvas_h * this.tiles;
    const s = Math.min(this.cssW / W, this.cssH / H) * 0.92;
    this.view.scale = s;
    this.view.ox = (this.cssW - this.doc.canvas_w * s) / 2;
    this.view.oy = (this.cssH - this.doc.canvas_h * s) / 2;
    this.requestDraw();
  }

  toWorld(sx, sy) { return [(sx - this.view.ox) / this.view.scale, (sy - this.view.oy) / this.view.scale]; }
  toScreen(x, y) { return [this.view.ox + x * this.view.scale, this.view.oy + y * this.view.scale]; }

  _pt(e) {
    const r = this.overlay.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  _snap(v, e) { return e.altKey ? v : snap(v, this.store.prefs.snap); }

  // Offsets at which a panel is drawn/hit: wrapped copies when tiling.
  _offsets() {
    const { canvas_w: W, canvas_h: H, tiling } = this.doc;
    if (!tiling && this.tiles === 1) return [[0, 0]];
    const out = [];
    for (const j of [-1, 0, 1]) for (const i of [-1, 0, 1]) out.push([i * W, j * H]);
    return out;
  }

  hitPanel(x, y) {
    const panels = this.doc.panels;
    const offs = this._offsets();
    for (let i = panels.length - 1; i >= 0; i--) {
      const p = panels[i];
      if (!p.visible) continue;
      for (const [ox, oy] of offs) {
        if (x >= p.x + ox && x <= p.x + p.w + ox && y >= p.y + oy && y <= p.y + p.h + oy) return { p, ox, oy };
      }
    }
    return null;
  }

  hitHandle(sx, sy) {
    const p = this.store.selectedPanel;
    if (!p || p.locked || !p.visible) return null;
    for (const h of HANDLES) {
      const [hx, hy] = this._handlePos(p, h);
      if (Math.abs(sx - hx) <= HANDLE + 2 && Math.abs(sy - hy) <= HANDLE + 2) return h;
    }
    return null;
  }

  _handlePos(p, h) {
    const [x0, y0] = this.toScreen(p.x, p.y);
    const [x1, y1] = this.toScreen(p.x + p.w, p.y + p.h);
    const x = h.includes('l') ? x0 : h.includes('r') ? x1 : (x0 + x1) / 2;
    const y = h.includes('t') ? y0 : h.includes('b') ? y1 : (y0 + y1) / 2;
    return [x, y];
  }

  // ---- pointer interaction -------------------------------------------
  _down(e) {
    this.overlay.setPointerCapture(e.pointerId);
    const [sx, sy] = this._pt(e);
    const [wx, wy] = this.toWorld(sx, sy);
    if (e.button === 1 || e.button === 2 || this.space) {
      this.drag = { kind: 'pan', sx, sy, ox: this.view.ox, oy: this.view.oy };
      return;
    }
    if (e.button !== 0) return;
    if (this.wires && this.wires.downFirst(e, sx, sy, wx, wy)) return;
    if (this.tool === 'draw') {
      const x = this._snap(wx, e), y = this._snap(wy, e);
      this.drag = { kind: 'draw', x0: x, y0: y, x1: x, y1: y };
      return;
    }
    const h = this.hitHandle(sx, sy);
    if (h) {
      const p = this.store.selectedPanel;
      this.drag = { kind: 'resize', h, id: p.id, r0: [p.x, p.y, p.x + p.w, p.y + p.h], moved: false };
      return;
    }
    if (this.wires && this.wires.downSelect(e, sx, sy, wx, wy)) return;
    const hit = this.hitPanel(wx, wy);
    if (hit) {
      this.store.select(hit.p.id, 'editor');
      if (!hit.p.locked) {
        this.drag = { kind: 'move', id: hit.p.id, wx, wy, x0: hit.p.x, y0: hit.p.y, moved: false };
      }
      this.requestDraw();
      return;
    }
    this.store.select(null, 'editor');
    this.drag = { kind: 'pan', sx, sy, ox: this.view.ox, oy: this.view.oy };
    this.requestDraw();
  }

  _move(e) {
    const [sx, sy] = this._pt(e);
    const [wx, wy] = this.toWorld(sx, sy);
    this.onCursor([wx, wy]);
    const d = this.drag;
    if (this.wires && (this.wires.drag || this.tool === 'wire')) { this.wires.move(e, sx, sy, wx, wy); return; }
    if (!d) {
      if (this.tool === 'select' && !this.space) {
        const h = this.hitHandle(sx, sy);
        const onWire = !h && this.wires && this.wires.move(e, sx, sy, wx, wy);
        const hit = h || onWire ? null : this.hitPanel(wx, wy);
        this.overlay.style.cursor = h ? CURSORS[h] : onWire ? 'pointer' : hit && !hit.p.locked ? 'move' : 'default';
        const hv = hit ? hit.p.id : null;
        if (hv !== this.hover) { this.hover = hv; this.requestDraw(); }
      }
      return;
    }
    if (d.kind === 'pan') {
      this.view.ox = d.ox + (sx - d.sx);
      this.view.oy = d.oy + (sy - d.sy);
      this.requestDraw();
      return;
    }
    if (d.kind === 'draw') {
      d.x1 = this._snap(wx, e); d.y1 = this._snap(wy, e);
      this.requestDraw();
      return;
    }
    const p = this.store.panel(d.id);
    if (!p) return;
    if (!d.moved) { this.store.checkpoint(); d.moved = true; }
    if (d.kind === 'move') {
      p.x = this._snap(d.x0 + (wx - d.wx), e);
      p.y = this._snap(d.y0 + (wy - d.wy), e);
    } else if (d.kind === 'resize') {
      let [x0, y0, x1, y1] = d.r0;
      const x = this._snap(wx, e), y = this._snap(wy, e);
      if (d.h.includes('l')) x0 = Math.min(x, x1 - MIN_SIZE);
      if (d.h.includes('r')) x1 = Math.max(x, x0 + MIN_SIZE);
      if (d.h.includes('t')) y0 = Math.min(y, y1 - MIN_SIZE);
      if (d.h.includes('b')) y1 = Math.max(y, y0 + MIN_SIZE);
      // Only the rectangle changes: bevel, corners and details keep their meters.
      p.x = x0; p.y = y0; p.w = x1 - x0; p.h = y1 - y0;
    }
    this.store.emit('live', 'editor');
    this.requestDraw();
  }

  _up(e) {
    if (this.wires && this.wires.up()) { this.requestDraw(); return; }
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    if (d.kind === 'draw') {
      const x0 = Math.min(d.x0, d.x1), y0 = Math.min(d.y0, d.y1);
      const w = Math.abs(d.x1 - d.x0), h = Math.abs(d.y1 - d.y0);
      if (w >= MIN_SIZE && h >= MIN_SIZE) {
        const style = this.newPanelStyle();
        const p = Object.assign(style, { id: uid(), name: `Panel ${this.doc.panels.length + 1}`, x: x0, y: y0, w, h });
        this.store.mutate((doc) => doc.panels.push(p), 'editor');
        this.store.select(p.id);
      }
      this.setTool('select');
      this.requestDraw();
      return;
    }
    if ((d.kind === 'move' || d.kind === 'resize') && d.moved) this.store.emit('doc', 'editor');
    this.requestDraw();
  }

  _wheel(e) {
    e.preventDefault();
    const [sx, sy] = this._pt(e);
    const [wx, wy] = this.toWorld(sx, sy);
    const k = Math.exp(-e.deltaY * 0.0015);
    const texel = this.doc.texel_density;
    const minS = Math.min(this.cssW / this.doc.canvas_w, this.cssH / this.doc.canvas_h) * 0.1;
    this.view.scale = clamp(this.view.scale * k, minS, texel * 48);
    this.view.ox = sx - wx * this.view.scale;
    this.view.oy = sy - wy * this.view.scale;
    this.requestDraw();
  }

  zoomLabel() {
    const pxPerTexel = this.view.scale / this.doc.texel_density;
    return `${fmt(pxPerTexel * 100, 0)}%`;
  }

  // ---- drawing --------------------------------------------------------
  requestDraw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => { this._raf = 0; this.draw(); });
  }

  draw() {
    if (!this.cssW) return;
    const { canvas_w: W, canvas_h: H } = this.doc;
    if (this.light) {
      this.gl.draw({
        mode: this.mode, canvasM: [W, H], offset: [this.view.ox, this.view.oy],
        scale: this.view.scale, viewport: [this.cssW, this.cssH], tiles: this.tiles,
        hrange: this.hrange, light: this.light, emis: this.emis, bloom: this.bloom,
      });
    }
    const ctx = this.ctx;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.cssW, this.cssH);
    const s = this.view.scale;
    const [cx0, cy0] = this.toScreen(0, 0);

    // snap grid
    const g = this.store.prefs.snap;
    if (g > 0 && g * s >= 10) {
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x <= W + 1e-9; x += g) { const X = Math.round(cx0 + x * s) + 0.5; ctx.moveTo(X, cy0); ctx.lineTo(X, cy0 + H * s); }
      for (let y = 0; y <= H + 1e-9; y += g) { const Y = Math.round(cy0 + y * s) + 0.5; ctx.moveTo(cx0, Y); ctx.lineTo(cx0 + W * s, Y); }
      ctx.stroke();
    }

    // canvas border
    ctx.strokeStyle = this.doc.tiling ? 'rgba(57,198,230,0.55)' : 'rgba(255,255,255,0.35)';
    ctx.setLineDash(this.doc.tiling ? [6, 4] : []);
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(cx0) + 0.5, Math.round(cy0) + 0.5, W * s, H * s);
    ctx.setLineDash([]);

    // panel outlines (clipped to the canvas, wrapped when tiling)
    ctx.save();
    if (this.tiles === 1) { ctx.beginPath(); ctx.rect(cx0, cy0, W * s, H * s); ctx.clip(); }
    const sel = this.store.selected;
    const offs = this._offsets();
    for (const p of this.doc.panels) {
      if (!p.visible) continue;
      const isSel = p.id === sel, isHover = p.id === this.hover;
      ctx.strokeStyle = isSel ? '#39c6e6' : isHover ? 'rgba(255,255,255,0.75)' : 'rgba(255,255,255,0.16)';
      ctx.lineWidth = isSel ? 1.5 : 1;
      ctx.setLineDash(p.locked ? [4, 3] : []);
      for (const [ox, oy] of offs) {
        const [x, y] = this.toScreen(p.x + ox, p.y + oy);
        ctx.strokeRect(x, y, p.w * s, p.h * s);
      }
    }
    ctx.setLineDash([]);
    ctx.restore();

    if (this.wires) this.wires.draw(ctx);
    const p = this.store.selectedPanel;
    if (p && p.visible) this._drawSelection(p);
    if (this.drag && this.drag.kind === 'draw') {
      const d = this.drag;
      const [x0, y0] = this.toScreen(Math.min(d.x0, d.x1), Math.min(d.y0, d.y1));
      const w = Math.abs(d.x1 - d.x0) * s, h = Math.abs(d.y1 - d.y0) * s;
      ctx.fillStyle = 'rgba(57,198,230,0.12)';
      ctx.strokeStyle = '#39c6e6';
      ctx.fillRect(x0, y0, w, h);
      ctx.strokeRect(x0 + 0.5, y0 + 0.5, w, h);
      this._label(`${fmt(Math.abs(d.x1 - d.x0) * 100, 1)} × ${fmt(Math.abs(d.y1 - d.y0) * 100, 1)} cm`, x0 + w / 2, y0 + h + 14);
    }
    if (this.mode === 'lit') this._drawLightGizmo();
  }

  _drawSelection(p) {
    const ctx = this.ctx, s = this.view.scale;
    const [x0, y0] = this.toScreen(p.x, p.y);
    const w = p.w * s, h = p.h * s;
    // Inner edge of the bevel, which stays a constant world distance on resize.
    const bw = p.bevel.width * s;
    if (bw > 2 && bw * 2 < Math.min(w, h)) {
      ctx.strokeStyle = 'rgba(57,198,230,0.45)';
      ctx.setLineDash([3, 3]);
      ctx.strokeRect(x0 + bw, y0 + bw, w - 2 * bw, h - 2 * bw);
      ctx.setLineDash([]);
    }
    // detail anchors
    ctx.fillStyle = 'rgba(240,176,60,0.9)';
    for (const d of p.details) {
      for (const [dx, dy] of detailCenters(d, p.x, p.y, p.x + p.w, p.y + p.h)) {
        const [X, Y] = this.toScreen(dx, dy);
        ctx.fillRect(X - 1.5, Y - 1.5, 3, 3);
      }
    }
    if (!p.locked) {
      ctx.fillStyle = '#0b0e12';
      ctx.strokeStyle = '#39c6e6';
      ctx.lineWidth = 1.5;
      for (const hd of HANDLES) {
        const [hx, hy] = this._handlePos(p, hd);
        ctx.fillRect(hx - HANDLE / 2 - 1, hy - HANDLE / 2 - 1, HANDLE + 2, HANDLE + 2);
        ctx.strokeRect(hx - HANDLE / 2 - 1, hy - HANDLE / 2 - 1, HANDLE + 2, HANDLE + 2);
      }
    }
    if (this.drag && (this.drag.kind === 'move' || this.drag.kind === 'resize')) {
      this._label(`${fmt(p.w * 100, 1)} × ${fmt(p.h * 100, 1)} cm`, x0 + w / 2, y0 + h + 16);
    }
  }

  _label(text, x, y) {
    const ctx = this.ctx;
    ctx.font = '11px ui-monospace, Consolas, monospace';
    const tw = ctx.measureText(text).width;
    ctx.fillStyle = 'rgba(0,0,0,0.75)';
    ctx.fillRect(x - tw / 2 - 4, y - 11, tw + 8, 15);
    ctx.fillStyle = '#d6dde6';
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y);
    ctx.textAlign = 'start';
  }

  _drawLightGizmo() {
    const ctx = this.ctx;
    const [ax, ay] = this.toScreen(this.doc.canvas_w / 2, this.doc.canvas_h / 2);
    this.lightGizmo.forEach((g, i) => {
      const [x, y] = this.toScreen(g.x, g.y);
      ctx.strokeStyle = i === 0 ? 'rgba(220,240,255,0.8)' : 'rgba(57,198,230,0.5)';
      ctx.setLineDash([2, 4]);
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ax, ay); ctx.stroke();
      ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(x, y, i === 0 ? 7 : 5, 0, Math.PI * 2); ctx.stroke();
      if (i === 0) { ctx.fillStyle = 'rgba(220,240,255,0.9)'; ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI * 2); ctx.fill(); }
    });
  }
}
