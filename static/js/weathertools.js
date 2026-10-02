// Weathering tools in the editor: the brush (paint or erase rust, dirt, wear,
// grime, oil, soot and heat by hand) and leak sources (click to place, drag to
// move). Strokes and leaks are part of the document; the server turns them
// into maps, so the overlay only previews the stroke being painted and
// outlines the selected one. A stroke painted on a panel moves with it.

import { storage } from './util.js';
import { endAt, resolveEnd } from './wires.js';

const BRUSH_KEY = 'tangent.brush';
export const brush = Object.assign({ channel: 'rust', mode: 'add', radius: 0.03, strength: 0.8, hardness: 0.5 },
  storage.get(BRUSH_KEY, {}));
export const saveBrush = () => storage.set(BRUSH_KEY, brush);

const CH_RGB = {
  rust: [200, 95, 35], dirt: [110, 85, 55], wear: [225, 230, 240], streaks: [70, 66, 58],
  oil: [25, 20, 12], soot: [12, 12, 12], heat: [90, 110, 235], all: [57, 198, 230],
};
export const LIQUID_CSS = { water: '#6cb8ff', oil: '#d2a650', coolant: '#4fe0a0' };
export const CHANNEL_CSS = Object.fromEntries(Object.entries(CH_RGB).map(([k, v]) => [k, `rgb(${v.join(',')})`]));
const HIT = 9;            // leak marker hit radius (css px)

// Mirrors core/brush.py resolve: the stroke's points, moved along with its panel.
export function strokePoints(s, doc) {
  if (s.panel && s.origin) {
    const p = doc.panels.find((q) => q.id === s.panel);
    if (p) {
      const dx = p.x - s.origin[0], dy = p.y - s.origin[1];
      if (dx || dy) return s.points.map(([x, y]) => [x + dx, y + dy]);
    }
  }
  return s.points;
}

export class WeatherTools {
  constructor(editor, store, { ensureEnabled = () => {}, onLeakSelect = () => {}, onStrokeSelect = () => {} } = {}) {
    this.editor = editor;
    this.store = store;
    this.ensureEnabled = ensureEnabled;
    this.onLeakSelect = onLeakSelect;
    this.onStrokeSelect = onStrokeSelect;
    this.selected = null;     // leak id
    this.selectedStroke = null;
    this.stroke = null;       // stroke being painted
    this.drag = null;         // leak being moved
    this.cursor = null;       // [sx, sy] for the brush outline
  }

  get tool() { return this.editor.tool; }

  get doc() { return this.store.doc; }

  selectLeak(id) {
    this.selected = id;
    this.onLeakSelect(id);
    this.editor.requestDraw();
  }

  selectStroke(id) {
    this.selectedStroke = id;
    this.onStrokeSelect(id);
    this.editor.requestDraw();
  }

  // The panel most of a stroke lies on (it is attached to it), or null.
  _panelUnder(points) {
    const votes = new Map();
    const step = Math.max(1, Math.floor(points.length / 24));
    for (let i = 0; i < points.length; i += step) {
      const hit = this.editor.hitPanel(points[i][0], points[i][1]);
      if (hit) votes.set(hit.p, (votes.get(hit.p) || 0) + 1);
    }
    let best = null, n = 0;
    for (const [p, v] of votes) if (v > n) { best = p; n = v; }
    return best;
  }

  leakAt(sx, sy) {
    const leaks = this.doc.leaks || [];
    for (let i = leaks.length - 1; i >= 0; i--) {
      const [x, y] = this.editor.toScreen(...resolveEnd(leaks[i], this.doc));
      if (Math.hypot(x - sx, y - sy) <= HIT) return leaks[i];
    }
    return null;
  }

  // Pointer handlers return true when they handled the event.
  down(e, sx, sy, wx, wy) {
    if (this.tool === 'brush') {
      this.ensureEnabled();
      const b = brush;
      this.stroke = { channel: b.mode === 'erase' && b.channel === 'all' ? 'all' : b.channel, mode: b.mode,
                      radius: b.radius, strength: b.strength, hardness: b.hardness, points: [[wx, wy]] };
      this.editor.requestDraw();
      return true;
    }
    if (this.tool !== 'leak') return false;
    const hit = this.leakAt(sx, sy);
    if (hit) {
      this.selectLeak(hit.id);
      this.drag = { id: hit.id, moved: false };
      return true;
    }
    this.ensureEnabled();
    const panel = this.editor.hitPanel(wx, wy);
    const leak = Object.assign({ id: 'k' + Math.random().toString(16).slice(2, 10), liquid: 'water', amount: 0.6,
                                 color: [0.10, 0.55, 0.32] }, endAt(wx, wy, panel ? panel.p : null));
    const prev = (this.doc.leaks || []).find((k) => k.id === this.selected);
    if (prev) Object.assign(leak, { liquid: prev.liquid, amount: prev.amount, color: [...prev.color] });
    this.store.mutate((d) => { d.leaks.push(leak); }, 'editor');
    this.selectLeak(leak.id);
    return true;
  }

  move(e, sx, sy, wx, wy) {
    if (this.tool === 'brush') {
      this.cursor = [sx, sy];
      const s = this.stroke;
      if (s) {
        const last = s.points[s.points.length - 1];
        if (Math.hypot(wx - last[0], wy - last[1]) >= Math.max(s.radius * 0.15, 0.5 / this.editor.view.scale)) {
          s.points.push([wx, wy]);
        }
      }
      this.editor.requestDraw();
      return true;
    }
    if (this.drag) {
      const k = (this.doc.leaks || []).find((q) => q.id === this.drag.id);
      if (!k) return true;
      if (!this.drag.moved) { this.store.checkpoint(); this.drag.moved = true; }
      const panel = this.editor.hitPanel(wx, wy);
      Object.assign(k, endAt(wx, wy, panel ? panel.p : null));
      this.editor.requestDraw();
      return true;
    }
    if (this.tool === 'leak') {
      this.editor.overlay.style.cursor = this.leakAt(sx, sy) ? 'move' : 'crosshair';
      return true;
    }
    return false;
  }

  up() {
    if (this.stroke) {
      const s = this.stroke;
      this.stroke = null;
      s.id = 's' + Math.random().toString(16).slice(2, 10);
      s.visible = true;
      const p = this._panelUnder(s.points);
      s.panel = p ? p.id : null;
      s.origin = p ? [p.x, p.y] : null;
      this.store.mutate((d) => { d.strokes.push(s); }, 'editor');
      this.selectStroke(s.id);
      return true;
    }
    if (this.drag) {
      const moved = this.drag.moved;
      this.drag = null;
      if (moved) this.store.emit('doc', 'editor');
      return true;
    }
    return false;
  }

  leave() {
    this.cursor = null;
    this.editor.requestDraw();
  }

  deleteSelectedStroke() {
    const id = this.selectedStroke;
    if (!id || !(this.doc.strokes || []).some((s) => s.id === id)) return false;
    this.store.mutate((d) => { d.strokes = d.strokes.filter((s) => s.id !== id); }, 'editor');
    this.selectStroke(null);
    return true;
  }

  deleteSelected() {
    if (!this.selected) return false;
    const id = this.selected;
    if (!(this.doc.leaks || []).some((k) => k.id === id)) return false;
    this.store.mutate((d) => { d.leaks = d.leaks.filter((k) => k.id !== id); }, 'editor');
    this.selectLeak(null);
    return true;
  }

  draw(ctx) {
    const ed = this.editor;
    const s = ed.view.scale;
    // Leak markers: a drop in the liquid's color, the selected one ringed.
    const leaks = this.doc.leaks || [];
    const strong = this.tool === 'leak';
    for (const k of leaks) {
      const [x, y] = ed.toScreen(...resolveEnd(k, this.doc));
      ctx.globalAlpha = strong ? 1 : 0.55;
      ctx.fillStyle = LIQUID_CSS[k.liquid] || '#fff';
      ctx.strokeStyle = 'rgba(0,0,0,0.7)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, y - 7);
      ctx.bezierCurveTo(x + 5, y - 1, x + 4, y + 4, x, y + 4);
      ctx.bezierCurveTo(x - 4, y + 4, x - 5, y - 1, x, y - 7);
      ctx.fill();
      ctx.stroke();
      if (k.id === this.selected && strong) {
        ctx.strokeStyle = '#39c6e6';
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(x, y - 1, 9, 0, Math.PI * 2); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    // The selected stroke: its footprint and path.
    const sel = this.selectedStroke && (this.doc.strokes || []).find((q) => q.id === this.selectedStroke);
    if (sel && !this.stroke) {
      const P = strokePoints(sel, this.doc).map(([x, y]) => ed.toScreen(x, y));
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const path = () => {
        ctx.beginPath();
        P.forEach(([X, Y], i) => (i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y)));
        if (P.length === 1) ctx.lineTo(P[0][0] + 0.01, P[0][1]);
      };
      path();
      ctx.lineWidth = Math.max(2, 2 * sel.radius * s);
      ctx.strokeStyle = sel.visible === false ? 'rgba(255,255,255,0.08)' : 'rgba(57,198,230,0.22)';
      ctx.stroke();
      path();
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = '#39c6e6';
      ctx.stroke();
      ctx.restore();
    }
    if (this.tool !== 'brush') return;
    const rgb = CH_RGB[brush.mode === 'erase' ? 'all' : brush.channel] || CH_RGB.all;
    // The stroke being painted.
    const st = this.stroke;
    if (st && st.points.length) {
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.lineWidth = Math.max(1, 2 * st.radius * s);
      ctx.strokeStyle = `rgba(${rgb.join(',')},${0.25 + 0.35 * st.strength})`;
      if (st.mode === 'erase') ctx.setLineDash([6, 4]);
      ctx.beginPath();
      st.points.forEach(([x, y], i) => {
        const [X, Y] = ed.toScreen(x, y);
        if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
      });
      if (st.points.length === 1) { const [X, Y] = ed.toScreen(...st.points[0]); ctx.lineTo(X + 0.01, Y); }
      ctx.stroke();
      ctx.restore();
    }
    // Brush outline: the radius, and the hard core inside it.
    if (this.cursor) {
      const [cx, cy] = this.cursor;
      const r = brush.radius * s;
      ctx.strokeStyle = 'rgba(255,255,255,0.85)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = `rgba(${rgb.join(',')},0.9)`;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, r * brush.hardness), 0, Math.PI * 2); ctx.stroke();
      ctx.setLineDash([]);
    }
  }
}
