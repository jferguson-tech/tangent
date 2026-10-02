// Property panels: field builders, panel inspector, document settings, layout form.

import { el, fmt, UNIT, uid, clone, clamp, kelvinToRgb, normalizeDoc, defaultEmission } from './util.js';

// ---- field builders --------------------------------------------------------
// A binder decides what an edit means: an undoable document change, a render
// setting, or a UI preference.
//   begin(): before the first change of an edit   live(): during a slider drag
//   end():   after the edit is committed
export function docBinder(store) {
  return {
    begin: () => store.checkpoint(),
    live: () => store.emit('live', 'props'),
    end: () => store.emit('doc', 'props'),
  };
}
export const plainBinder = (onEnd = () => {}, onLive = () => {}) => ({ begin() {}, live: onLive, end: onEnd });

function row(label, ...ctl) {
  return el('label', { class: 'field' }, el('span', { text: label, title: label }), el('div', { class: 'ctl' }, ...ctl));
}

export function makeFields(binder, ctx = {}) {
  const density = () => (ctx.density ? ctx.density() : 512);
  const minPx = ctx.minPx || 2;

  function commit(fn) { binder.begin(); fn(); binder.end(); }

  return {
    // Length stored in meters, edited in `unit`. Optional pixel badge and alias warning.
    len(label, get, set, { unit = 'cm', min = 0, max = 1e6, step, px = true, warn = false, digits = 3, title } = {}) {
      const f = UNIT[unit];
      const input = el('input', { type: 'number', step: step ?? (unit === 'mm' ? 0.5 : 0.5), min: min * f, max: max * f, title });
      const badge = el('span', { class: 'px' });
      const refresh = () => {
        const v = get();
        if (document.activeElement !== input) input.value = fmt(v * f, digits);
        if (px) {
          const p = v * density();
          badge.textContent = `${fmt(p, p < 10 ? 1 : 0)} px`;
          const bad = warn && p > 0 && p < minPx;
          badge.classList.toggle('bad', bad);
          badge.title = bad ? `Below ${minPx} px at ${density()} px/m: this edge will alias. Widen it or raise the texel density.` : `${fmt(p, 2)} px at ${density()} px/m`;
          badge.textContent = (bad ? '⚠ ' : '') + badge.textContent;
        }
      };
      input.addEventListener('change', () => {
        const v = parseFloat(input.value);
        if (!isFinite(v)) { refresh(); return; }
        commit(() => set(clamp(v / f, min, max)));
        refresh();
      });
      refresh();
      const r = row(label, input, el('span', { class: 'unit', text: unit }), px ? badge : null);
      r.refresh = refresh;
      return r;
    },
    num(label, get, set, { min = -1e9, max = 1e9, step = 1, int = false } = {}) {
      const input = el('input', { type: 'number', min, max, step, value: get() });
      input.addEventListener('change', () => {
        let v = parseFloat(input.value);
        if (!isFinite(v)) { input.value = get(); return; }
        v = clamp(int ? Math.round(v) : v, min, max);
        commit(() => set(v));
        input.value = v;
      });
      return row(label, input);
    },
    range(label, get, set, { min = 0, max = 1, step = 0.01, digits = 2, suffix = '' } = {}) {
      const input = el('input', { type: 'range', min, max, step, value: get() });
      const val = el('span', { class: 'val', text: fmt(get(), digits) + suffix });
      let started = false;
      input.addEventListener('input', () => {
        if (!started) { binder.begin(); started = true; }
        set(parseFloat(input.value));
        val.textContent = fmt(parseFloat(input.value), digits) + suffix;
        binder.live();
      });
      input.addEventListener('change', () => { started = false; binder.end(); });
      return row(label, input, val);
    },
    select(label, get, set, options) {
      const opts = Array.isArray(options) ? options.map((o) => (Array.isArray(o) ? o : [o, o])) : Object.entries(options);
      const s = el('select', {}, opts.map(([v, t]) => el('option', { value: v, text: t })));
      s.value = String(get());
      s.addEventListener('change', () => commit(() => set(s.value)));
      return row(label, s);
    },
    check(label, get, set, title) {
      const c = el('input', { type: 'checkbox', checked: !!get() });
      c.addEventListener('change', () => commit(() => set(c.checked)));
      const r = row(label, c);
      if (title) r.title = title;
      return r;
    },
    color(label, get, set) {
      const toHex = (rgb) => '#' + rgb.map((v) => Math.round(clamp(v, 0, 1) * 255).toString(16).padStart(2, '0')).join('');
      const c = el('input', { type: 'color', value: toHex(get()) });
      c.addEventListener('input', () => {
        const h = c.value;
        set([1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255));
        binder.live();
      });
      c.addEventListener('change', () => binder.end());
      return row(label, c);
    },
    text(label, get, set) {
      const t = el('input', { type: 'text', value: get() });
      t.addEventListener('change', () => commit(() => set(t.value)));
      return row(label, t);
    },
  };
}

// ---- profile graphics ------------------------------------------------------
function interp(points, t) {
  const p = [...points].sort((a, b) => a[0] - b[0]);
  if (t <= p[0][0]) return p[0][1];
  for (let i = 1; i < p.length; i++) {
    if (t <= p[i][0]) {
      const [x0, y0] = p[i - 1], [x1, y1] = p[i];
      return x1 > x0 ? y0 + (y1 - y0) * (t - x0) / (x1 - x0) : y1;
    }
  }
  return p[p.length - 1][1];
}

function drawCurve(canvas, samples, points = null) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 280, h = canvas.clientHeight || 56;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  const pad = 8;
  const X = (t) => pad + t * (w - 2 * pad), Y = (v) => h - pad - v * (h - 2 * pad);
  g.strokeStyle = '#242b36';
  g.strokeRect(X(0), Y(1), X(1) - X(0), Y(0) - Y(1));
  // Cross-section: the outer edge is on the left, the flat top on the right.
  g.fillStyle = 'rgba(57,198,230,0.12)';
  g.strokeStyle = '#39c6e6';
  g.lineWidth = 1.5;
  g.beginPath();
  g.moveTo(X(0), Y(0));
  samples.forEach((v, i) => g.lineTo(X(i / (samples.length - 1)), Y(v)));
  g.lineTo(X(1), Y(0));
  g.closePath();
  g.fill();
  g.beginPath();
  samples.forEach((v, i) => (i ? g.lineTo : g.moveTo).call(g, X(i / (samples.length - 1)), Y(v)));
  g.stroke();
  if (points) {
    g.fillStyle = '#f0b03c';
    for (const [px, py] of points) { g.beginPath(); g.arc(X(px), Y(py), 4, 0, Math.PI * 2); g.fill(); }
  }
  g.fillStyle = '#7d8896';
  g.font = '10px ui-monospace, monospace';
  g.fillText('edge', X(0) + 2, Y(0) - 3);
  g.textAlign = 'right';
  g.fillText('top', X(1) - 2, Y(1) + 10);
  return { X, Y, pad, w, h };
}

function curveEditor(bevel, binder, redraw) {
  const c = el('canvas', { class: 'curve-editor', title: 'Drag points. Double-click to add a point, right-click to remove one.' });
  const samples = () => Array.from({ length: 64 }, (_, i) => interp(bevel.points, i / 63));
  let geo = null, dragI = -1, started = false;
  const paint = () => { geo = drawCurve(c, samples(), bevel.points); };
  const toT = (e) => {
    const r = c.getBoundingClientRect();
    return [clamp((e.clientX - r.left - geo.pad) / (geo.w - 2 * geo.pad), 0, 1),
            clamp(1 - (e.clientY - r.top - geo.pad) / (geo.h - 2 * geo.pad), 0, 1)];
  };
  const nearest = (e) => {
    const r = c.getBoundingClientRect();
    let best = -1, bd = 10;
    bevel.points.forEach(([px, py], i) => {
      const d = Math.hypot(geo.X(px) - (e.clientX - r.left), geo.Y(py) - (e.clientY - r.top));
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  };
  const sortPts = () => bevel.points.sort((a, b) => a[0] - b[0]);
  c.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    dragI = nearest(e);
    if (dragI >= 0) c.setPointerCapture(e.pointerId);
  });
  c.addEventListener('pointermove', (e) => {
    if (dragI < 0) return;
    if (!started) { binder.begin(); started = true; }
    const [t, v] = toT(e);
    const pts = bevel.points;
    const isEnd = dragI === 0 || dragI === pts.length - 1;
    pts[dragI][1] = v;
    if (!isEnd) pts[dragI][0] = clamp(t, pts[dragI - 1][0] + 0.01, pts[dragI + 1][0] - 0.01);
    paint();
    binder.live();
  });
  const up = () => {
    if (dragI >= 0 && started) { binder.end(); redraw(); }
    dragI = -1; started = false;
  };
  c.addEventListener('pointerup', up);
  c.addEventListener('pointercancel', up);
  c.addEventListener('dblclick', (e) => {
    if (bevel.points.length >= 16) return;
    const [t, v] = toT(e);
    binder.begin();
    bevel.points.push([t, v]);
    sortPts();
    binder.end();
    paint();
  });
  c.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    const i = nearest(e);
    if (i > 0 && i < bevel.points.length - 1) {
      binder.begin();
      bevel.points.splice(i, 1);
      binder.end();
      paint();
    }
  });
  requestAnimationFrame(paint);
  return c;
}

// ---- emission controls -------------------------------------------------------
const SWATCHES = [
  ['Cyan', [0.35, 0.85, 1.0]], ['Amber', [1.0, 0.62, 0.18]], ['Red', [1.0, 0.18, 0.12]],
  ['Green', [0.3, 1.0, 0.45]], ['Magenta', [1.0, 0.25, 0.85]],
];

// Glow toggle, color (picker, swatches, temperature) and strength for one Emission.
function emissionFields(F, em, meta, binder, rerender, { diffuser = false } = {}) {
  const box = el('div', { class: 'emission' });
  const set = (fn) => { binder.begin(); fn(); binder.end(); rerender(); };
  const rows = [F.check('Glow', () => em.enabled, (v) => { em.enabled = v; rerender(); },
    'Make this surface emit light. It lights the pathtraced scene and glows in the editor.')];
  if (em.enabled) {
    rows.push(
      F.color('Color', () => em.color, (v) => { em.color = v; }),
      el('div', { class: 'swatches' },
        SWATCHES.map(([name, c]) => el('button', {
          class: 'swatch', title: name, style: `background: rgb(${c.map((v) => Math.round(v * 255)).join(',')})`,
          onclick: () => set(() => { em.color = [...c]; }),
        })),
        [2700, 4000, 6500, 10000].map((k) => {
          const c = kelvinToRgb(k);
          return el('button', {
            class: 'swatch k', title: `${k} K`, text: `${k / 1000}k`,
            style: `background: rgb(${c.map((v) => Math.round(v * 255)).join(',')})`,
            onclick: () => set(() => { em.color = c; }),
          });
        })),
      F.range('Strength', () => em.strength, (v) => { em.strength = v; }, { min: 0, max: 20, step: 0.1, digits: 1 }),
      el('div', { class: 'btnrow' }, Object.entries(meta.emission_presets).map(([name, v]) => el('button', {
        class: 'small', text: name, title: `Strength ${v}`, onclick: () => set(() => { em.strength = v; }),
      }))),
      diffuser ? F.range('Diffuser', () => em.diffuser, (v) => { em.diffuser = v; }, { min: 0, max: 1, step: 0.05 }) : null,
    );
  }
  box.append(...rows.filter(Boolean));
  return box;
}

// ---- panel inspector ---------------------------------------------------------
const CORNER_LABELS = ['↖', '↗', '↘', '↙'];

export class PanelProps {
  constructor(root, store, meta, { onAction }) {
    this.root = root;
    this.store = store;
    this.meta = meta;
    this.onAction = onAction;
    this.binder = docBinder(store);
    this.warnings = {};
    this.linkCorners = true;
    this.openDetails = new Set();
    this.lenRows = [];
  }

  setWarnings(w) {
    this.warnings = w || {};
    this._renderWarnings();
  }

  _renderWarnings() {
    const box = this.root.querySelector('.warnbox');
    const p = this.store.selectedPanel;
    const list = p ? this.warnings[p.id] || [] : [];
    if (!box) return;
    box.hidden = !list.length;
    box.replaceChildren(el('ul', {}, list.map((w) => el('li', { text: w }))));
  }

  // Refresh pixel badges and values without rebuilding (keeps focus).
  refreshValues() { this.lenRows.forEach((r) => r.refresh && r.refresh()); }

  render() {
    const p = this.store.selectedPanel;
    this.lenRows = [];
    if (!p) {
      this.root.replaceChildren(el('div', { class: 'empty' },
        el('p', { text: 'Nothing selected.' }),
        el('p', { class: 'hint', text: 'Click a panel or wire to edit it. Press R to draw a panel, W to add a wire, or use Auto-layout.' })));
      return;
    }
    const F = makeFields(this.binder, { density: () => this.store.doc.texel_density, minPx: this.meta.min_bevel_px });
    const L = (...a) => { const r = F.len(...a); this.lenRows.push(r); return r; };
    const isTop = this.store.doc.panels.at(-1) === p;
    const isBottom = this.store.doc.panels[0] === p;

    const header = el('section', {},
      F.text('Name', () => p.name, (v) => { p.name = v; }),
      el('div', { class: 'btnrow' },
        el('button', { class: 'small', text: p.locked ? '🔒 Locked' : '🔓 Lock', onclick: () => this.onAction('lock') }),
        el('button', { class: 'small', text: p.visible ? '👁 Visible' : '— Hidden', onclick: () => this.onAction('visible') }),
        el('button', { class: 'small', text: '▲ Up', disabled: isTop, title: 'Move up in the stack (])', onclick: () => this.onAction('up') }),
        el('button', { class: 'small', text: '▼ Down', disabled: isBottom, title: 'Move down in the stack ([)', onclick: () => this.onAction('down') }),
        el('button', { class: 'small', text: '⧉ Duplicate', title: 'Ctrl+D', onclick: () => this.onAction('duplicate') }),
        el('button', { class: 'small danger', text: '✕ Delete', title: 'Delete', onclick: () => this.onAction('delete') })));

    const geo = el('section', {}, el('h4', { text: 'Rectangle' }),
      el('div', { class: 'grid2' },
        L('X', () => p.x, (v) => { p.x = v; }, { min: -100, px: false }),
        L('Y', () => p.y, (v) => { p.y = v; }, { min: -100, px: false }),
        L('W', () => p.w, (v) => { p.w = v; }, { min: 0.01, px: false }),
        L('H', () => p.h, (v) => { p.h = v; }, { min: 0.01, px: false })),
      el('p', { class: 'hint', text: 'Resizing changes only the rectangle. Bevel, corners and details keep their size in meters.' }));

    const corners = el('div', { class: 'corners' }, [0, 1, 3, 2].map((i) =>
      L(CORNER_LABELS[i], () => p.corners[i], (v) => {
        if (this.linkCorners) p.corners = p.corners.map(() => v); else p.corners[i] = v;
      }, { unit: 'mm', step: 1, px: false })));
    const link = el('label', { class: 'chk' }, el('input', { type: 'checkbox', checked: this.linkCorners,
      onchange: (e) => { this.linkCorners = e.target.checked; } }), 'link');
    const shape = el('section', {}, el('h4', { text: 'Shape' }),
      F.select('Mode', () => p.mode, (v) => { p.mode = v; }, { raise: 'Raise (add)', inset: 'Inset (subtract)', max: 'Max (union)' }),
      L('Depth', () => p.depth, (v) => { p.depth = v; }, { unit: 'mm', step: 0.5, px: false }),
      F.select('Corners', () => p.corner_style, (v) => { p.corner_style = v; this.render(); }, { chamfer: 'Chamfer', round: 'Round' }),
      el('div', { class: 'row', style: 'justify-content:space-between' }, el('span', { class: 'muted', text: 'Corner size' }), link),
      corners);

    const bevel = this._bevelSection('Bevel', p.bevel, F, L);

    const g = p.groove;
    const groove = el('section', {}, el('h4', { text: 'Inner groove' }),
      L('Width', () => g.width, (v) => { g.width = v; }, { unit: 'mm', warn: true, step: 0.5 }),
      L('Depth', () => g.depth, (v) => { g.depth = v; }, { unit: 'mm', step: 0.25, px: false }),
      L('Inset', () => g.offset, (v) => { g.offset = v; }, { unit: 'mm', step: 1 }),
      g.width > 0 ? emissionFields(F, g.emission, this.meta, this.binder, () => this.render()) : null,
      el('p', { class: 'hint', text: 'A channel running parallel to the outline. Width 0 turns it off.' }));

    const face = el('section', {}, el('h4', { text: 'Emission (light panel)' }),
      emissionFields(F, p.emission, this.meta, this.binder, () => this.render(), { diffuser: true }),
      el('p', { class: 'hint', text: 'The flat top inside the bevel glows. Details and later panels on top block it.' }));

    const details = el('section', {}, el('h4', { text: `Details (${p.details.length})` }),
      p.details.map((d, i) => this._detail(p, d, i, F, L)),
      el('div', { class: 'detail-add' }, Object.entries(this.meta.detail_presets).map(([k, preset]) =>
        el('button', { class: 'small', text: '+ ' + preset.kind, onclick: () => {
          this.store.mutate(() => p.details.push(Object.assign({ emission: defaultEmission() }, clone(preset))), 'props');
          this.openDetails.add(p.details.length - 1);
          this.render();
        } }))));

    this.root.replaceChildren(el('div', { class: 'warnbox', hidden: true }), header, geo, shape, bevel, groove, face, details);
    this._renderWarnings();
  }

  _bevelSection(title, bv, F, L) {
    const sec = el('section', {}, el('h4', { text: title }));
    const thumb = el('canvas', { class: 'profile-thumb' });
    const draw = () => {
      if (bv.profile === 'custom') return;
      requestAnimationFrame(() => drawCurve(thumb, this.meta.profiles[bv.profile] || []));
    };
    const rebuild = () => { const n = this._bevelSection(title, bv, F, L); sec.replaceWith(n); };
    sec.append(
      L('Width', () => bv.width, (v) => { bv.width = v; }, { unit: 'mm', warn: true, step: 0.5 }),
      F.select('Profile', () => bv.profile, (v) => { bv.profile = v; rebuild(); }, Object.keys(this.meta.profiles)),
      bv.profile === 'custom' ? curveEditor(bv, this.binder, () => {}) : thumb);
    draw();
    return sec;
  }

  _detail(p, d, i, F, L) {
    const open = this.openDetails.has(i);
    const det = el('details', { class: 'detail', open });
    det.addEventListener('toggle', () => (det.open ? this.openDetails.add(i) : this.openDetails.delete(i)));
    const anchors = Object.fromEntries(this.meta.anchors.map((a) => [a, {
      tl: 'Top-left', t: 'Top', tr: 'Top-right', l: 'Left', c: 'Center', r: 'Right', bl: 'Bottom-left',
      b: 'Bottom', br: 'Bottom-right', corners: 'All 4 corners', 'left-right': 'Left + right', 'top-bottom': 'Top + bottom' }[a] || a]));
    const body = el('div', { class: 'body' },
      F.text('Label', () => d.kind, (v) => { d.kind = v; }),
      F.select('Shape', () => d.shape, (v) => { d.shape = v; }, { circle: 'Circle', hex: 'Hexagon', rect: 'Rounded rect' }),
      F.select('Mode', () => d.mode, (v) => { d.mode = v; }, { raise: 'Raise', inset: 'Inset' }),
      F.select('Anchor', () => d.anchor, (v) => { d.anchor = v; }, anchors),
      el('div', { class: 'grid2' },
        L('↔', () => d.ox, (v) => { d.ox = v; }, { unit: 'mm', min: -10, px: false, step: 1, title: 'Inward offset from the anchor (horizontal)' }),
        L('↕', () => d.oy, (v) => { d.oy = v; }, { unit: 'mm', min: -10, px: false, step: 1, title: 'Inward offset from the anchor (vertical)' }),
        L('W', () => d.w, (v) => { d.w = v; }, { unit: 'mm', min: 0.5e-3, step: 1 }),
        L('H', () => d.h, (v) => { d.h = v; }, { unit: 'mm', min: 0.5e-3, step: 1 })),
      d.shape === 'rect' ? L('Radius', () => d.radius, (v) => { d.radius = v; }, { unit: 'mm', step: 0.5, px: false }) : null,
      F.num('Rotation°', () => d.rotation, (v) => { d.rotation = v; }, { min: -360, max: 360, step: 15 }),
      el('div', { class: 'grid2' },
        F.num('×', () => d.count, (v) => { d.count = v; }, { min: 1, max: 64, int: true }),
        L('gap', () => d.spacing, (v) => { d.spacing = v; }, { unit: 'mm', step: 1, px: false })),
      L('Depth', () => d.depth, (v) => { d.depth = v; }, { unit: 'mm', step: 0.25, px: false }),
      L('Bevel', () => d.bevel.width, (v) => { d.bevel.width = v; }, { unit: 'mm', warn: true, step: 0.25 }),
      F.select('Profile', () => d.bevel.profile, (v) => { d.bevel.profile = v; }, Object.keys(this.meta.profiles).filter((k) => k !== 'custom')),
      emissionFields(F, d.emission, this.meta, this.binder, () => this.render()),
      el('div', { class: 'btnrow' },
        el('button', { class: 'small', text: 'Duplicate', onclick: () => {
          this.store.mutate(() => p.details.splice(i + 1, 0, clone(d)), 'props'); this.render();
        } }),
        el('button', { class: 'small danger', text: 'Remove', onclick: () => {
          this.store.mutate(() => p.details.splice(i, 1), 'props'); this.openDetails.clear(); this.render();
        } })));
    const dot = d.emission.enabled
      ? el('span', { class: 'glowdot', style: `background: rgb(${d.emission.color.map((v) => Math.round(v * 255)).join(',')})` }) : null;
    det.append(el('summary', {}, dot, el('span', { text: d.kind || 'detail' }),
      el('span', { class: 'muted', text: ` ${d.shape} · ${d.anchor}${d.count > 1 ? ' · ×' + d.count : ''}` })), body);
    return det;
  }
}

// ---- document settings ----------------------------------------------------------
export class DocProps {
  constructor(root, store, meta, { onPrefs }) {
    this.root = root;
    this.store = store;
    this.meta = meta;
    this.onPrefs = onPrefs;
  }

  render() {
    const doc = this.store.doc;
    const F = makeFields(docBinder(this.store), { density: () => doc.texel_density, minPx: this.meta.min_bevel_px });
    const res = () => [Math.round(doc.canvas_w * doc.texel_density), Math.round(doc.canvas_h * doc.texel_density)];
    const [rx, ry] = res();
    const presets = this.meta.texel_presets;
    const custom = this._customDensity || !presets.includes(doc.texel_density);
    const densitySel = F.select('Texel density',
      () => (custom ? 'custom' : String(doc.texel_density)),
      (v) => {
        this._customDensity = v === 'custom';
        if (v !== 'custom') doc.texel_density = parseFloat(v);
      },
      [...presets.map((p) => [String(p), `${p} px/m${p === 512 ? ' (default)' : ''}`]), ['custom', 'Custom…']]);
    const pot = (n) => (n & (n - 1)) === 0;
    const resPresets = [512, 1024, 2048, 4096];

    const prefsBinder = plainBinder(() => { this.store.savePrefs(); this.onPrefs(); });
    const P = makeFields(prefsBinder, { density: () => doc.texel_density });

    this.root.replaceChildren(
      el('section', {}, el('h4', { text: 'Texel density' }),
        densitySel,
        custom ? F.num('px / m', () => doc.texel_density, (v) => { doc.texel_density = v; }, { min: 16, max: 8192, step: 1 }) : null,
        el('p', { class: 'hint', text: 'Bevels, depths and details are stored in meters. The density converts them to pixels, like a game texel density spec.' })),
      el('section', {}, el('h4', { text: 'Canvas' }),
        F.len('Width', () => doc.canvas_w, (v) => { doc.canvas_w = v; }, { unit: 'm', min: 0.05, step: 0.05, px: true }),
        F.len('Height', () => doc.canvas_h, (v) => { doc.canvas_h = v; }, { unit: 'm', min: 0.05, step: 0.05, px: true }),
        el('p', { class: (pot(rx) && pot(ry)) ? 'muted' : 'warn', text: `Output: ${rx} × ${ry} px${(pot(rx) && pot(ry)) ? '' : ' (not a power of two)'}` }),
        el('div', { class: 'btnrow' }, el('span', { class: 'muted', text: 'Fit to' }), resPresets.map((r) => el('button', {
          class: 'small', text: `${r}²`, title: `Set the canvas to ${r / doc.texel_density} m square so the output is ${r} px`,
          onclick: () => this.store.mutate(() => { doc.canvas_w = doc.canvas_h = r / doc.texel_density; }, 'props'),
        }))),
        el('p', { class: 'hint', text: 'Changing the canvas never rescales panels. Their sizes stay in meters.' })),
      el('section', {}, el('h4', { text: 'Tiling' }),
        F.check('Seamless', () => doc.tiling, (v) => { doc.tiling = v; },
          'Wrap distance fields across the canvas edges'),
        el('p', { class: 'hint', text: 'Off by default. A layout that stays inside the canvas with a border or seam already repeats cleanly. Turn this on when panels cross the border; they wrap to the opposite side.' })),
      el('section', {}, el('h4', { text: 'Normal map' }),
        F.range('Strength', () => doc.normal_strength, (v) => { doc.normal_strength = v; }, { min: 0, max: 4, step: 0.05 }),
        F.select('Convention', () => doc.normal_convention, (v) => { doc.normal_convention = v; },
          { gl: 'OpenGL (Y+): Blender, Unity, Godot', dx: 'DirectX (Y−): Unreal, 3ds Max' }),
        el('p', { class: 'hint', text: 'The editor always previews OpenGL. The convention applies on export.' })),
      el('section', {}, el('h4', { text: 'Editor' }),
        P.len('Snap grid', () => this.store.prefs.snap, (v) => { this.store.prefs.snap = v; }, { unit: 'cm', step: 0.5, px: true }),
        el('p', { class: 'hint', text: 'Hold Alt while dragging to ignore the snap grid.' })),
    );
  }
}

// ---- auto-layout form ---------------------------------------------------------
export function renderLayoutForm(root, store, meta) {
  const doc = store.doc;
  if (!doc.layout || !Object.keys(doc.layout).length) doc.layout = clone(meta.layout_defaults);
  for (const [k, v] of Object.entries(meta.layout_defaults)) if (!(k in doc.layout)) doc.layout[k] = v;
  const P = doc.layout;
  const F = makeFields(plainBinder(() => store.emit('layoutParams')), { density: () => doc.texel_density, minPx: meta.min_bevel_px });
  const set = (k) => (v) => { P[k] = v; };
  const get = (k) => () => P[k];
  root.replaceChildren(...[
    F.num('Seed', get('seed'), set('seed'), { min: 0, max: 1e9, int: true }),
    F.len('Min size', get('min_size'), set('min_size'), { unit: 'cm', min: 0.02, step: 1, px: false }),
    F.len('Max size', get('max_size'), set('max_size'), { unit: 'cm', min: 0.04, step: 5, px: false }),
    F.num('Max depth', get('max_depth'), set('max_depth'), { min: 0, max: 12, int: true }),
    F.range('Stop chance', get('stop_chance'), set('stop_chance')),
    F.len('Seam', get('seam'), set('seam'), { unit: 'mm', step: 0.5, warn: true }),
    F.len('Border', get('border'), set('border'), { unit: 'mm', step: 1 }),
    F.len('Cut grid', get('grid'), set('grid'), { unit: 'cm', step: 0.5, px: false }),
    F.select('Symmetry', get('symmetry'), set('symmetry'), { none: 'None', x: 'Mirror X', y: 'Mirror Y', xy: 'Mirror X + Y' }),
    doc.tiling ? F.check('Wrap offset', get('wrap_offset'), set('wrap_offset'), 'Shift the layout so panels cross the edges (seamless mode)') : null,
    F.len('Depth', get('depth'), set('depth'), { unit: 'mm', step: 0.5, px: false }),
    F.range('Depth var.', get('depth_variation'), set('depth_variation')),
    F.len('Bevel', get('bevel_width'), set('bevel_width'), { unit: 'mm', step: 0.5, warn: true }),
    F.select('Profile', get('profile'), set('profile'), Object.keys(meta.profiles).filter((k) => k !== 'custom')),
    F.select('Corners', get('corner_style'), set('corner_style'), { chamfer: 'Chamfer', round: 'Round' }),
    F.len('Corner', get('corner'), set('corner'), { unit: 'mm', step: 1, px: false }),
    F.range('Inset chance', get('inset_chance'), set('inset_chance')),
    F.range('Nest chance', get('nest_chance'), set('nest_chance')),
    F.range('Groove chance', get('groove_chance'), set('groove_chance')),
    F.range('Details', get('detail_density'), set('detail_density')),
    F.range('Lights', get('emissive_density'), set('emissive_density')),
    F.range('Wires', get('wire_density'), set('wire_density'), { min: 0, max: 1, step: 0.01 }),
    F.range('Simulated', get('wire_sim_share'), set('wire_sim_share')),
    F.range('Harnesses', get('harness_chance'), set('harness_chance')),
    F.select('Light colors', get('light_palette'), set('light_palette'),
      Object.fromEntries(meta.light_palettes.map((k) => [k, k[0].toUpperCase() + k.slice(1)]))),
  ].filter(Boolean));
}

// Style for a newly drawn panel: copy the selected panel, else the layout defaults.
export function newPanelStyle(store, meta) {
  const sel = store.selectedPanel;
  if (sel) {
    const s = clone(sel);
    delete s.id; s.details = []; s.locked = false; s.visible = true;
    return s;
  }
  const L = Object.assign({}, meta.layout_defaults, store.doc.layout || {});
  return {
    id: uid(), name: 'Panel', x: 0, y: 0, w: 0.3, h: 0.3,
    corner_style: L.corner_style, corners: [L.corner, L.corner, L.corner, L.corner],
    mode: 'raise', depth: L.depth,
    bevel: { width: L.bevel_width, profile: L.profile, points: [[0, 0], [0.5, 0.7], [1, 1]] },
    groove: { offset: 0.02, width: 0, depth: 0.002, emission: defaultEmission() },
    emission: defaultEmission(),
    details: [], locked: false, visible: true,
  };
}

// ---- wire inspector ------------------------------------------------------------
export class WireProps {
  constructor(root, store, meta, { onAction, onResetShape }) {
    this.root = root;
    this.store = store;
    this.meta = meta;
    this.onAction = onAction;
    this.onResetShape = onResetShape;
    this.binder = docBinder(store);
    this.lenRows = [];
  }

  refreshValues() { this.lenRows.forEach((r) => r.refresh && r.refresh()); }

  _endText(end) {
    if (!end.panel) return 'free point';
    const p = this.store.panel(end.panel);
    return p ? `${p.name} · ${end.anchor}` : 'free point (panel removed)';
  }

  render() {
    const w = this.store.selectedWire;
    this.lenRows = [];
    if (!w) return false;
    const F = makeFields(this.binder, { density: () => this.store.doc.texel_density, minPx: this.meta.min_bevel_px });
    const L = (...a) => { const r = F.len(...a); this.lenRows.push(r); return r; };
    const rerender = () => this.render();
    const sim = w.mode === 'sim';
    const status = sim ? (w.settled ? 'settled' : 'not simulated yet') : 'routed';
    const detach = (which) => el('button', {
      class: 'small', text: 'Detach', disabled: !w[which].panel, title: 'Keep this end where it is, but stop following the panel',
      onclick: () => this.store.mutate(() => { w[which].panel = null; }, 'props'),
    });
    this.root.replaceChildren(
      el('section', {},
        F.text('Name', () => w.name, (v) => { w.name = v; }),
        el('div', { class: 'btnrow' },
          el('button', { class: 'small', text: w.locked ? '🔒 Locked' : '🔓 Lock', title: 'Locked wires are kept by auto-layout and not simulated', onclick: () => this.onAction('lock') }),
          el('button', { class: 'small', text: w.visible ? '👁 Visible' : '— Hidden', onclick: () => this.onAction('visible') }),
          el('button', { class: 'small', text: '⧉ Duplicate', onclick: () => this.onAction('duplicate') }),
          el('button', { class: 'small danger', text: '✕ Delete', onclick: () => this.onAction('delete') }))),
      el('section', {}, el('h4', {}, 'Path', el('span', { class: 'spacer' }), el('span', { class: 'muted', text: status })),
        F.select('Mode', () => w.mode, (v) => { w.mode = v; if (v === 'sim') { w.points = []; w.settled = false; this.onResetShape(w); } rerender(); },
          { sim: 'Simulated (gravity)', route: 'Routed (Manhattan)' }),
        sim ? F.range('Slack', () => w.slack, (v) => { w.slack = v; w.settled = false; }, { min: 1, max: 2, step: 0.01, digits: 2, suffix: '×' }) : null,
        sim ? el('div', { class: 'btnrow' }, el('button', { class: 'small', text: 'Reset shape', title: 'Back to the starting sag; simulate again to settle it', onclick: () => this.onResetShape(w) })) : null,
        !sim ? L('Bend radius', () => w.corner_radius, (v) => { w.corner_radius = v; }, { unit: 'mm', step: 1, px: false }) : null,
        !sim ? F.check('45° runs', () => w.allow45, (v) => { w.allow45 = v; }, 'Allow diagonal runs as well as right angles') : null,
        !sim ? L('Clearance', () => w.clearance, (v) => { w.clearance = v; }, { unit: 'mm', step: 1, px: false, title: 'Gap kept to wires routed before this one' }) : null,
        el('p', { class: 'hint', text: sim
          ? 'Hangs under gravity. Use the simulation controls on the left to settle it; drag it while running to pull it around.'
          : 'Finds a right-angle path that avoids raised details, follows seams and keeps clear of earlier routed wires.' })),
      el('section', {}, el('h4', { text: 'Ends' }),
        el('div', { class: 'field' }, el('span', { text: 'Start' }), el('div', { class: 'ctl' }, el('span', { text: this._endText(w.a) }), detach('a'))),
        el('div', { class: 'field' }, el('span', { text: 'End' }), el('div', { class: 'ctl' }, el('span', { text: this._endText(w.b) }), detach('b'))),
        el('p', { class: 'hint', text: 'Drag an end handle in the view to move it. Dropping it on a panel attaches it, so it follows that panel.' })),
      el('section', {}, el('h4', { text: 'Cable' }),
        F.select('Profile', () => w.profile, (v) => { w.profile = v; }, { round: 'Round tube', ribbon: 'Ribbon cable', hose: 'Ribbed hose' }),
        L('Radius', () => w.radius, (v) => { w.radius = v; }, { unit: 'mm', min: 0.0005, step: 0.5, warn: true }),
        F.num('Strands', () => w.bundle, (v) => { w.bundle = v; }, { min: 1, max: 16, int: true }),
        w.bundle > 1 ? L('Spacing', () => w.bundle_spacing, (v) => { w.bundle_spacing = v; }, { unit: 'mm', step: 0.5, px: false, title: '0 = strands touching' }) : null,
        F.check('Connectors', () => w.connectors, (v) => { w.connectors = v; }, 'Plugs at both ends'),
        L('Clip spacing', () => w.clip_spacing, (v) => { w.clip_spacing = v; }, { unit: 'cm', step: 1, px: false, title: '0 = no clips' })),
      el('section', {}, el('h4', { text: 'Emission' }), emissionFields(F, w.emission, this.meta, this.binder, rerender)),
    );
    return true;
  }
}
