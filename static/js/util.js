// Small DOM, timing and unit helpers.

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

export function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === 'class') e.className = v;
    else if (k === 'style') e.style.cssText = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'text') e.textContent = v;
    else if (k in e && typeof v !== 'string') e[k] = v;
    else e.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    e.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return e;
}

export function debounce(fn, ms) {
  let t = 0;
  const d = (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
  d.cancel = () => clearTimeout(t);
  return d;
}

export const clone = (o) => JSON.parse(JSON.stringify(o));
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

// ---- units: documents store meters ------------------------------------
export const UNIT = { m: 1, cm: 100, mm: 1000 };

export function fmt(v, digits = 2) {
  if (!isFinite(v)) return '—';
  const s = v.toFixed(digits);
  return s.includes('.') ? s.replace(/\.?0+$/, '') : s;
}

export function fmtLen(m, unit = 'cm', digits = 2) {
  return `${fmt(m * UNIT[unit], digits)} ${unit}`;
}

export function pxOf(meters, density) { return meters * density; }

export function snap(v, step) { return step > 0 ? Math.round(v / step) * step : v; }

export function uid(prefix = 'p') {
  return prefix + Math.random().toString(16).slice(2, 10);
}

export const storage = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
    catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* storage blocked */ }
  },
};

export function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = el('a', { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ---- emission --------------------------------------------------------------
export const defaultEmission = () => ({ enabled: false, color: [0.35, 0.85, 1.0], strength: 4.0, diffuser: 0.0 });

// Fill in fields added after a document was saved, so older files keep working.
export function normalizeDoc(doc) {
  for (const p of doc.panels || []) {
    p.emission = Object.assign(defaultEmission(), p.emission || {});
    p.groove = p.groove || { offset: 0.02, width: 0, depth: 0.002 };
    p.groove.emission = Object.assign(defaultEmission(), p.groove.emission || {});
    for (const d of p.details || []) d.emission = Object.assign(defaultEmission(), d.emission || {});
  }
  const m = doc.materials || {};
  doc.materials = {
    metal: Object.assign({ color: [0.42, 0.44, 0.47], roughness: 0.45 }, m.metal || {}),
    paint: Object.assign({ enabled: false, color: [0.30, 0.32, 0.34], roughness: 0.5, metallic: 0 }, m.paint || {}),
    wire: Object.assign({ color: [0.035, 0.035, 0.04], roughness: 0.5, metallic: 0 }, m.wire || {}),
    variation: m.variation ?? 0.15,
  };
  doc.weathering = Object.assign({ enabled: false, preset: 'clean', age: 0, seed: 1, edge_wear: 0.5, dirt: 0.5, streaks: 0.5,
    rust: 0.5, heat: 0.5, wire_wear: 0.5, auto_leaks: 0 }, doc.weathering || {});
  doc.leaks = doc.leaks || [];
  doc.strokes = doc.strokes || [];
  for (const st of doc.strokes) {
    if (!st.id) st.id = 's' + Math.random().toString(16).slice(2, 10);
    if (st.visible === undefined) st.visible = true;
  }
  doc.wires = doc.wires || [];
  doc.wire_sim = Object.assign({ gravity_angle: 90, gravity: 1, collide: true, auto_resettle: false }, doc.wire_sim || {});
  for (const w of doc.wires) {
    Object.assign(w, Object.assign({
      mode: 'sim', profile: 'hose', radius: 0.014, slack: 1.15, corner_radius: 0.025, allow45: false,
      clearance: 0.01, connectors: true, clip_spacing: 0, bundle: 1, bundle_spacing: 0, points: [],
      settled: false, locked: false, visible: true,
    }, w));
    w.emission = Object.assign(defaultEmission(), w.emission || {});
  }
  return doc;
}

// Approximate blackbody color (sRGB, 0..1) for a temperature in kelvin.
export function kelvinToRgb(k) {
  const t = clamp(k, 1000, 40000) / 100;
  let r, g, b;
  if (t <= 66) {
    r = 255;
    g = 99.4708025861 * Math.log(t) - 161.1195681661;
    b = t <= 19 ? 0 : 138.5177312231 * Math.log(t - 10) - 305.0447927307;
  } else {
    r = 329.698727446 * Math.pow(t - 60, -0.1332047592);
    g = 288.1221695283 * Math.pow(t - 60, -0.0755148492);
    b = 255;
  }
  return [r, g, b].map((v) => clamp(v, 0, 255) / 255);
}
