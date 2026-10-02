// Pathtraced preview pane: settings, job control, frame polling, orbit camera.

import { $, el, clone, clamp, debounce, storage, fmt } from './util.js';
import { api } from './api.js';
import { makeFields, plainBinder } from './props.js';

const SETTINGS_KEY = 'tangent.render.v1';
const SETTINGS_VERSION = 2;

// Mirrors render/scene.py _lights: world space, x right, y up the texture, z up.
export function computeLights(doc, settings, meta) {
  const preset = meta.light_presets[settings.light.preset] || meta.light_presets.scifi_spot;
  const S = Math.max(doc.canvas_w, doc.canvas_h);
  const lights = preset.lights.map((spec0, i) => {
    const spec = { ...spec0 };
    if (i === 0) {
      for (const k of ['azimuth', 'elevation', 'cone', 'color']) if (k in settings.light) spec[k] = settings.light[k];
      spec.power *= settings.light.intensity ?? 1;
    }
    const az = spec.azimuth * Math.PI / 180, elv = spec.elevation * Math.PI / 180;
    const D = spec.distance * S;
    const aim = [spec.aim[0] * S, spec.aim[1] * S, 0];
    const pos = [aim[0] + D * Math.cos(elv) * Math.cos(az), aim[1] + D * Math.cos(elv) * Math.sin(az), D * Math.sin(elv)];
    const d = [aim[0] - pos[0], aim[1] - pos[1], aim[2] - pos[2]];
    const n = Math.hypot(...d);
    const half = spec.cone * Math.PI / 360;
    return {
      pos, dir: d.map((v) => v / n), color: spec.color.map((c) => c * spec.power * D * D),
      cosInner: Math.cos(half * (1 - spec.softness)), cosOuter: Math.cos(half),
    };
  });
  return { lights, env: preset.env };
}

export class RenderPane {
  constructor(store, meta, { onLightsChanged }) {
    this.store = store;
    this.meta = meta;
    this.onLightsChanged = onLightsChanged || (() => {});
    this.available = meta.pathtracer.available;
    const saved = storage.get(SETTINGS_KEY, null);
    // Settings saved before version 2 carry the old, too-strong bloom defaults.
    if (saved && (saved.version || 1) < SETTINGS_VERSION) delete saved.bloom;
    this.settings = clone(meta.render_defaults);
    if (saved) {
      for (const [k, v] of Object.entries(saved)) {
        if (k in this.settings && typeof v === typeof this.settings[k]) {
          this.settings[k] = (v && typeof v === 'object' && !Array.isArray(v)) ? Object.assign(this.settings[k], v) : v;
        }
      }
    }
    this.envs = meta.environments || [];
    if (!this.envs.some((e) => e.id === this.settings.environment.hdri)) {
      this.settings.environment.hdri = meta.render_defaults.environment.hdri;
    }
    this.img = $('#rImg');
    this.msg = $('#rMsg');
    this.stateEl = $('#rState');
    this.sppEl = $('#rSpp');
    this.version = -1;
    this.polling = false;
    this.job = 0;
    this.auto = $('#rAuto');
    this.auto.checked = storage.get('tangent.render.auto', true);
    this.auto.addEventListener('change', () => {
      storage.set('tangent.render.auto', this.auto.checked);
      if (this.auto.checked) this.start();
    });
    $('#rRestart').addEventListener('click', () => this.start());
    $('#rStop').addEventListener('click', () => this.stop());
    $('#rToggle').addEventListener('click', () => {
      const s = $('#rSettings');
      s.hidden = !s.hidden;
      $('#rToggle').textContent = s.hidden ? 'Settings ▾' : 'Settings ▴';
    });
    this.schedule = debounce(() => { if (this.auto.checked) this.start(); }, 300);
    this._orbitStart = this._throttle(() => this.start(), 140);
    this._setupOrbit();
    this.buildSettings();
    if (!this.available) {
      this._state('unavailable');
      this.msg.textContent = meta.pathtracer.error || 'Pathtracer unavailable';
    } else {
      this._waitReady();
    }
  }

  _throttle(fn, ms) {
    let last = 0, t = 0;
    return () => {
      const now = performance.now();
      clearTimeout(t);
      if (now - last >= ms) { last = now; fn(); } else t = setTimeout(() => { last = performance.now(); fn(); }, ms - (now - last));
    };
  }

  save() { storage.set(SETTINGS_KEY, { ...this.settings, version: SETTINGS_VERSION }); }

  _state(s) {
    this.stateEl.textContent = s;
    this.stateEl.className = 'pill ' + s;
  }

  async _waitReady() {
    const st = await api.renderStatus().catch(() => ({ state: 'error', error: 'server unreachable' }));
    if (st.state === 'compiling') {
      this._state('compiling');
      this.msg.textContent = 'Compiling the pathtracer… (only slow on the very first run)';
      setTimeout(() => this._waitReady(), 700);
      return;
    }
    if (st.state === 'error') {
      this._state('error');
      this.msg.textContent = st.error || 'Pathtracer error';
      return;
    }
    this.msg.textContent = '';
    if (this.auto.checked) this.start();
  }

  async start() {
    if (!this.available) return;
    try {
      const r = await api.renderStart(this.store.doc, this.settings);
      this.job = r.job;
      this._state('rendering');
      this.msg.textContent = '';
      this._poll();
    } catch (e) {
      this._state('error');
      this.msg.textContent = String(e.message || e);
    }
  }

  async stop() {
    if (!this.available) return;
    await api.renderStop().catch(() => {});
    this.polling = false;
    this._state('stopped');
  }

  async _poll() {
    if (this.polling) return;
    this.polling = true;
    while (this.polling) {
      try {
        const f = await api.renderFrame(this.version);
        if (f) {
          this.version = f.version;
          if (f.info.job === this.job) {
            const url = URL.createObjectURL(f.blob);
            const old = this.img.src;
            this.img.onload = () => { if (old.startsWith('blob:')) URL.revokeObjectURL(old); };
            this.img.src = url;
            this.sppEl.textContent = `${f.info.spp} / ${f.info.max_spp} spp · ${fmt(f.info.pass_ms, 0)} ms/pass`;
            if (f.info.done) { this._state('done'); this.polling = false; break; }
          }
        }
      } catch (e) {
        this._state('error');
        this.msg.textContent = 'Lost connection to the render server.';
        this.polling = false;
        break;
      }
      await new Promise((r) => setTimeout(r, 90));
    }
  }

  _setupOrbit() {
    const view = $('#renderView');
    let drag = null;
    view.addEventListener('pointerdown', (e) => {
      view.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, yaw: this.settings.camera.yaw, pitch: this.settings.camera.pitch };
    });
    view.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const c = this.settings.camera;
      c.yaw = drag.yaw - (e.clientX - drag.x) * 0.4;
      c.pitch = clamp(drag.pitch + (e.clientY - drag.y) * 0.3, 5, 89);
      this._orbitStart();
    });
    const end = () => { if (drag) { drag = null; this.save(); } };
    view.addEventListener('pointerup', end);
    view.addEventListener('pointercancel', end);
    view.addEventListener('wheel', (e) => {
      e.preventDefault();
      const c = this.settings.camera;
      c.distance = clamp(c.distance * Math.exp(e.deltaY * 0.001), 0.25, 5);
      this.save();
      this._orbitStart();
    }, { passive: false });
    view.addEventListener('dblclick', () => {
      this.settings.camera = clone(this.meta.render_defaults.camera);
      this.save();
      this.start();
    });
  }

  buildSettings() {
    const s = this.settings;
    const root = $('#rSettings');
    const changed = () => { this.save(); this.onLightsChanged(); this.schedule(); };
    const F = makeFields(plainBinder(changed, () => this.onLightsChanged()));
    const lp = this.meta.light_presets, mp = this.meta.material_presets;
    const applyLightPreset = (name) => {
      const key = lp[name].lights[0];
      Object.assign(s.light, { preset: name, azimuth: key.azimuth, elevation: key.elevation, cone: key.cone, color: [...key.color], intensity: 1 });
      this.buildSettings();
    };
    const applyMaterial = (name) => {
      const m = mp[name];
      Object.assign(s.material, { preset: name, albedo: [...m.albedo], metallic: m.metallic, roughness: m.roughness });
      this.buildSettings();
    };
    const es = s.environment;
    const thumbUrl = () => `/api/env/thumb?id=${encodeURIComponent(es.hdri)}&preset=${encodeURIComponent(s.light.preset)}`;
    const thumb = el('img', { class: 'env-thumb', src: thumbUrl(), alt: '', title: 'Environment preview (equirectangular)' });
    const upload = el('input', { type: 'file', accept: '.hdr', hidden: true });
    const uploadStatus = el('span', { class: 'muted' });
    upload.addEventListener('change', async () => {
      const f = upload.files[0];
      upload.value = '';
      if (!f) return;
      uploadStatus.textContent = 'Uploading…';
      try {
        const r = await api.envUpload(f);
        this.envs = r.environments;
        es.hdri = r.id;
        this.buildSettings();
        changed();
      } catch (e) {
        uploadStatus.textContent = `Upload failed: ${e.message}`;
      }
    });
    const envOptions = Object.fromEntries(this.envs.map((e) => [e.id, e.kind === 'file' ? `${e.label} (.hdr)` : e.label]));
    const resolutions = { '480x300': '480 × 300', '640x400': '640 × 400', '960x600': '960 × 600', '1280x800': '1280 × 800', '1600x1000': '1600 × 1000' };
    root.replaceChildren(
      el('h4', { text: 'Light' }),
      F.select('Preset', () => s.light.preset, applyLightPreset, Object.fromEntries(Object.entries(lp).map(([k, v]) => [k, v.label]))),
      F.range('Azimuth', () => s.light.azimuth, (v) => { s.light.azimuth = v; }, { min: 0, max: 360, step: 1, digits: 0, suffix: '°' }),
      F.range('Elevation', () => s.light.elevation, (v) => { s.light.elevation = v; }, { min: 3, max: 89, step: 1, digits: 0, suffix: '°' }),
      F.range('Cone', () => s.light.cone, (v) => { s.light.cone = v; }, { min: 5, max: 120, step: 1, digits: 0, suffix: '°' }),
      F.range('Intensity', () => s.light.intensity, (v) => { s.light.intensity = v; }, { min: 0, max: 4, step: 0.05 }),
      F.color('Color', () => s.light.color, (v) => { s.light.color = v; }),
      F.range('Scene light', () => s.scene_light, (v) => { s.scene_light = v; }, { min: 0, max: 1.5, step: 0.05 }),
      el('h4', { text: 'Environment' }),
      F.select('HDRI', () => es.hdri, (v) => { es.hdri = v; thumb.src = thumbUrl(); }, envOptions),
      el('div', { class: 'env-row' }, thumb,
        el('div', {},
          el('button', { class: 'small', text: 'Upload .hdr…', title: 'Adds a Radiance .hdr file to the hdri/ folder', onclick: () => upload.click() }),
          upload, el('div', {}, uploadStatus),
          el('p', { class: 'hint', text: 'Or drop .hdr files into the hdri/ folder and reload.' }))),
      F.range('Intensity', () => es.intensity, (v) => { es.intensity = v; }, { min: 0, max: 4, step: 0.05 }),
      F.range('Rotation', () => es.rotation, (v) => { es.rotation = v; }, { min: 0, max: 360, step: 1, digits: 0, suffix: '°' }),
      F.select('Background', () => es.background, (v) => { es.background = v; }, this.meta.backgrounds),
      el('h4', { text: 'Material' }),
      F.select('Preset', () => s.material.preset, applyMaterial, Object.fromEntries(Object.entries(mp).map(([k, v]) => [k, v.label]))),
      F.color('Albedo', () => s.material.albedo, (v) => { s.material.albedo = v; }),
      F.range('Metallic', () => s.material.metallic, (v) => { s.material.metallic = v; }),
      F.range('Roughness', () => s.material.roughness, (v) => { s.material.roughness = v; }, { min: 0.02, max: 1 }),
      F.range('Panel variation', () => s.material.variation, (v) => { s.material.variation = v; }, { min: 0, max: 0.6 }),
      el('h4', { text: 'Wire material' }),
      F.select('Preset', () => s.wire_material.preset, (name) => {
        const m = this.meta.wire_material_presets[name];
        Object.assign(s.wire_material, { preset: name, albedo: [...m.albedo], metallic: m.metallic, roughness: m.roughness });
        this.buildSettings();
      }, Object.fromEntries(Object.entries(this.meta.wire_material_presets).map(([k, v]) => [k, v.label]))),
      F.color('Color', () => s.wire_material.albedo, (v) => { s.wire_material.albedo = v; }),
      F.range('Metallic', () => s.wire_material.metallic, (v) => { s.wire_material.metallic = v; }),
      F.range('Roughness', () => s.wire_material.roughness, (v) => { s.wire_material.roughness = v; }, { min: 0.02, max: 1 }),
      el('h4', { text: 'Glow' }),
      F.check('Pathtrace bloom', () => s.bloom.pathtrace, (v) => { s.bloom.pathtrace = v; }, 'Bloom on the pathtraced image. Lighting stays physically based either way.'),
      F.check('Editor bloom', () => s.bloom.editor, (v) => { s.bloom.editor = v; }),
      F.range('Threshold', () => s.bloom.threshold, (v) => { s.bloom.threshold = v; }, { min: 0, max: 4, step: 0.05 }),
      F.range('Bloom', () => s.bloom.intensity, (v) => { s.bloom.intensity = v; }, { min: 0, max: 2, step: 0.05 }),
      F.range('Radius', () => s.bloom.radius, (v) => { s.bloom.radius = v; }, { min: 0.25, max: 3, step: 0.05 }),
      F.check('Editor spill', () => s.editor.spill, (v) => { s.editor.spill = v; }, 'Fake light spill from emitters in the realtime view'),
      F.range('Spill gain', () => s.editor.spill_gain, (v) => { s.editor.spill_gain = v; }, { min: 0, max: 4, step: 0.05 }),
      el('h4', { text: 'Surface' }),
      F.check('Displacement', () => s.displacement, (v) => { s.displacement = v; },
        'On: rays hit the real height field, so bevels self-shadow. Off: flat plane with normal-map shading only.'),
      F.range('Height scale', () => s.height_scale, (v) => { s.height_scale = v; }, { min: 0.25, max: 8, step: 0.25, suffix: '×' }),
      F.select('Tiles', () => String(s.tiles), (v) => { s.tiles = +v; }, { 1: 'Single', 3: '3 × 3 repeat' }),
      el('h4', { text: 'Output' }),
      F.select('Resolution', () => `${s.width}x${s.height}`, (v) => { const [w, h] = v.split('x').map(Number); s.width = w; s.height = h; }, resolutions),
      F.num('Max samples', () => s.max_spp, (v) => { s.max_spp = v; }, { min: 1, max: 65536, int: true }),
      F.num('Bounces', () => s.bounces, (v) => { s.bounces = v; }, { min: 0, max: 8, int: true }),
      F.range('Exposure', () => s.exposure, (v) => { s.exposure = v; }, { min: -4, max: 4, step: 0.1, suffix: ' EV' }),
      F.num('Firefly clamp', () => s.clamp, (v) => { s.clamp = v; }, { min: 0, max: 1000, step: 1 }),
    );
  }
}
