// Surface tab: materials (bare metal, paint, wires), weathering settings,
// liquid leaks and the weathering brush. Materials, weathering, leaks and
// strokes are part of the document: saved, undoable and exported. The brush
// settings are a per-browser preference.

import { el } from './util.js';
import { makeFields, docBinder, plainBinder } from './props.js';
import { brush, saveBrush, LIQUID_CSS, CHANNEL_CSS } from './weathertools.js';

const PRESET_KEYS = ['age', 'edge_wear', 'dirt', 'streaks', 'rust', 'heat', 'wire_wear', 'auto_leaks'];

export class SurfaceProps {
  constructor(root, store, meta, { tools = () => null, setTool = () => {}, currentTool = () => 'select' } = {}) {
    this.root = root;
    this.store = store;
    this.meta = meta;
    this.binder = docBinder(store);
    this.tools = tools;
    this.setTool = setTool;
    this.currentTool = currentTool;
  }

  render() {
    const doc = this.store.doc;
    const m = doc.materials;
    const w = doc.weathering;
    const F = makeFields(this.binder);
    const rerender = () => this.render();
    const presets = this.meta.weathering_presets;
    // Changing a strength by hand turns the preset into "custom".
    const custom = (fn) => (v) => { fn(v); w.preset = 'custom'; };
    const applyPreset = (name) => {
      w.preset = name;
      if (presets[name]) {
        for (const k of PRESET_KEYS) if (k in presets[name]) w[k] = presets[name][k];
        if (name !== 'clean') w.enabled = true;
      }
      rerender();
    };
    const matPresets = Object.entries(this.meta.material_presets);
    const metalPresets = Object.fromEntries([['', 'Choose…'], ...matPresets.filter(([, p]) => p.metallic >= 0.5).map(([k, p]) => [k, p.label])]);
    const paintPresets = Object.fromEntries([['', 'Choose…'], ...matPresets.filter(([, p]) => p.metallic < 0.5).map(([k, p]) => [k, p.label])]);
    const wirePresets = Object.fromEntries([['', 'Choose…'], ...Object.entries(this.meta.wire_material_presets).map(([k, p]) => [k, p.label])]);
    const fromPreset = (target, table, withMetallic) => (name) => {
      const p = table[name];
      if (!p) return;
      target.color = [...p.albedo];
      target.roughness = p.roughness;
      if (withMetallic) target.metallic = p.metallic;
      if (target === m.paint) m.paint.enabled = true;
      rerender();
    };
    const presetLabels = Object.fromEntries([...Object.entries(presets).map(([k, p]) => [k, p.label]), ['custom', 'Custom']]);
    const slider = (label, key, title) => {
      const r = F.range(label, () => w[key], custom((v) => { w[key] = v; }), { min: 0, max: 1, step: 0.01 });
      if (title) r.title = title;
      return r;
    };

    this.root.replaceChildren(
      el('section', {}, el('h4', { text: 'Weathering' }),
        F.check('Enabled', () => w.enabled, (v) => { w.enabled = v; rerender(); },
          'Simulate wear, dirt, water, rust, heat, leaks and wire wear, and apply brush strokes'),
        F.select('Preset', () => w.preset, applyPreset, presetLabels),
        slider('Age', 'age', 'How long the surface has been in use: drives every simulated effect except leaks and brush strokes'),
        slider('Edge wear', 'edge_wear'),
        slider('Dirt', 'dirt'),
        slider('Streaks', 'streaks', 'Water running down in the gravity direction'),
        slider('Rust', 'rust'),
        slider('Heat & soot', 'heat', 'Temper colors and scorching around vents and bright lights, and soot rising from vents'),
        slider('Wire wear', 'wire_wear', 'Wires rub paint off the edges they rest on, fade and crack, and drip water from their low points'),
        F.num('Auto leaks', () => w.auto_leaks, custom((v) => { w.auto_leaks = v; }),
          { min: 0, max: this.meta.max_auto_leaks || 16, int: true }),
        el('div', { class: 'grid2 seedrow' },
          F.num('Seed', () => w.seed, (v) => { w.seed = v; }, { min: 0, max: 2147483647, int: true }),
          el('button', { class: 'small', text: '🎲 New seed', onclick: () => {
            this.store.mutate((d) => { d.weathering.seed = Math.floor(Math.random() * 1e6); }, 'props');
            rerender();
          } })),
        el('p', { class: 'hint', text: 'Water, drips, soot and leaks follow the gravity direction set in Wire simulation. '
          + 'Auto leaks start at random bolts, vents and wire connectors. '
          + 'The simulations take a few seconds after a change on a large canvas; while you drag a panel the view shows materials without weathering.' })),
      this._leaks(),
      this._brush(),
      this._strokes(),
      el('section', {}, el('h4', { text: 'Bare metal' }),
        F.select('Preset', () => '', fromPreset(m.metal, this.meta.material_presets, false), metalPresets),
        F.color('Color', () => m.metal.color, (v) => { m.metal.color = v; }),
        F.range('Roughness', () => m.metal.roughness, (v) => { m.metal.roughness = v; }, { min: 0.02, max: 1, step: 0.01 }),
        el('p', { class: 'hint', text: 'The panels themselves. Without paint, edge wear polishes the metal; with paint, it shows through chips.' })),
      el('section', {}, el('h4', { text: 'Paint' }),
        F.check('Painted', () => m.paint.enabled, (v) => { m.paint.enabled = v; rerender(); }, 'A paint layer over the metal'),
        m.paint.enabled ? F.select('Preset', () => '', fromPreset(m.paint, this.meta.material_presets, true), paintPresets) : null,
        m.paint.enabled ? F.color('Color', () => m.paint.color, (v) => { m.paint.color = v; }) : null,
        m.paint.enabled ? F.range('Roughness', () => m.paint.roughness, (v) => { m.paint.roughness = v; }, { min: 0.02, max: 1, step: 0.01 }) : null,
        m.paint.enabled ? F.range('Metallic', () => m.paint.metallic, (v) => { m.paint.metallic = v; }, { min: 0, max: 1, step: 0.01 }) : null),
      el('section', {}, el('h4', { text: 'Wires' }),
        F.select('Preset', () => '', fromPreset(m.wire, this.meta.wire_material_presets, true), wirePresets),
        F.color('Color', () => m.wire.color, (v) => { m.wire.color = v; }),
        F.range('Roughness', () => m.wire.roughness, (v) => { m.wire.roughness = v; }, { min: 0.02, max: 1, step: 0.01 }),
        F.range('Metallic', () => m.wire.metallic, (v) => { m.wire.metallic = v; }, { min: 0, max: 1, step: 0.01 })),
      el('section', {}, el('h4', { text: 'Variation' }),
        F.range('Per panel', () => m.variation, (v) => { m.variation = v; }, { min: 0, max: 0.6, step: 0.01 }),
        el('p', { class: 'hint', text: 'Small color and roughness differences between panels.' })),
    );
  }

  _toolButton(tool, label) {
    const on = this.currentTool() === tool;
    return el('button', { class: 'small' + (on ? ' on' : ''), text: on ? `${label}: on` : label,
                          title: on ? 'Back to the Select tool' : undefined,
                          onclick: () => this.setTool(on ? 'select' : tool) });
  }

  _leaks() {
    const doc = this.store.doc;
    const leaks = doc.leaks || [];
    const tools = this.tools();
    const selected = tools ? tools.selected : null;
    const liquids = Object.fromEntries(Object.entries(this.meta.liquids).map(([k, v]) => [k, v.label]));
    const rows = leaks.map((k, i) => {
      const F = makeFields(this.binder);
      const head = el('div', { class: 'leakhead', title: 'Select this leak (drag it in the view with the Leak tool)',
                               onclick: () => { if (tools) tools.selectLeak(k.id); } },
        el('span', { class: 'drop', style: `background:${LIQUID_CSS[k.liquid] || '#fff'}` }),
        el('span', { text: `Leak ${i + 1}` }),
        el('span', { class: 'muted', text: k.panel ? 'on a panel' : 'free' }),
        el('button', { class: 'small icon', text: '✕', title: 'Delete this leak', onclick: (e) => {
          e.stopPropagation();
          this.store.mutate((d) => { d.leaks = d.leaks.filter((q) => q.id !== k.id); }, 'props');
          this.render();
        } }));
      return el('div', { class: 'leakrow' + (k.id === selected ? ' on' : '') }, head,
        F.select('Liquid', () => k.liquid, (v) => { k.liquid = v; this.render(); }, liquids),
        F.range('Amount', () => k.amount, (v) => { k.amount = v; }, { min: 0, max: 1, step: 0.01 }),
        k.liquid === 'coolant' ? F.color('Tint', () => k.color, (v) => { k.color = v; }) : null);
    });
    return el('section', {}, el('h4', { text: `Leaks (${leaks.length})` }),
      el('div', { class: 'btnrow' },
        this._toolButton('leak', 'Place leaks (K)'),
        leaks.length ? el('button', { class: 'small', text: 'Clear all', onclick: () => {
          this.store.mutate((d) => { d.leaks = []; }, 'props');
          this.render();
        } }) : null),
      ...rows,
      el('p', { class: 'hint', text: 'Liquid pours from each leak, runs down, pools on ledges and follows seams. '
        + 'Water leaves grime and mineral rings, oil a dark glossy stain, coolant a tinted crust. '
        + 'A leak placed on a panel moves with it.' }));
  }

  _brush() {
    const B = makeFields(plainBinder(() => { saveBrush(); this.render(); }));
    const chans = Object.assign({}, this.meta.brush_channels, brush.mode === 'erase' ? { all: 'Everything' } : {});
    if (!(brush.channel in chans)) brush.channel = 'rust';
    return el('section', {}, el('h4', { text: 'Brush' }),
      el('div', { class: 'btnrow' }, this._toolButton('brush', 'Paint (B)')),
      B.select('Mode', () => brush.mode, (v) => { brush.mode = v; if (v === 'add' && brush.channel === 'all') brush.channel = 'rust'; },
        { add: 'Add', erase: 'Erase' }),
      B.select('Effect', () => brush.channel, (v) => { brush.channel = v; }, chans),
      B.len('Radius', () => brush.radius, (v) => { brush.radius = v; }, { unit: 'cm', min: 0.002, max: 0.5, step: 0.5, px: false, digits: 1 }),
      B.range('Strength', () => brush.strength, (v) => { brush.strength = v; }, { min: 0.05, max: 1, step: 0.01 }),
      B.range('Hardness', () => brush.hardness, (v) => { brush.hardness = v; }, { min: 0, max: 0.95, step: 0.01 }),
      el('p', { class: 'hint', text: 'Settings for new strokes. Paint an effect on, or erase it (erasing also removes what the simulation made there). '
        + 'A stroke painted on a panel moves with it.' }));
  }

  _strokes() {
    const strokes = this.store.doc.strokes || [];
    const tools = this.tools();
    const selected = tools ? tools.selectedStroke : null;
    const labels = Object.assign({ all: 'Everything' }, this.meta.brush_channels);
    const select = (id) => { if (tools) tools.selectStroke(id); };
    const mutate = (fn) => { this.store.mutate(fn, 'props'); this.render(); };
    const move = (id, by) => mutate((d) => {
      const i = d.strokes.findIndex((q) => q.id === id), j = i + by;
      if (i < 0 || j < 0 || j >= d.strokes.length) return;
      [d.strokes[i], d.strokes[j]] = [d.strokes[j], d.strokes[i]];
    });
    // Newest (painted last, on top) first.
    const rows = strokes.map((st, i) => [st, i]).reverse().map(([st, i]) => {
      const on = st.id === selected;
      const hidden = st.visible === false;
      const head = el('div', { class: 'leakhead', title: 'Select this stroke (it is outlined in the view)',
                               onclick: () => select(on ? null : st.id) },
        el('button', { class: 'small icon', text: hidden ? '◌' : '●', title: hidden ? 'Show' : 'Hide', onclick: (e) => {
          e.stopPropagation();
          mutate((d) => { const q = d.strokes.find((x) => x.id === st.id); if (q) q.visible = hidden; });
        } }),
        el('span', { class: 'drop sq', style: `background:${CHANNEL_CSS[st.mode === 'erase' ? 'all' : st.channel]}` }),
        el('span', { class: hidden ? 'muted' : '', text: `${i + 1} · ${st.mode === 'erase' ? 'Erase ' : ''}${labels[st.channel] || st.channel}` }),
        el('span', { class: 'muted', text: st.panel ? 'on a panel' : 'free' }),
        el('button', { class: 'small icon', text: '↑', title: 'Later (paints over more)', disabled: i === strokes.length - 1,
                       onclick: (e) => { e.stopPropagation(); move(st.id, 1); } }),
        el('button', { class: 'small icon', text: '↓', title: 'Earlier', disabled: i === 0,
                       onclick: (e) => { e.stopPropagation(); move(st.id, -1); } }),
        el('button', { class: 'small icon', text: '✕', title: 'Delete this stroke', onclick: (e) => {
          e.stopPropagation();
          if (on) select(null);
          mutate((d) => { d.strokes = d.strokes.filter((q) => q.id !== st.id); });
        } }));
      if (!on) return el('div', { class: 'leakrow' + (hidden ? ' off' : '') }, head);
      const F = makeFields(this.binder);
      const chans = Object.assign({}, this.meta.brush_channels, st.mode === 'erase' ? { all: 'Everything' } : {});
      return el('div', { class: 'leakrow on' }, head,
        F.select('Mode', () => st.mode, (v) => { st.mode = v; if (v === 'add' && st.channel === 'all') st.channel = 'rust'; this.render(); },
          { add: 'Add', erase: 'Erase' }),
        F.select('Effect', () => st.channel, (v) => { st.channel = v; this.render(); }, chans),
        F.len('Radius', () => st.radius, (v) => { st.radius = v; }, { unit: 'cm', min: 0.002, max: 0.5, step: 0.5, px: false, digits: 1 }),
        F.range('Strength', () => st.strength, (v) => { st.strength = v; }, { min: 0.05, max: 1, step: 0.01 }),
        F.range('Hardness', () => st.hardness, (v) => { st.hardness = v; }, { min: 0, max: 0.95, step: 0.01 }),
        el('div', { class: 'btnrow' },
          el('button', { class: 'small', text: 'Use as brush', title: 'Copy these settings to the brush', onclick: () => {
            Object.assign(brush, { mode: st.mode, channel: st.channel, radius: st.radius, strength: st.strength, hardness: st.hardness });
            saveBrush();
            this.render();
          } })));
    });
    return el('section', {}, el('h4', { text: `Strokes (${strokes.length})` }),
      strokes.length ? el('div', { class: 'btnrow' },
        el('button', { class: 'small', text: 'Clear all', onclick: () => { select(null); mutate((d) => { d.strokes = []; }); } })) : null,
      strokes.length ? el('div', { class: 'strokelist' }, ...rows) : null,
      el('p', { class: 'hint', text: strokes.length
        ? 'Click a stroke to select and edit it; Delete removes the selected stroke while the Brush tool is active. Later strokes paint over earlier ones.'
        : 'Strokes you paint are listed here, where you can edit, hide, reorder or delete them.' }));
  }
}
