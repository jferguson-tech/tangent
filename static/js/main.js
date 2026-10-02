// App entry: wires the store, editor, property panels, baker and render pane.

import { $, $$, el, clone, fmt, uid, download, snap, debounce } from './util.js';
import { api } from './api.js';
import { Store } from './store.js';
import { GLView } from './glview.js';
import { Editor } from './editor.js';
import { PanelProps, DocProps, WireProps, renderLayoutForm, newPanelStyle, makeFields, plainBinder } from './props.js';
import { WireLayer, SimController, resolveEnd, isStale } from './wires.js';
import { sagCurve } from './wiresim.js';
import { RenderPane, computeLights } from './render.js';

const MAP_FOR_MODE = { lit: 'normal', normal: 'normal', height: 'height', ao: 'ao', curvature: 'curvature', id: 'id', emissive: 'emissive' };

async function main() {
  const meta = await api.meta();
  const restored = Store.restore();
  const store = new Store(restored && Array.isArray(restored.panels) ? restored : clone(meta.default_document));

  // ---- GL view (falls back to outlines only if WebGL2 is missing) ----------
  let glview;
  try {
    glview = new GLView($('#gl'));
  } catch (e) {
    glview = { resize() {}, draw() {}, async setMaps() {} };
    $('#viewport').append(el('div', { class: 'render-msg', text: `WebGL2 unavailable: ${e.message}. Outlines only.` }));
  }

  // ---- editor --------------------------------------------------------------
  const editor = new Editor({
    viewport: $('#viewport'), overlay: $('#overlay'), glview, store,
    newPanelStyle: () => newPanelStyle(store, meta),
    onCursor: (p) => {
      const d = store.doc.texel_density;
      $('#stCursor').textContent = p ? `x ${fmt(p[0] * 100, 1)} cm  y ${fmt(p[1] * 100, 1)} cm  ·  px ${Math.floor(p[0] * d)}, ${Math.floor(p[1] * d)}` : '—';
      $('#stZoom').textContent = `zoom ${editor.zoomLabel()}`;
    },
    onToolChange: (t) => $$('#tools button').forEach((b) => b.classList.toggle('on', b.dataset.tool === t)),
  });

  // ---- wires: simulation controller and editor layer ---------------------------
  const sim = new SimController(store, {
    onFrame: () => { editor.requestDraw(); updateSimStatus(); },
    onState: () => { updateSimUI(); invalidateDetail(); baker.request('full'); },
  });
  editor.wires = new WireLayer(editor, store, sim);

  // ---- render pane + lighting shared with the editor -------------------------
  // Prefiltered HDRI levels for the editor's "Render lights" view.
  let envWanted = null;
  const loadEditorEnv = async (id, preset) => {
    const key = id === 'gradient' ? `${id}|${preset}` : id;
    if (glview.envId === key || envWanted === key || !glview.setEnv) return;
    envWanted = key;
    try {
      const data = await api.envGl(id, preset);
      if (envWanted !== key) return;
      glview.setEnv({ ...data, key });
      editor.requestDraw();
    } catch (e) {
      $('#stBake').textContent = `environment failed to load: ${e.message}`;
    } finally {
      if (envWanted === key) envWanted = null;
    }
  };
  const updateEditorLight = () => {
    const doc = store.doc;
    const s = render.settings;
    const L = computeLights(doc, s, meta);
    const sl = s.scene_light ?? 1;
    const dim = (c) => c.map((v) => v * sl);
    editor.bloom = { enabled: s.bloom.editor, threshold: s.bloom.threshold, intensity: s.bloom.intensity, radius: s.bloom.radius };
    Object.assign(editor.emis, { spill: s.editor.spill, spillGain: s.editor.spill_gain });
    editor.lightGizmo = L.lights.map((l) => ({ x: l.pos[0] + doc.canvas_w / 2, y: doc.canvas_h / 2 - l.pos[1] }));
    if ($('#matchLights').checked) {
      editor.light = {
        lights: L.lights.map((l) => ({ ...l, color: dim(l.color) })),
        albedo: s.material.albedo, metallic: s.material.metallic, roughness: s.material.roughness,
        wire: s.wire_material,
        ambient: dim(L.env[0].map((v) => v * 4)),
        env: { intensity: s.environment.intensity * sl, rotation: s.environment.rotation },
      };
      loadEditorEnv(s.environment.hdri, s.light.preset);
    } else {
      // Even raking light from the key light's direction, on a neutral clay material.
      const az = s.light.azimuth * Math.PI / 180, elv = Math.max(s.light.elevation, 15) * Math.PI / 180;
      editor.light = {
        lights: [{ pos: [Math.cos(elv) * Math.cos(az), Math.cos(elv) * Math.sin(az), Math.sin(elv)], color: dim([2.6, 2.6, 2.7]), directional: true }],
        albedo: [0.55, 0.56, 0.58], metallic: 0, roughness: 0.5, ambient: dim([0.22, 0.23, 0.26]),
        wire: { albedo: [0.12, 0.12, 0.13], metallic: 0, roughness: 0.5 },
      };
    }
    editor.requestDraw();
  };
  const render = new RenderPane(store, meta, { onLightsChanged: updateEditorLight });
  $('#matchLights').addEventListener('change', updateEditorLight);
  updateEditorLight();

  // Wires the simulation is moving are drawn live by the overlay, so bakes hide them.
  const bakeDoc = () => {
    const live = sim.activeIds();
    if (!live.size) return store.doc;
    const doc = clone(store.doc);
    for (const w of doc.wires) if (live.has(w.id)) w.visible = false;
    return doc;
  };

  // ---- zoom-aware detail -----------------------------------------------------------
  // The overview is at most 1024 px across the canvas. Zoomed in further, the
  // visible area is baked at up to the full texel density (never more than the
  // screen can show) and drawn over the overview.
  const DETAIL_MAX_PX = 2000;
  const detail = { norm: null, overviewPpm: 0, have: null, inflight: false, again: false, seq: 0 };
  const visibleRect = () => {
    const d = store.doc;
    const [ax, ay] = editor.toWorld(0, 0);
    const [bx, by] = editor.toWorld(editor.cssW, editor.cssH);
    const r = { x0: Math.max(0, ax), y0: Math.max(0, ay), x1: Math.min(d.canvas_w, bx), y1: Math.min(d.canvas_h, by) };
    return r.x1 > r.x0 && r.y1 > r.y0 ? r : null;
  };
  const detailWanted = () => {
    const d = store.doc;
    if (!detail.norm || !glview.setDetail || editor.tiles !== 1) return null;
    const vis = visibleRect();
    if (!vis) return null;
    let ppm = Math.min(d.texel_density, editor.view.scale * (editor.dpr || 1));
    if (ppm < detail.overviewPpm * 1.25) return null;          // the overview is already sharp enough
    const mx = (vis.x1 - vis.x0) * 0.15, my = (vis.y1 - vis.y0) * 0.15;
    const r = { x0: Math.max(0, vis.x0 - mx), y0: Math.max(0, vis.y0 - my),
                x1: Math.min(d.canvas_w, vis.x1 + mx), y1: Math.min(d.canvas_h, vis.y1 + my) };
    const side = Math.max(r.x1 - r.x0, r.y1 - r.y0);
    if (side * ppm > DETAIL_MAX_PX) ppm = DETAIL_MAX_PX / side;
    if (ppm < detail.overviewPpm * 1.25) return null;
    return { ...r, px_per_m: ppm, vis };
  };
  const covers = (have, want) => have && have.x0 <= want.vis.x0 + 1e-9 && have.y0 <= want.vis.y0 + 1e-9
    && have.x1 >= want.vis.x1 - 1e-9 && have.y1 >= want.vis.y1 - 1e-9 && have.px_per_m >= want.px_per_m * 0.85;
  const dropDetail = () => {
    glview.clearDetail && glview.clearDetail();
    detail.have = null;
    $('#stDetail').textContent = '';
  };
  const updateDetail = async () => {
    const want = detailWanted();
    if (!want) { if (detail.have) { dropDetail(); editor.requestDraw(); } return; }
    if (covers(detail.have, want)) return;
    if (detail.inflight) { detail.again = true; return; }
    detail.inflight = true;
    const seq = ++detail.seq;
    const token = glview.detailToken;
    const W = store.doc.canvas_w, H = store.doc.canvas_h;
    $('#stDetail').textContent = 'detail…';
    try {
      const { vis, ...region } = want;
      const r = await api.bakeDetail(bakeDoc(), region, detail.norm);
      if (seq !== detail.seq || token !== glview.detailToken) return;     // edited meanwhile
      const [x0, y0, x1, y1] = r.info.region;
      if (await glview.setDetail(r.maps, [x0 / W, y0 / H, x1 / W, y1 / H], token)) {
        detail.have = { x0, y0, x1, y1, px_per_m: r.info.px_per_m };
        $('#stDetail').textContent = `detail ${r.info.width}×${r.info.height} @ ${fmt(r.info.px_per_m, 0)} px/m · ${fmt(r.info.ms, 0)} ms`;
        editor.requestDraw();
      }
    } catch (e) {
      $('#stDetail').textContent = `detail failed: ${e.message}`;
    } finally {
      detail.inflight = false;
      if (detail.again) { detail.again = false; scheduleDetail(); }
    }
  };
  const scheduleDetail = debounce(updateDetail, 180);
  editor.onViewChange = scheduleDetail;
  // Any edit makes the patch stale: drop it now, re-bake after the overview.
  const invalidateDetail = () => { detail.norm = null; detail.seq++; dropDetail(); };

  // ---- baking: one request in flight, newest request wins -----------------------
  const baker = {
    inflight: false, queued: null,
    request(kind) {
      if (this.inflight) { if (this.queued !== 'full') this.queued = kind; return; }
      this._run(kind);
    },
    async _run(kind) {
      this.inflight = true;
      $('#bakeBusy').classList.add('on');
      const fast = kind === 'fast';
      const maps = fast
        ? [...new Set([MAP_FOR_MODE[editor.mode], ...(editor.mode === 'lit' ? ['emissive'] : [])])]
        : ['normal', 'height', 'ao', 'curvature', 'id', 'emissive', 'spill', 'wires'];
      if (fast && editor.mode === 'lit') maps.push('wires');
      try {
        const r = await api.bake(bakeDoc(), maps, fast ? 512 : 1024);
        if (r.wire_paths) Object.assign(editor.wires.paths, r.wire_paths);
        await glview.setMaps(r.maps);
        if (!fast || maps.includes('height')) editor.hrange = [r.info.height_min, r.info.height_max];
        if ('emissive_scale' in r.info) editor.emis.scale = r.info.emissive_scale;
        if ('spill_scale' in r.info) editor.emis.spillScale = r.info.spill_scale;
        if (!fast) {
          // The detail patch encodes with exactly the overview's scaling.
          const i = r.info;
          detail.norm = { height_min: i.height_min, height_max: i.height_max,
            emissive_scale: i.emissive_scale ?? 0, curvature_scale: i.curvature_scale ?? 0 };
          detail.overviewPpm = i.width / store.doc.canvas_w;
          scheduleDetail();
        }
        $('#stBake').textContent = `bake ${fmt(r.info.ms, 0)} ms · ${r.info.width}×${r.info.height}${r.info.width < r.info.full_width ? ` (preview of ${r.info.full_width}×${r.info.full_height})` : ''}`;
        if (!fast) {
          panelProps.setWarnings(r.warnings);
          layersWarn = r.warnings;
          renderLayers();
          const n = Object.keys(r.warnings).length;
          $('#stWarn').textContent = n ? `⚠ ${n} panel${n > 1 ? 's' : ''} with warnings` : '';
        }
        editor.requestDraw();
      } catch (e) {
        $('#stBake').textContent = `bake failed: ${e.message}`;
      } finally {
        this.inflight = false;
        $('#bakeBusy').classList.remove('on');
        const q = this.queued;
        this.queued = null;
        if (q) this._run(q);
      }
    },
  };

  // ---- side panels --------------------------------------------------------
  const action = (name) => {
    const doc = store.doc;
    const w = store.selectedWire;
    if (w) { wireAction(name, w); return; }
    const p = store.selectedPanel;
    if (!p) return;
    const i = doc.panels.indexOf(p);
    switch (name) {
      case 'delete':
        store.mutate(() => doc.panels.splice(i, 1));
        store.select(null);
        break;
      case 'duplicate': {
        const q = clone(p);
        const off = Math.max(store.prefs.snap * 2, 0.02);
        Object.assign(q, { id: uid(), name: `${p.name} copy`, x: p.x + off, y: p.y + off, locked: false });
        store.mutate(() => doc.panels.splice(i + 1, 0, q));
        store.select(q.id);
        break;
      }
      case 'up':
        if (i < doc.panels.length - 1) store.mutate(() => { [doc.panels[i], doc.panels[i + 1]] = [doc.panels[i + 1], doc.panels[i]]; });
        break;
      case 'down':
        if (i > 0) store.mutate(() => { [doc.panels[i], doc.panels[i - 1]] = [doc.panels[i - 1], doc.panels[i]]; });
        break;
      case 'lock':
        store.mutate(() => { p.locked = !p.locked; });
        break;
      case 'visible':
        store.mutate(() => { p.visible = !p.visible; });
        break;
    }
    panelProps.render();
  };

  const wireAction = (name, w) => {
    const doc = store.doc;
    const i = doc.wires.indexOf(w);
    switch (name) {
      case 'delete':
        store.mutate(() => doc.wires.splice(i, 1));
        store.select(null);
        break;
      case 'duplicate': {
        const q = clone(w);
        const off = Math.max(store.prefs.snap * 2, 0.02);
        q.id = 'w' + Math.random().toString(16).slice(2, 10);
        q.name = `${w.name} copy`;
        q.locked = false;
        for (const end of [q.a, q.b]) { end.dy = (end.dy || 0) + off; end.y = (end.y || 0) + off; }
        q.points = (q.points || []).map(([x, y]) => [x, y + off]);
        store.mutate(() => doc.wires.splice(i + 1, 0, q));
        store.select(q.id);
        break;
      }
      case 'lock': store.mutate(() => { w.locked = !w.locked; }); break;
      case 'visible': store.mutate(() => { w.visible = !w.visible; }); break;
    }
    renderInspector();
  };
  const resetWireShape = (w) => {
    store.mutate((d) => {
      w.points = sagCurve(resolveEnd(w.a, d), resolveEnd(w.b, d), w.slack, d.wire_sim.gravity_angle);
      w.settled = false;
    }, 'props');
  };

  const panelProps = new PanelProps($('#tabPanel'), store, meta, { onAction: action });
  const wireProps = new WireProps($('#tabPanel'), store, meta, { onAction: action, onResetShape: resetWireShape });
  const renderInspector = () => { if (!wireProps.render()) panelProps.render(); };
  const refreshInspector = () => { if (store.selectedWire) wireProps.refreshValues(); else panelProps.refreshValues(); };
  const docProps = new DocProps($('#tabDoc'), store, meta, { onPrefs: () => editor.requestDraw() });

  let layersWarn = {};
  function renderLayers() {
    const panels = store.doc.panels;
    $('#panelCount').textContent = panels.length ? `(${panels.length})` : '';
    const rows = [...panels].reverse().map((p) => el('div', {
      class: 'layer' + (p.id === store.selected ? ' sel' : ''),
      onclick: () => store.select(p.id),
    },
    el('button', { class: 'ico' + (p.visible ? ' on' : ''), title: p.visible ? 'Hide' : 'Show', text: '👁',
      onclick: (e) => { e.stopPropagation(); store.mutate(() => { p.visible = !p.visible; }); } }),
    el('button', { class: 'ico' + (p.locked ? ' on' : ''), title: p.locked ? 'Unlock' : 'Lock (kept by auto-layout)', text: p.locked ? '🔒' : '🔓',
      onclick: (e) => { e.stopPropagation(); store.mutate(() => { p.locked = !p.locked; }); } }),
    el('span', { class: 'name', text: p.name }),
    layersWarn[p.id] ? el('span', { class: 'wdot', title: layersWarn[p.id].join('\n'), text: '⚠' }) : null,
    el('span', { class: 'badge', text: p.mode === 'inset' ? 'in' : p.mode === 'max' ? 'max' : '' })));
    $('#layers').replaceChildren(...(rows.length ? rows : [el('div', { class: 'empty', text: 'No panels yet' })]));
    const sel = $('#layers .sel');
    if (sel) sel.scrollIntoView({ block: 'nearest' });
  }

  function renderWires() {
    const doc = store.doc;
    $('#wireCount').textContent = doc.wires.length ? `(${doc.wires.length})` : '';
    const rows = [...doc.wires].reverse().map((w) => el('div', {
      class: 'layer' + (w.id === store.selected ? ' sel' : ''),
      onclick: () => store.select(w.id),
    },
    el('button', { class: 'ico' + (w.visible ? ' on' : ''), title: w.visible ? 'Hide' : 'Show', text: '👁',
      onclick: (e) => { e.stopPropagation(); store.mutate(() => { w.visible = !w.visible; }); } }),
    el('button', { class: 'ico' + (w.locked ? ' on' : ''), title: w.locked ? 'Unlock' : 'Lock (kept by auto-layout, not simulated)', text: w.locked ? '🔒' : '🔓',
      onclick: (e) => { e.stopPropagation(); store.mutate(() => { w.locked = !w.locked; }); } }),
    el('span', { class: 'name', text: w.name }),
    isStale(w, doc) ? el('span', { class: 'badge stale', title: 'Not settled for its current endpoints', text: '●' }) : null,
    el('span', { class: 'badge', text: w.mode === 'sim' ? 'sim' : 'route' })));
    $('#wireList').replaceChildren(...(rows.length ? rows : [el('div', { class: 'empty', text: 'No wires. Press W to add one.' })]));
  }

  function updateSimStatus() {
    const n = sim.simWires().length;
    const st = sim.state;
    const extra = st === 'running' ? ` · ${fmt(sim.stepMs, 1)} ms/step · motion ${fmt(sim.sim.lastSpeed * 1000, 3)} mm` : '';
    $('#simStatus').textContent = `${st} · ${n} simulated wire${n === 1 ? '' : 's'}${extra}`;
  }

  function updateSimUI() {
    const st = sim.state;
    $('#simRun').textContent = st === 'running' ? '⏸ Pause' : st === 'paused' ? '▶ Resume' : '▶ Start';
    updateSimStatus();
    renderWires();
    editor.requestDraw();
  }

  // Gravity dial and simulation settings (stored in the document).
  const simBinder = {
    begin: () => store.checkpoint(),
    live: () => { sim._params(); drawDial(); },
    end: () => {
      for (const w of store.doc.wires) if (w.mode === 'sim' && sim.state !== 'running') w.settled = false;
      store.emit('doc', 'simsettings');
    },
  };
  function renderSimFields() {
    const S = store.doc.wire_sim;
    const F = makeFields(simBinder);
    $('#simFields').replaceChildren(
      F.num('Direction°', () => Math.round(S.gravity_angle), (v) => { S.gravity_angle = ((v % 360) + 360) % 360; }, { min: -360, max: 720, step: 15 }),
      F.range('Strength', () => S.gravity, (v) => { S.gravity = v; }, { min: 0, max: 3, step: 0.05 }),
      F.check('Collide', () => S.collide, (v) => { S.collide = v; }, 'Wires collide with each other and stack'),
      F.check('Auto settle', () => S.auto_resettle, (v) => { S.auto_resettle = v; }, 'Re-simulate automatically when a panel moves under a simulated wire'),
    );
    drawDial();
  }
  function drawDial() {
    const c = $('#gravDial'), g = c.getContext('2d');
    const a = (store.doc.wire_sim.gravity_angle * Math.PI) / 180;
    g.clearRect(0, 0, 64, 64);
    g.strokeStyle = '#303947';
    g.beginPath(); g.arc(32, 32, 26, 0, Math.PI * 2); g.stroke();
    const ex = 32 + Math.cos(a) * 22, ey = 32 + Math.sin(a) * 22;
    g.strokeStyle = '#39c6e6'; g.fillStyle = '#39c6e6'; g.lineWidth = 2.5;
    g.beginPath(); g.moveTo(32, 32); g.lineTo(ex, ey); g.stroke();
    g.beginPath();
    g.moveTo(ex + Math.cos(a) * 4, ey + Math.sin(a) * 4);
    g.lineTo(ex + Math.cos(a + 2.4) * 8, ey + Math.sin(a + 2.4) * 8);
    g.lineTo(ex + Math.cos(a - 2.4) * 8, ey + Math.sin(a - 2.4) * 8);
    g.fill();
    g.fillStyle = '#7d8896'; g.font = '9px monospace'; g.textAlign = 'center';
    g.fillText('g', 32, 35);
  }
  {
    const c = $('#gravDial');
    let dragging = false;
    const setFrom = (e) => {
      const r = c.getBoundingClientRect();
      const ang = (Math.atan2(e.clientY - r.top - r.height / 2, e.clientX - r.left - r.width / 2) * 180) / Math.PI;
      store.doc.wire_sim.gravity_angle = ((Math.round(ang / (e.shiftKey ? 1 : 15)) * (e.shiftKey ? 1 : 15)) % 360 + 360) % 360;
      simBinder.live();
    };
    c.addEventListener('pointerdown', (e) => { c.setPointerCapture(e.pointerId); dragging = true; simBinder.begin(); setFrom(e); });
    c.addEventListener('pointermove', (e) => { if (dragging) setFrom(e); });
    c.addEventListener('pointerup', () => { if (dragging) { dragging = false; simBinder.end(); } });
  }
  $('#simRun').addEventListener('click', () => {
    if (sim.state === 'running') sim.pause(); else sim.start().catch((e) => { $('#simStatus').textContent = `simulation failed: ${e.message}`; });
  });
  $('#simStep').addEventListener('click', () => sim.step());
  $('#simReset').addEventListener('click', () => sim.reset());
  $('#wireMode').value = store.prefs.wireMode || 'sim';
  $('#wireMode').addEventListener('change', (e) => { store.prefs.wireMode = e.target.value; store.savePrefs(); });

  // Simulated wires that have never settled start simulating on their own.
  const autoSettle = (force) => {
    if (sim.state === 'running') return;
    const doc = store.doc;
    const want = sim.simWires(doc).some((w) => force ? !w.settled : (doc.wire_sim.auto_resettle && isStale(w, doc)));
    if (want) sim.start().catch(() => {});
  };

  function updateDocInfo() {
    const d = store.doc;
    const rx = Math.round(d.canvas_w * d.texel_density), ry = Math.round(d.canvas_h * d.texel_density);
    $('#docInfo').textContent = `${rx} × ${ry} px · ${d.texel_density} px/m · ${fmt(d.canvas_w, 3)} × ${fmt(d.canvas_h, 3)} m${d.tiling ? ' · seamless' : ''}`;
    $('#btnUndo').disabled = !store.undoStack.length;
    $('#btnRedo').disabled = !store.redoStack.length;
  }

  // Snap grid control in the tools section.
  const snapRow = makeFields(plainBinder(() => { store.savePrefs(); editor.requestDraw(); docProps.render(); }),
    { density: () => store.doc.texel_density })
    .len('Snap grid', () => store.prefs.snap, (v) => { store.prefs.snap = v; }, { unit: 'cm', step: 0.5 });
  $('#snapField').replaceChildren(snapRow);

  // ---- store events ---------------------------------------------------------
  store.on((type, source) => {
    if (type === 'doc') {
      invalidateDetail();
      baker.request('full');
      render.schedule();
      updateDocInfo();
      renderLayers();
      renderWires();
      updateEditorLight();
      if (source !== 'sim') { sim.docChanged(); autoSettle(false); }
      if (source === 'simsettings') renderSimFields();
      if (source === 'props' || source === 'sim') refreshInspector(); else renderInspector();
      docProps.render();
      renderLayoutForm($('#layoutForm'), store, meta);
      snapRow.refresh();
      editor.requestDraw();
    } else if (type === 'live') {
      invalidateDetail();
      baker.request('fast');
      if (sim.running) sim.syncEnds();
      if (source === 'editor') refreshInspector();
      editor.requestDraw();
    } else if (type === 'select') {
      renderInspector();
      renderLayers();
      renderWires();
      editor.requestDraw();
    }
  });

  // ---- toolbar ----------------------------------------------------------------
  $$('#tools button').forEach((b) => b.addEventListener('click', () => editor.setTool(b.dataset.tool)));
  $$('#viewModes button').forEach((b) => b.addEventListener('click', () => {
    $$('#viewModes button').forEach((x) => x.classList.toggle('on', x === b));
    editor.mode = b.dataset.mode;
    editor.requestDraw();
  }));
  $('#tile3').addEventListener('change', (e) => { editor.tiles = e.target.checked ? 3 : 1; editor.fit(); scheduleDetail(); });
  $('#btnFit').addEventListener('click', () => editor.fit());
  $$('#rightTabs button').forEach((b) => b.addEventListener('click', () => {
    $$('#rightTabs button').forEach((x) => x.classList.toggle('on', x === b));
    $('#tabPanel').hidden = b.dataset.tab !== 'panel';
    $('#tabDoc').hidden = b.dataset.tab !== 'doc';
  }));

  $('#btnUndo').addEventListener('click', () => store.undo());
  $('#btnRedo').addEventListener('click', () => store.redo());
  $('#btnNew').addEventListener('click', () => {
    if (sim.running) sim.pause();
    sim.signature = null;
    sim.sim.setWires([]);
    store.replace(clone(meta.default_document));
    editor.fit();
    autoSettle(true);
  });
  $('#btnSave').addEventListener('click', () => {
    download(new Blob([JSON.stringify(store.doc, null, 1)], { type: 'application/json' }), 'panel.tangent.json');
  });
  $('#btnOpen').addEventListener('click', () => $('#fileOpen').click());
  $('#fileOpen').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    try {
      const doc = JSON.parse(await f.text());
      if (!Array.isArray(doc.panels)) throw new Error('not a Tangent document');
      store.replace(doc);
      editor.fit();
    } catch (err) {
      $('#stBake').textContent = `open failed: ${err.message}`;
    }
  });

  const runLayout = async () => {
    try {
      const r = await api.layout(store.doc, store.doc.layout);
      if (sim.running) sim.pause();
      sim.signature = null;
      sim.sim.setWires([]);
      store.replace(r.doc);
      autoSettle(true);
    } catch (e) {
      $('#stBake').textContent = `layout failed: ${e.message}`;
    }
  };
  $('#btnGenerate').addEventListener('click', runLayout);
  $('#btnShuffleWires').addEventListener('click', async () => {
    const params = { ...store.doc.layout, wire_seed: 1 + Math.floor(Math.random() * 1e6) };
    try {
      const r = await api.shuffleWires(store.doc, params);
      if (sim.running) sim.pause();
      sim.signature = null;
      sim.sim.setWires([]);
      store.replace(r.doc);
      autoSettle(true);
    } catch (e) {
      $('#stBake').textContent = `wire shuffle failed: ${e.message}`;
    }
  });
  $('#btnShuffle').addEventListener('click', () => {
    store.doc.layout.seed = Math.floor(Math.random() * 1e6);
    store.doc.layout.wire_seed = 0;
    renderLayoutForm($('#layoutForm'), store, meta);
    runLayout();
  });

  // ---- export dialog ------------------------------------------------------------
  const dlg = $('#exportDlg');
  $('#btnExport').addEventListener('click', () => {
    const d = store.doc;
    const rx = Math.round(d.canvas_w * d.texel_density), ry = Math.round(d.canvas_h * d.texel_density);
    $('#exportInfo').textContent = `${rx} × ${ry} px at ${d.texel_density} px/m · ${d.normal_convention === 'gl' ? 'OpenGL (Y+)' : 'DirectX (Y−)'} normals`;
    $('#exportStatus').textContent = '';
    dlg.showModal();
  });
  $('#exportGo').addEventListener('click', async (e) => {
    e.preventDefault();
    const form = $('#exportForm');
    const maps = $$('input[name=map]:checked', form).map((c) => c.value);
    if (!maps.length) { $('#exportStatus').textContent = 'Pick at least one map.'; return; }
    const name = form.elements.name.value.trim() || 'panel';
    const opts = { maps, bits: +form.elements.bits.value, name };
    if (form.elements.ss.value) opts.ss = +form.elements.ss.value;
    $('#exportStatus').textContent = 'Baking at full resolution…';
    $('#exportGo').disabled = true;
    try {
      const r = await api.export(store.doc, opts);
      download(await r.blob(), `${name.replace(/[^A-Za-z0-9_.-]+/g, '_')}_maps.zip`);
      dlg.close();
    } catch (err) {
      $('#exportStatus').textContent = `Export failed: ${err.message}`;
    } finally {
      $('#exportGo').disabled = false;
    }
  });

  // ---- keyboard -------------------------------------------------------------------
  window.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || dlg.open) return;
    const ctrl = e.ctrlKey || e.metaKey;
    const k = e.key.toLowerCase();
    if (ctrl && k === 'z' && !e.shiftKey) { e.preventDefault(); store.undo(); return; }
    if (ctrl && (k === 'y' || (k === 'z' && e.shiftKey))) { e.preventDefault(); store.redo(); return; }
    if (ctrl && k === 'd') { e.preventDefault(); action('duplicate'); return; }
    if (ctrl && k === 's') { e.preventDefault(); $('#btnSave').click(); return; }
    if (ctrl) return;
    if (k === ' ') { editor.space = true; e.preventDefault(); return; }
    if (k === 'v') editor.setTool('select');
    else if (k === 'r') editor.setTool('draw');
    else if (k === 'w') editor.setTool('wire');
    else if (k === 'f') editor.fit();
    else if (k === 'escape') {
      if (editor.tool === 'wire' && editor.wires.pending) { editor.wires.cancel(); return; }
      editor.setTool('select');
      store.select(null);
    }
    else if (k === 'delete' || k === 'backspace') action('delete');
    else if (k === 'l') action('lock');
    else if (k === 'h') action('visible');
    else if (k === ']') action('up');
    else if (k === '[') action('down');
    else if (k.startsWith('arrow')) {
      const p = store.selectedPanel;
      if (!p || p.locked) return;
      e.preventDefault();
      const step = (store.prefs.snap || 0.005) * (e.shiftKey ? 10 : 1);
      const dx = k === 'arrowleft' ? -step : k === 'arrowright' ? step : 0;
      const dy = k === 'arrowup' ? -step : k === 'arrowdown' ? step : 0;
      store.mutate(() => { p.x = snap(p.x + dx, step / (e.shiftKey ? 10 : 1)); p.y = snap(p.y + dy, step / (e.shiftKey ? 10 : 1)); }, 'editor');
    }
  });
  window.addEventListener('keyup', (e) => { if (e.key === ' ') editor.space = false; });

  // Debug handle for the browser console and automated UI checks.
  window.__tangent = { store, editor, render, baker, meta, sim, detail };

  // ---- first paint -------------------------------------------------------------------
  renderLayoutForm($('#layoutForm'), store, meta);
  renderInspector();
  docProps.render();
  renderLayers();
  renderWires();
  renderSimFields();
  updateSimUI();
  updateDocInfo();
  baker.request('full');
  autoSettle(true);
}

main().catch((e) => {
  document.body.append(el('div', { class: 'warnbox', text: `Failed to start: ${e.message}` }));
  console.error(e);
});
