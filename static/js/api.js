// Thin wrappers over the Flask JSON API.

async function post(url, body, asBlob = false) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return asBlob ? r : r.json();
}

export const api = {
  meta: () => fetch('/api/meta').then((r) => r.json()),
  bake: (doc, maps, maxRes, ss = 1, fast = false) => post('/api/bake', { doc, maps, max_res: maxRes, ss, fast }),
  layout: (doc, params) => post('/api/layout', { doc, params }),
  bakeDetail: (doc, region, norm, maps) => post('/api/bake', { doc, region, norm, maps }),
  shuffleWires: (doc, params) => post('/api/layout/wires', { doc, params }),
  profile: (profile, points) => post('/api/profile', { profile, points }),
  export: (doc, opts) => post('/api/export', { doc, ...opts }, true),
  envList: () => fetch('/api/env').then((r) => r.json()),
  envGl: (id, preset) => fetch(`/api/env/gl?id=${encodeURIComponent(id)}&preset=${encodeURIComponent(preset)}`).then((r) => {
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  }),
  async envUpload(file) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch('/api/env/upload', { method: 'POST', body: fd });
    const j = await r.json().catch(() => ({ error: `${r.status} ${r.statusText}` }));
    if (!r.ok) throw new Error(j.error || `${r.status}`);
    return j;
  },
  renderStart: (doc, settings) => post('/api/render/start', { doc, settings }),
  renderStop: () => post('/api/render/stop', {}),
  renderStatus: () => fetch('/api/render/status').then((r) => r.json()),
  async renderFrame(after) {
    const r = await fetch(`/api/render/frame?after=${after}`, { cache: 'no-store' });
    if (r.status === 204) return null;
    if (!r.ok) throw new Error(`${r.status}`);
    return {
      version: +r.headers.get('X-Frame-Version'),
      info: JSON.parse(r.headers.get('X-Frame-Info') || '{}'),
      blob: await r.blob(),
    };
  },
};
