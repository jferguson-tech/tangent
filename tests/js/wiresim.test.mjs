// Run with: node --test tests/js   (pytest runs this too when Node is installed)
import test from 'node:test';
import assert from 'node:assert/strict';
import { WireSim, sagCurve, polylineLength } from '../../static/js/wiresim.js';

const flatField = (h = 0) => ({ w: 64, h: 64, data: new Float32Array(64 * 64).fill(h), width: 1, height: 1, wrap: false });

function run(sim, frames = 600) {
  for (let i = 0; i < frames && !sim.settled; i++) sim.step();
  return sim;
}

const spec = (over = {}) => ({ id: 'w1', a: [0.2, 0.4], b: [0.8, 0.4], points: null, radius: 0.004, thick: 0.004, slack: 1.2, ...over });

test('sag curve has the requested length', () => {
  const P = sagCurve([0, 0], [1, 0], 1.2, 90, 200);
  assert.ok(Math.abs(polylineLength(P) - 1.2) / 1.2 < 0.02);
  assert.ok(P[100][1] > 0.28, 'sags toward +y (down) by default');
});

test('default gravity sags the wire down and it settles', () => {
  const sim = new WireSim();
  sim.setField(flatField());
  sim.setWires([spec({ points: [[0.2, 0.4], [0.8, 0.4]] })]);
  run(sim);
  assert.ok(sim.settled, 'reaches rest');
  const P = sim.shapes().w1;
  const mid = P[Math.floor(P.length / 2)];
  assert.ok(mid[1] > 0.55, `middle hangs below the endpoints (y=${mid[1]})`);
  assert.ok(Math.abs(mid[0] - 0.5) < 0.02, 'symmetric');
  assert.deepEqual(P[0], [0.2, 0.4]);
  assert.deepEqual(P[P.length - 1], [0.8, 0.4]);
  assert.ok(Math.abs(sim.length('w1') - 0.72) / 0.72 < 0.03, `keeps its length (${sim.length('w1')})`);
});

test('gravity direction is configurable', () => {
  const sim = new WireSim();
  sim.setField(flatField());
  sim.setParams({ gravityAngle: 0 });   // toward +x (right)
  sim.setWires([spec({ a: [0.5, 0.2], b: [0.5, 0.8] })]);
  run(sim);
  const P = sim.shapes().w1;
  assert.ok(P[Math.floor(P.length / 2)][0] > 0.6, 'sags to the right');
});

test('wire rests on the height field', () => {
  const f = flatField(0);
  for (let y = 0; y < 64; y++) for (let x = 24; x < 40; x++) f.data[y * 64 + x] = 0.02;   // raised strip
  const sim = new WireSim();
  sim.setField(f);
  sim.setWires([spec({ a: [0.1, 0.5], b: [0.9, 0.5], slack: 1.02 })]);
  run(sim);
  const w = sim.wires[0];
  for (let i = 0; i < w.n; i++) {
    const [h] = sim.sample(w.x[i], w.y[i]);
    assert.ok(w.z[i] >= h + w.thick - 1e-4, `particle ${i} is not inside the surface`);
  }
  const mid = Math.floor(w.n / 2);
  assert.ok(w.z[mid] > 0.02, 'drapes over the raised strip');
});

test('crossing wires stack when collision is on', () => {
  const make = (collide) => {
    const sim = new WireSim();
    sim.setField(flatField());
    sim.setParams({ collide, gravity: 0 });
    sim.setWires([
      spec({ id: 'h', a: [0.1, 0.5], b: [0.9, 0.5], slack: 1.0, radius: 0.006, thick: 0.006 }),
      spec({ id: 'v', a: [0.5, 0.1], b: [0.5, 0.9], slack: 1.0, radius: 0.006, thick: 0.006 }),
    ]);
    for (let i = 0; i < 200; i++) sim.step();
    const a = sim.wires[0], b = sim.wires[1];
    let minD = Infinity;
    for (let i = 0; i < a.n; i++) for (let j = 0; j < b.n; j++) {
      minD = Math.min(minD, Math.hypot(a.x[i] - b.x[j], a.y[i] - b.y[j], a.z[i] - b.z[j]));
    }
    return minD;
  };
  assert.ok(make(true) > 0.0105, 'kept apart by about two radii');
  assert.ok(make(false) < 0.004, 'passes through when off');
});

test('grabbing pulls the wire and releasing lets it fall back', () => {
  const sim = new WireSim();
  sim.setField(flatField());
  sim.setWires([spec()]);
  run(sim);
  assert.ok(sim.grabAt(0.5, 0.6, 0.2));
  sim.dragTo(0.5, 0.2);
  for (let i = 0; i < 60; i++) sim.step();
  const g = sim.wires[0];
  assert.ok(Math.abs(g.y[sim.grab.i] - 0.2) < 1e-9, 'grabbed point follows the cursor');
  assert.ok(!sim.settled);
  sim.release();
  run(sim);
  const P = sim.shapes().w1;
  assert.ok(P[Math.floor(P.length / 2)][1] > 0.55, 'falls back down');
});

test('many wires stay fast', () => {
  const sim = new WireSim();
  sim.setField(flatField());
  const specs = [];
  for (let k = 0; k < 100; k++) specs.push(spec({ id: 'w' + k, a: [0.1, 0.05 + k * 0.009], b: [0.9, 0.05 + k * 0.009], radius: 0.003, thick: 0.003 }));
  sim.setWires(specs);
  const t = performance.now();
  for (let i = 0; i < 10; i++) sim.step();
  const ms = (performance.now() - t) / 10;
  // Coarse guard against algorithmic regressions (an all-pairs collision would take
  // seconds); typical is ~40 ms on an idle machine, so busy machines still pass.
  assert.ok(ms < 250, `100 wires step in ${ms.toFixed(1)} ms`);
});

for (const gravity of [1, 3]) test(`a long thin wire draped over steep bevels comes to rest (gravity ${gravity})`, () => {
  // 2 m canvas, 512 px field with a raised 34 mm plateau whose edges slope at ~50 degrees.
  const n = 512, W = 2;
  const data = new Float32Array(n * n);
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) {
    const d = Math.min(x - 180, 330 - x, y - 120, 400 - y) * (W / n);   // distance inside the plateau (m)
    data[y * n + x] = Math.max(0, Math.min(0.034, d * 1.2));
  }
  const sim = new WireSim();
  sim.setField({ w: n, h: n, data, width: W, height: W, wrap: false });
  sim.setParams({ gravity });
  sim.setWires([
    { id: 'long', a: [0.4, 0.5], b: [1.55, 1.2], points: null, radius: 0.003, thick: 0.003, slack: 1.25 },
    { id: 'hose', a: [0.6, 1.3], b: [1.4, 0.4], points: null, radius: 0.007, thick: 0.007, slack: 1.2 },
  ]);
  run(sim, 1500);
  assert.ok(sim.settled, `settles within 25 s (motion ${(sim.lastSpeed * 1000).toFixed(3)} mm/frame)`);
  const w = sim.wires[0];
  for (let i = 0; i < w.n; i++) assert.ok(w.z[i] >= sim.sample(w.x[i], w.y[i])[0] + w.thick - 1e-3, 'stays on the surface');
});
