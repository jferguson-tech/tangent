// Document store with undo/redo and change notifications.
//
// Events: 'doc'    geometry/settings changed and committed (rebake + rerender)
//         'live'   geometry changing during a drag (fast rebake only)
//         'select' selection changed
// Each event carries a `source` so a view can skip refreshing itself.

import { clone, storage, debounce, normalizeDoc } from './util.js';

const AUTOSAVE_KEY = 'tangent.doc.v1';
const MAX_HISTORY = 150;

export class Store {
  constructor(doc) {
    this.doc = normalizeDoc(doc);
    this.selected = null;
    this.undoStack = [];
    this.redoStack = [];
    this.listeners = new Set();
    this.prefs = Object.assign({ snap: 0.01 }, storage.get('tangent.prefs', {}));
    this._autosave = debounce(() => storage.set(AUTOSAVE_KEY, this.doc), 600);
  }

  static restore() { return storage.get(AUTOSAVE_KEY, null); }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }

  emit(type, source = null) {
    if (type === 'doc') this._autosave();
    for (const fn of this.listeners) fn(type, source);
  }

  savePrefs() { storage.set('tangent.prefs', this.prefs); }

  // Call before a mutation that should be undoable.
  checkpoint() {
    this.undoStack.push(JSON.stringify(this.doc));
    if (this.undoStack.length > MAX_HISTORY) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  // Apply a mutation as one undo step and notify.
  mutate(fn, source = null) {
    this.checkpoint();
    fn(this.doc);
    this.emit('doc', source);
  }

  replace(doc, { undoable = true } = {}) {
    if (undoable) this.checkpoint();
    this.doc = normalizeDoc(doc);
    if (this.selected && !this.panel(this.selected) && !this.wire(this.selected)) this.selected = null;
    this.emit('doc');
    this.emit('select');
  }

  undo() {
    if (!this.undoStack.length) return;
    this.redoStack.push(JSON.stringify(this.doc));
    this.doc = JSON.parse(this.undoStack.pop());
    if (this.selected && !this.panel(this.selected) && !this.wire(this.selected)) this.selected = null;
    this.emit('doc');
    this.emit('select');
  }

  redo() {
    if (!this.redoStack.length) return;
    this.undoStack.push(JSON.stringify(this.doc));
    this.doc = JSON.parse(this.redoStack.pop());
    if (this.selected && !this.panel(this.selected) && !this.wire(this.selected)) this.selected = null;
    this.emit('doc');
    this.emit('select');
  }

  select(id, source = null) {
    if (this.selected === id) return;
    this.selected = id;
    this.emit('select', source);
  }

  panel(id = this.selected) { return this.doc.panels.find((p) => p.id === id) || null; }

  get selectedPanel() { return this.panel(this.selected); }

  wire(id = this.selected) { return (this.doc.wires || []).find((w) => w.id === id) || null; }

  get selectedWire() { return this.wire(this.selected); }

  snapshot() { return clone(this.doc); }
}
