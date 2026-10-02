"""Progressive render job manager.

One background worker owns the renderer. Starting a new job cancels the
current one after its in-flight pass (passes are short). The browser polls
for the latest frame by version number.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from core.png import encode_png
from .scene import build_scene, tonemap


class RenderManager:
    def __init__(self, renderer):
        self.renderer = renderer
        self._cv = threading.Condition()
        self._job = None
        self._job_id = 0
        self._frame = None            # (version, png bytes, info)
        self._version = 0
        self._state = "compiling"
        self._error = None
        # Frames are tone-mapped, bloomed and PNG-encoded on their own thread,
        # always from the newest snapshot, so encoding never stalls rendering.
        self._pub_cv = threading.Condition()
        self._pub_pending = None
        threading.Thread(target=self._publisher, name="publisher", daemon=True).start()
        self._thread = threading.Thread(target=self._run, name="pathtracer", daemon=True)
        self._thread.start()

    # ---- public API (request threads) -----------------------------------
    def start(self, doc, settings):
        scene = build_scene(doc, settings)
        with self._cv:
            self._job_id += 1
            self._job = (self._job_id, scene)
            self._cv.notify_all()
            return self._job_id

    def stop(self):
        with self._cv:
            self._job_id += 1
            self._job = None
            if self._state == "rendering":
                self._state = "idle"
            self._cv.notify_all()

    def frame(self, after=-1):
        with self._cv:
            if self._frame and self._frame[0] > after:
                return self._frame
            return None

    def status(self):
        with self._cv:
            info = dict(self._frame[2]) if self._frame else {}
            info.update(state=self._state, error=self._error, renderer=self.renderer.name,
                        version=self._version)
            return info

    # ---- publisher ----------------------------------------------------
    def _queue_publish(self, accum, spp, job_id, scene, pass_ms, done):
        with self._pub_cv:
            self._pub_pending = (accum.copy(), spp, job_id, scene, pass_ms, done)
            self._pub_cv.notify()

    def _publisher(self):
        while True:
            with self._pub_cv:
                while self._pub_pending is None:
                    self._pub_cv.wait()
                item, self._pub_pending = self._pub_pending, None
            try:
                self._publish(*item)
            except Exception as e:  # keep publishing later frames
                with self._cv:
                    self._error = f"frame encode failed: {e}"

    # ---- worker -------------------------------------------------------
    def _publish(self, accum, spp, job_id, scene, pass_ms, done):
        s = scene["settings"]
        img = tonemap(accum, spp, s["exposure"], s["bloom"])
        data = encode_png(img, level=1)
        with self._cv:
            if job_id != self._job_id:
                return
            self._version += 1
            self._frame = (self._version, data, {
                "job": job_id, "spp": spp, "max_spp": s["max_spp"],
                "pass_ms": round(pass_ms, 1), "done": done,
                "width": s["width"], "height": s["height"],
            })
            if done:
                self._state = "done"

    def _run(self):
        try:
            self.renderer.warmup()
        except Exception as e:  # compile failure leaves the editor usable
            with self._cv:
                self._state, self._error = "error", f"pathtracer failed to compile: {e}"
            return
        with self._cv:
            self._state = "idle"
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job_id, scene = self._job
                self._state = "rendering"
            s = scene["settings"]
            accum = np.zeros((s["height"], s["width"], 3), np.float64)
            spp, last_pub, pass_ms = 0, 0.0, 0.0
            while True:
                with self._cv:
                    if self._job is None or self._job[0] != job_id:
                        break
                t0 = time.perf_counter()
                try:
                    self.renderer.render_pass(scene, accum)
                except Exception as e:
                    with self._cv:
                        self._state, self._error = "error", str(e)
                        self._job = None
                    break
                spp += 1
                pass_ms = (time.perf_counter() - t0) * 1000
                done = spp >= s["max_spp"]
                now = time.perf_counter()
                # Publish often at first, then at most ~4 times per second.
                if done or spp <= 4 or now - last_pub > 0.25:
                    self._queue_publish(accum, spp, job_id, scene, pass_ms, done)
                    last_pub = now
                if done:
                    with self._cv:
                        if self._job and self._job[0] == job_id:
                            self._job = None
                    break
