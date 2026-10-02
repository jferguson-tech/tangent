"""CPU renderer backed by the Numba pathtracer."""
import numpy as np

from .base import Renderer
from .hdri import EnvMap
from .scene import emitter_tables
from . import cpu_pathtracer as pt   # the only Numba import path


class CPURenderer(Renderer):
    name = "cpu-numba"

    def warmup(self):
        acc = np.zeros((4, 4, 3))
        Hm = np.zeros((8, 8))
        Nm = np.zeros((8, 8, 3))
        Nm[..., 2] = 1
        Vm = np.full((8, 8), 0.5)
        sc = np.array([-1, 1, -1, 1, 0, 0.01, 2, 2, 0.25, 0, 1, 0, 1, 0, 1e-4], np.float64)
        cam = np.array([0, -2, 2, 0, 0.7071, -0.7071, 1, 0, 0, 0, 0.7071, 0.7071, 0.4, 1.0])
        mat = np.array([0.5, 0.5, 0.5, 1.0, 0.3, 0.03, 0.03, 0.03, 0.6])
        lights = np.array([[0, -1, 1, 0, 0.7071, -0.7071, 1, 1, 1, 0.9, 0.8, 0.01, 0, 0]], np.float64)
        em = EnvMap(np.full((8, 16, 3), 0.5, np.float32))
        ep = np.array([1.0, 0.0, 0.0, 1.0])
        opt = np.array([2.0, 20.0, 0.02])
        Em = np.zeros((8, 8, 3))
        Em[3:5, 3:5] = 2.0
        emit = emitter_tables(Em, 2.0, 2.0)
        args = (em.img, em.blur, em.pdf, em.marg, em.cond, ep, *emit, np.array([1.0, 1.0, 1.0]), opt)
        pt.render_pass(acc, Hm, Nm, Vm, sc, cam, mat, lights, *args)
        sc[10] = 0.0
        pt.render_pass(acc, Hm, Nm, Vm, sc, cam, mat, lights, *args)

    def render_pass(self, scene, accum):
        pt.render_pass(accum, scene["Hm"], scene["Nm"], scene["Vm"], scene["sc"],
                       scene["cam"], scene["mat"], scene["lights"], scene["E"],
                       scene["EB"], scene["P"], scene["M"], scene["C"], scene["ep"],
                       scene["Em"], scene["EPD"], scene["ECDF"], scene["EIDX"], scene["emp"],
                       scene["opt"])
