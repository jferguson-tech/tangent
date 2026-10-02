"""Renderer interface. CPU (Numba) today; GPU backends can implement this later."""


class Renderer:
    name = "base"

    def warmup(self):
        """Compile or initialize ahead of the first frame (optional)."""

    def render_pass(self, scene, accum):
        """Add one sample per pixel into accum, a float64 (h, w, 3) array."""
        raise NotImplementedError
