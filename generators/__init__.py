from .base import Generator, Grid
from .panels import PanelGenerator

GENERATORS = {"panels": PanelGenerator()}


def get_generator(name):
    return GENERATORS.get(name, GENERATORS["panels"])
