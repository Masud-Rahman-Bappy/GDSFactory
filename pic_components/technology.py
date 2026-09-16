"""Shared defaults and layer names, defined locally for this project.

Dimensions are micrometres. Importing this module does not activate a PDK.
PAD (13/0) denotes passivation openings; metal pads use ROUTING (12/0).
The project chip outline uses CHIP_OUTLINE (290/0), preserving DIE (101/0).
"""
from __future__ import annotations

import gdsfactory as gf
from gdsfactory.typings import Layer

class Settings:
    """Shared layout defaults, in micrometres."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(Settings, cls).__new__(cls)
        return cls.instance

    DEFAULT_WG_WIDTH = 0.5
    DEFAULT_RADIUS = 5.0
    DEFAULT_EDGE_SEP = 100.0
    DEFAULT_ROUTE_WIDTH = 25.0
    DEFAULT_TEXT_SIZE = 50.0
    DEFAULT_DXDY = 30.0
    DEFAULT_GRATING_DIST = 250.0



class LayerMap:
    """Named (GDS layer, datatype) pairs; no geometry or PDK activation."""

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(LayerMap, cls).__new__(cls)
        return cls.instance

    WG: Layer = (1, 0)
    LABEL: Layer = (2, 0)
    BOSCH: Layer = (7, 0)
    HEATER: Layer = (11, 0)
    ROUTING: Layer = (12, 0)
    PAD: Layer = (13, 0)  # Passivation opening, distinct from the metal pad.
    FLOORPLAN: Layer = (100, 0)
    DIE: Layer = (101, 0)
    SEM: Layer = (200, 0)
    ANT_EDGE_TRENCH: Layer = (201, 0)
    ANT_HANDLING: Layer = (202, 0)
    ANT_THERMAL_TRENCH: Layer = (203, 0)
    ANNOTATION: Layer = (210, 0)
    CHIP_OUTLINE: Layer = (290, 0)  # Existing full-chip outline for this project.
    WAFER: Layer = (999, 0)



LAYERS = LayerMap


def waveguide_xs(width=None, layer=None, radius=None):
    """Return an optical cross-section with o1/o2 ports; dimensions are um."""
    width = Settings.DEFAULT_WG_WIDTH if width is None else width
    layer = LAYERS.WG if layer is None else layer
    radius = Settings.DEFAULT_RADIUS if radius is None else radius
    section = gf.Section(
        width=width, layer=layer, port_names=("o1", "o2"),
        port_types=("optical", "optical"), name="Wvg",
    )
    return gf.CrossSection(sections=(section,), radius=radius)



def routing_xs(rtWidth=None, layer=None):
    """Return an electrical cross-section, reading defaults on each call."""
    rtWidth = Settings.DEFAULT_ROUTE_WIDTH if rtWidth is None else rtWidth
    layer = LAYERS.ROUTING if layer is None else layer
    return gf.cross_section.cross_section(
        width=rtWidth, layer=layer, port_names=("e0", "e1"),
        port_types=("electrical", "electrical"),
    )

