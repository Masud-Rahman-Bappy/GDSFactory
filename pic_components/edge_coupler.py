"""Linear tapers and complete edge couplers, in micrometres.

The default edge coupler has a 25 um x 0.35 um tip followed by a
182 um linear taper to a 1 um waveguide. o1 is at the free tip, facing
west; o2 is at the broad taper end, facing east. Total length is 207 um.
The deep-trench mask is added separately by PIC_full.py.
"""
from __future__ import annotations

from math import isfinite
import gdsfactory as gf

from .technology import LAYERS


@gf.cell
def linear_taper(
    length: float = 182.0,
    width1: float = 0.35,
    width2: float = 1.0,
    layer: gf.typings.LayerSpec = LAYERS.WG,
) -> gf.Component:
    """Return a two-port linear width transition on the requested layer."""
    for name, value in (("length",length),("width1",width1),("width2",width2)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    c = gf.Component()
    c.add_polygon([(0,-width1/2),(length,-width2/2),
                   (length,width2/2),(0,width1/2)], layer=layer)
    c.add_port("o1",center=(0,0),width=width1,orientation=180,layer=layer)
    c.add_port("o2",center=(length,0),width=width2,orientation=0,layer=layer)
    return c


@gf.cell
def edge_coupler(
    tip_length: float = 25.0,
    tip_width: float = 0.35,
    taper_length: float = 182.0,
    wg_width: float = 1.0,
    layer: gf.typings.LayerSpec = LAYERS.WG,
    connect_tip: bool = True,
    hierarchical: bool = False,
) -> gf.Component:
    """Build the narrow edge extension and its connecting taper.

    hierarchical=True keeps the tip and taper editable in their hierarchy.
    connect_tip=False returns only the tip, for the legacy disconnected-tip
    chip option. In that case o2 lies at the narrow tip's inner end.
    Rotate/translate a reference to place a left-facing or bottom-facing tip.
    """
    for name,value in (("tip_length",tip_length),("tip_width",tip_width),
                       ("taper_length",taper_length),("wg_width",wg_width)):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isinstance(connect_tip,bool):
        raise ValueError("connect_tip must be True or False")
    c = gf.Component()
    c.add_polygon([(0,-tip_width/2),(tip_length,-tip_width/2),
                   (tip_length,tip_width/2),(0,tip_width/2)],layer=layer)
    c.add_port("o1",center=(0,0),width=tip_width,orientation=180,
               layer=layer,port_type="optical")
    if connect_tip:
        taper = c << linear_taper(length=taper_length,width1=tip_width,
                                   width2=wg_width,layer=layer)
        taper.dmovex(tip_length)
        c.add_port("o2",port=taper.ports["o2"])
    else:
        c.add_port("o2",center=(tip_length,0),width=tip_width,orientation=0,
                   layer=layer,port_type="optical")
    c.info["tip_length_um"] = tip_length
    c.info["tip_width_um"] = tip_width
    c.info["taper_length_um"] = taper_length if connect_tip else 0.0
    c.info["total_length_um"] = tip_length + (taper_length if connect_tip else 0.0)
    c.info["connected_tip"] = connect_tip
    if not hierarchical:
        c.flatten()
    return c
