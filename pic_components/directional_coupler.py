"""Directional couplers with selectable custom Euler L or S bends.

Use directional_coupler(bend_type="l") or directional_coupler(bend_type="s").
All lengths, widths and gaps are micrometres. Activate a PDK before building.
"""
from __future__ import annotations

from math import isfinite
import gdsfactory as gf

from .technology import waveguide_xs
from .bends import euler_l_bend, euler_s_bend

def move_to_anchor(selection, x=None, y=None, anchor="bottom_center"):
    """Move references using an anchor on their combined bounding box.

    x and y are absolute parent coordinates in micrometres. None leaves that
    coordinate unchanged. Call before flattening and before exporting ports.
    """
    anchors = {
        "top_left": (0.0, 1.0),
        "top_center": (0.5, 1.0),
        "top_right": (1.0, 1.0),
        "center_left": (0.0, 0.5),
        "center": (0.5, 0.5),
        "center_right": (1.0, 0.5),
        "bottom_left": (0.0, 0.0),
        "bottom_center": (0.5, 0.0),
        "bottom_right": (1.0, 0.0),
    }
    if anchor not in anchors:
        raise ValueError(f"anchor must be one of: {', '.join(anchors)}")
    if any(value is not None and not isfinite(float(value)) for value in (x, y)):
        raise ValueError("Target coordinates must be finite or None")

    refs = tuple(selection) if isinstance(selection, (list, tuple)) else (selection,)
    if not refs:
        raise ValueError("selection must contain at least one reference")
    if len({id(ref) for ref in refs}) != len(refs):
        raise ValueError("selection must not contain the same reference twice")
    if x is None and y is None:
        return

    xmin = min(float(ref.dxmin) for ref in refs)
    xmax = max(float(ref.dxmax) for ref in refs)
    ymin = min(float(ref.dymin) for ref in refs)
    ymax = max(float(ref.dymax) for ref in refs)
    fx, fy = anchors[anchor]
    anchor_x = xmin + fx * (xmax - xmin)
    anchor_y = ymin + fy * (ymax - ymin)
    dx = 0.0 if x is None else float(x) - anchor_x
    dy = 0.0 if y is None else float(y) - anchor_y
    for ref in refs:
        ref.dmove((dx, dy))



def _validate_path(straight_length, wg_width, bend_q):
    for name, value in (
        ("straight_length", straight_length),
        ("wg_width", wg_width),
        ("bend_q", bend_q),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")



@gf.cell
def s_bend_straight_s_bend(
    straight_length: float = 80.0,
    wg_width: float = 1.0,
    bend_q: float = 5.0,
    hierarchical: bool = False,
) -> gf.Component:
    """Return the original S bend + straight + S bend as a two-port cell."""
    _validate_path(straight_length, wg_width, bend_q)
    c = gf.Component()
    xs = waveguide_xs(wg_width)
    s_bend = euler_s_bend(q=bend_q, width=wg_width)

    bend_in = c << s_bend
    bend_in.dmirror_x()

    middle_straight = c << gf.components.straight(
        length=straight_length,
        cross_section=xs,
    )
    middle_straight.connect("o1", bend_in.ports["o2"])

    bend_out = c << s_bend
    bend_out.connect("o1", middle_straight.ports["o2"])

    c.add_port("o1", port=bend_in.ports["o1"])
    c.add_port("o2", port=bend_out.ports["o2"])
    middle_y = float(middle_straight.ports["o1"].dy)

    if not hierarchical:
        c.flatten()
    c.info["straight_length"] = straight_length
    c.info["waveguide_width"] = wg_width
    c.info["middle_y"] = middle_y
    return c



@gf.cell
def l_bend_straight_l_bend(
    straight_length: float = 80.0,
    wg_width: float = 1.0,
    bend_q: float = 5.0,
    hierarchical: bool = False,
) -> gf.Component:
    """Return L bend + horizontal straight + L bend.

    The straight is at the top of the path; both external ports face down.
    o1 is on the left and o2 is on the right. Each custom L bend turns 90 deg.
    """
    _validate_path(straight_length, wg_width, bend_q)
    c = gf.Component()
    xs = waveguide_xs(wg_width)
    l_bend = euler_l_bend(
        q=bend_q,
        width=wg_width,
        theta_deg=45.0,
        handedness="left",
    )

    # Reflect the first L across y=x: north, then east into the straight.
    bend_in = c << l_bend
    bend_in.dmirror(p1=(0.0, 0.0), p2=(1.0, 1.0))
    # Anchor the input port at the origin, including any mirror translation.
    bend_in.dmove((
        -float(bend_in.ports["o1"].dx),
        -float(bend_in.ports["o1"].dy),
    ))

    middle_straight = c << gf.components.straight(
        length=straight_length,
        cross_section=xs,
    )
    middle_straight.connect("o1", bend_in.ports["o2"])

    # Reflect the second L across the horizontal axis: east, then south.
    # A translation preserves this reflection when positioning the join.
    bend_out = c << l_bend
    bend_out.dmirror(p1=(0.0, 0.0), p2=(1.0, 0.0))
    bend_out.dmove((
        float(middle_straight.ports["o2"].dx - bend_out.ports["o1"].dx),
        float(middle_straight.ports["o2"].dy - bend_out.ports["o1"].dy),
    ))

    c.add_port("o1", port=bend_in.ports["o1"])
    c.add_port("o2", port=bend_out.ports["o2"])
    c.info["middle_y"] = float(middle_straight.ports["o1"].dy)
    c.info["straight_length"] = straight_length
    c.info["waveguide_width"] = wg_width
    if not hierarchical:
        c.flatten()
    return c



def _mirrored_path_pair(
    path,
    straight_length,
    wg_width,
    gap,
    move_selection,
    target_x,
    target_y,
    reference_point,
    hierarchical=False,
):
    """Place a reflected copy above a path using the requested edge gap."""
    c = gf.Component()
    first_path = c << path
    mirrored_path = c << path
    mirrored_path.dmirror(p1=(0.0, 0.0), p2=(1.0, 0.0))

    # Bounding boxes include waveguide width: do not add the width again.
    vertical_shift = first_path.dymax - mirrored_path.dymin + gap
    mirrored_path.dmovey(vertical_shift)

    selections = {
        "first": (first_path,),
        "mirrored": (mirrored_path,),
        "pair": (first_path, mirrored_path),
    }
    move_to_anchor(
        selections[move_selection],
        x=target_x,
        y=target_y,
        anchor=reference_point,
    )

    # Measure from transformed ports instead of assuming that mirroring
    # contributes no translation. This also handles optional absolute moves.
    middle_offset = float(path.info["middle_y"]) - float(path.ports["o1"].dy)
    first_middle_y = float(first_path.ports["o1"].dy) + middle_offset
    mirrored_middle_y = float(mirrored_path.ports["o1"].dy) - middle_offset
    center_spacing = abs(mirrored_middle_y - first_middle_y)
    vertical_edge_separation = float(mirrored_path.dymin - first_path.dymax)

    if first_path.ports["o1"].dy >= mirrored_path.ports["o1"].dy:
        upper_path, lower_path = first_path, mirrored_path
    else:
        upper_path, lower_path = mirrored_path, first_path

    c.add_port("o1", port=lower_path.ports["o1"])
    c.add_port("o2", port=upper_path.ports["o1"])
    c.add_port("o3", port=upper_path.ports["o2"])
    c.add_port("o4", port=lower_path.ports["o2"])

    if not hierarchical:
        c.flatten()
    c.info["gap"] = gap  # Requested gap before optional individual placement.
    c.info["vertical_edge_separation"] = vertical_edge_separation
    c.info["center_spacing"] = center_spacing
    c.info["straight_length"] = straight_length
    c.info["waveguide_width"] = wg_width
    return c



@gf.cell
def directional_coupler(
    bend_type: str = "s",
    straight_length: float = 80.0,
    wg_width: float = 1.0,
    bend_q: float = 5.0,
    gap: float = 0.8,
    move_selection: str = "pair",
    target_x: float | None = None,
    target_y: float | None = None,
    reference_point: str = "bottom_center",
    hierarchical: bool = False,
) -> gf.Component:
    """Return an S-bend or L-bend directional coupler.

    hierarchical=True preserves paths, straights and bends as child cells.
    bend_type: "s" or "l", case-insensitive.
    straight_length: length of each parallel coupling straight in um.
    gap: edge-to-edge spacing in um; use 0.0 for touching waveguides.

    S: preserves the original geometry and port numbering, with horizontal
       external ports (o1/o2 on the right, o3/o4 on the left).
    L: vertical external ports; o1/o4 face down, o2/o3 face up.
       Positions are bottom-left, top-left, top-right, bottom-right.

    Optional absolute placement uses the selected bounding-box anchor.
    Moving "pair" preserves the gap. Moving one path can change the gap
    or alignment. None leaves that target coordinate unchanged.
    """
    if not isinstance(bend_type, str) or bend_type.strip().lower() not in ("s", "l"):
        raise ValueError("bend_type must be 's' or 'l'")
    kind = bend_type.strip().lower()
    _validate_path(straight_length, wg_width, bend_q)
    if not isfinite(gap) or gap < 0:
        raise ValueError("gap must be finite and non-negative")
    if move_selection not in ("first", "mirrored", "pair"):
        raise ValueError("move_selection must be 'first', 'mirrored', or 'pair'")

    path_factory = s_bend_straight_s_bend if kind == "s" else l_bend_straight_l_bend
    path = path_factory(
        straight_length=straight_length,
        wg_width=wg_width,
        bend_q=bend_q,
        hierarchical=hierarchical,
    )
    c = _mirrored_path_pair(
        path, straight_length, wg_width, gap,
        move_selection, target_x, target_y, reference_point, hierarchical,
    )
    c.info["bend_type"] = kind.upper()
    return c



@gf.cell
def zero_gap_s_bend_pair(
    straight_length: float = 80.0,
    wg_width: float = 1.0,
    bend_q: float = 5.0,
    gap: float = 0.8,
    move_selection: str = "pair",
    target_x: float | None = None,
    target_y: float | None = None,
    reference_point: str = "bottom_center",
    hierarchical: bool = False,
) -> gf.Component:
    """Original S-bend entry point; existing scripts can keep using this call."""
    return directional_coupler(
        bend_type="s",
        straight_length=straight_length,
        wg_width=wg_width,
        bend_q=bend_q,
        gap=gap,
        move_selection=move_selection,
        target_x=target_x,
        target_y=target_y,
        reference_point=reference_point,
        hierarchical=hierarchical,
    )



@gf.cell
def directional_coupler_l(
    straight_length: float = 80.0,
    wg_width: float = 1.0,
    bend_q: float = 5.0,
    gap: float = 0.8,
    move_selection: str = "pair",
    target_x: float | None = None,
    target_y: float | None = None,
    reference_point: str = "bottom_center",
    hierarchical: bool = False,
) -> gf.Component:
    """Convenience entry point for the L-bend directional coupler."""
    return directional_coupler(
        bend_type="l",
        straight_length=straight_length,
        wg_width=wg_width,
        bend_q=bend_q,
        gap=gap,
        move_selection=move_selection,
        target_x=target_x,
        target_y=target_y,
        reference_point=reference_point,
        hierarchical=hierarchical,
    )

