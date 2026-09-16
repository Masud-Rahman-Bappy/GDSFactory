"""MZI with L/S directional couplers and equal straight arms.

Call mzi_equal_arms(bend_type="l", arm_length=100.0, q=5.0).
All dimensions are micrometres except the dimensionless bend scaling q.
Chip-level heater overlays are added by PIC_full.py after MZI placement.
"""
from __future__ import annotations

from math import hypot, isfinite, pi, sqrt
import gdsfactory as gf

from .technology import waveguide_xs
from .bends import euler_l_bend, smooth_s_bend
from .directional_coupler import directional_coupler

def _check_bend_selection(component, kind: str, source: str, horizontal_outputs=False) -> None:
    """Verify geometry using port directions, independently of its label."""
    expected_axis = 90.0 if kind == "l" else 0.0
    for name in ("o1", "o2", "o3", "o4"):
        angle = float(component.ports[name].orientation)
        axis = 0.0 if horizontal_outputs and name in ('o3', 'o4') else expected_axis
        error = abs((angle - axis + 90.0) % 180.0 - 90.0)
        if error > 1e-6:
            raise RuntimeError(
                f"Requested {kind.upper()}, but {name} faces {angle:g} degrees. "
                f"Expected {'vertical' if axis == 90 else 'horizontal'} ports. "
                f"Check the saved module: {source}"
            )



def _check_mzi_join(first, second, tolerance: float) -> None:
    """Require coincident ports, opposing directions, equal widths and layers."""
    distance = hypot(float(first.dx - second.dx), float(first.dy - second.dy))
    angle_error = abs(
        (float(first.orientation) - float(second.orientation)) % 360.0 - 180.0
    )
    if distance > tolerance or angle_error > 1e-6:
        raise ValueError(
            "Both arms must meet the second coupler. "
            f"Port mismatch: {distance:.6g} um, {angle_error:.6g} degrees. "
            "Check that the coupler has matching input/output port spacing."
        )
    if abs(float(first.dwidth - second.dwidth)) > tolerance:
        raise ValueError("Connected waveguide widths do not match")
    if first.layer != second.layer:
        raise ValueError("Connected waveguide layers do not match")



def _l_coupler_with_arm_ports(coupler, bend, side: str) -> gf.Component:
    """Add two L turns on the arm side of an L coupler.

    The bare L coupler's upper/lower ports point up/down. The added turns
    provide two parallel horizontal connections for the MZI straight arms.
    The opposite side retains the original vertical external ports.
    """
    c = gf.Component()
    dc = c << coupler
    top_turn = c << bend
    bottom_turn = c << bend
    ports = {name: dc.ports[name] for name in ("o1", "o2", "o3", "o4")}
    tolerance = max(float(c.kcl.dbu), 1e-6)

    # Use rotations of the same left-handed L bend. Reversing the port used
    # for connection gives the complementary turn without mirror ambiguity.
    if side == "output":
        top_turn.connect("o2", dc.ports["o3"])
        bottom_turn.connect("o1", dc.ports["o4"])
        _check_mzi_join(top_turn.ports["o2"], dc.ports["o3"], tolerance)
        _check_mzi_join(bottom_turn.ports["o1"], dc.ports["o4"], tolerance)
        ports["o3"] = top_turn.ports["o1"]
        ports["o4"] = bottom_turn.ports["o2"]
    elif side == "input":
        top_turn.connect("o1", dc.ports["o2"])
        bottom_turn.connect("o2", dc.ports["o1"])
        _check_mzi_join(top_turn.ports["o1"], dc.ports["o2"], tolerance)
        _check_mzi_join(bottom_turn.ports["o2"], dc.ports["o1"], tolerance)
        ports["o2"] = top_turn.ports["o2"]
        ports["o1"] = bottom_turn.ports["o1"]
    else:
        raise ValueError("side must be 'input' or 'output'")

    for name, port in ports.items():
        c.add_port(name, port=port)
    return c



def _l_coupler_with_s_arm_ports(coupler, bend, side: str, horizontal_outputs=False,
                              output_bend_type='s') -> gf.Component:
    """Replace each arm-side L pair by one S with the same tangent ports."""
    if side not in ('input', 'output'):
        raise ValueError("side must be 'input' or 'output'")
    width = float(coupler.info['waveguide_width'])
    span = 2.0 * float(bend.ports['o2'].dx - bend.ports['o1'].dx)
    s_cell = smooth_s_bend(length=span, offset=-span if side == 'output' else span,
                           width=width)
    path = gf.Component()
    if horizontal_outputs and side != 'input':
        raise ValueError('Horizontal outputs apply only to the output coupler')
    output_cell = None
    output_s_length = 0.0
    if horizontal_outputs:
        if output_bend_type == 'l':
            # Two full-sized Euler quarters: east -> south -> east.
            # Keep the supplied q and width; the vertical offset is 2 spans.
            output_cell = gf.Component()
            first = output_cell << bend
            first.dmirror(p1=(0.0,0.0),p2=(1.0,0.0))
            second = output_cell << bend
            second.connect('o1',first.ports['o2'])
            _check_mzi_join(second.ports['o1'],first.ports['o2'],path.kcl.dbu)
            output_cell.add_port('o1',port=first.ports['o1'])
            output_cell.add_port('o2',port=second.ports['o2'])
        else:
            output_cell = smooth_s_bend(length=span,offset=-span/2,width=width)
            output_s_length = float(output_cell.info['centerline_length'])
    outer = path << (output_cell if horizontal_outputs else bend)
    middle = path << gf.components.straight(
        length=float(coupler.info['straight_length']), cross_section=waveguide_xs(width),
    )
    transition = path << s_cell
    tolerance = max(float(path.kcl.dbu), 1e-6)
    if side == 'output':
        # Use exactly the original external input L, including its sampled
        # polygon orientation, to avoid changing it on the 1 nm GDS grid.
        outer.dmirror(p1=(0.0, 0.0), p2=(1.0, 1.0))
        outer.dmove((-float(outer.ports['o1'].dx), -float(outer.ports['o1'].dy)))
        middle.connect('o1', outer.ports['o2'])
        transition.connect('o1', middle.ports['o2'])
        _check_mzi_join(middle.ports['o1'], outer.ports['o2'], tolerance)
        _check_mzi_join(transition.ports['o1'], middle.ports['o2'], tolerance)
        path.add_port('o1', port=outer.ports['o1'])
        path.add_port('o2', port=transition.ports['o2'])
        external = 'o1'
        external_target = coupler.ports['o2']
    else:
        transition.dmove((-span/2, -span/2))
        middle.connect('o1', transition.ports['o2'])
        # Use the original external output L, rather than reflecting a
        # sampled input bend into an approximately equivalent output bend.
        if not horizontal_outputs:
            outer.dmirror(p1=(0.0, 0.0), p2=(1.0, 0.0))
        outer.dmove((float(middle.ports['o2'].dx-outer.ports['o1'].dx),
                     float(middle.ports['o2'].dy-outer.ports['o1'].dy)))
        _check_mzi_join(middle.ports['o1'], transition.ports['o2'], tolerance)
        _check_mzi_join(outer.ports['o1'], middle.ports['o2'], tolerance)
        path.add_port('o1', port=transition.ports['o1'])
        path.add_port('o2', port=outer.ports['o2'])
        external = 'o2'
        external_target = coupler.ports['o3']

    result = gf.Component()
    lower = result << path
    upper = result << path
    upper.dmirror(p1=(0.0, 0.0), p2=(1.0, 0.0))
    # Explicit anchoring handles mirror implementations that include a
    # bounding-box-dependent translation.
    output_shift = span/2 if horizontal_outputs else 0.0
    output_y_shift = span/2 if horizontal_outputs and output_bend_type=='l' else 0.0
    upper.dmove((float(external_target.dx+output_shift-upper.ports[external].dx),
                 float(external_target.dy+output_y_shift-upper.ports[external].dy)))
    for name, port in [('o1', lower.ports['o1']), ('o2', upper.ports['o1']),
                       ('o3', upper.ports['o2']), ('o4', lower.ports['o2'])]:
        result.add_port(name, port=port)
    result.info['arm_s_bend_length'] = float(s_cell.info['centerline_length'])
    result.info['arm_s_bend_minimum_radius_um'] = s_cell.info['minimum_radius_um']
    if horizontal_outputs:
        result.info['output_transition_type'] = 'L_PAIR' if output_bend_type=='l' else 'S_QUINTIC'
        result.info['output_transition_euler_bend_count'] = 2 if output_bend_type=='l' else 0
        result.info['output_transition_s_length_um'] = output_s_length
        result.info['output_transition_x_span_um'] = span
        result.info['output_transition_y_span_um'] = span if output_bend_type=='l' else span/2
    return result


@gf.cell
def mzi_equal_arms(
    bend_type: str = "s",
    wg_width: float = 1.0,
    coupler_gap: float = 0.8,
    q: float = 5.0,
    arm_length: float = 100.0,
    coupling_length: float = 80.0,
    hierarchical: bool = False,
    arm_bend_type: str = "l",
    horizontal_outputs: bool = False,
    output_bend_type: str = "s",
) -> gf.Component:
    """Connect two identical S or L couplers with equal straight arm sections.

    hierarchical=True keeps couplers, arms and bends as editable subcells.
    bend_type selects "s" or "l" (case-insensitive). The default preserves
    the original S layout. arm_length controls BOTH connecting straight
    sections; coupling_length controls the parallel straight region INSIDE
    each coupler. coupler_gap is the edge-to-edge gap inside each coupler.

    wg_width applies to all waveguides. q is passed unchanged to every custom
    Euler bend, including the extra arm turns for L. q changes bend size;
    it does not scale wg_width, coupler_gap, or either straight length.

    S: two straight arms directly connect the couplers.
    L: each arm has an additional L turn at each end (four added turns in
       total), so vertical coupler ports can connect to horizontal straights.
       arm_length excludes those turns; both complete arms remain equal.
       arm_bend_type="s" replaces each arm-side pair of L turns with one
       smooth quintic S bend, keeping all ports, coupling straights and arms
       in their original positions. The unmarked outer L turns are retained.

    External port positions: o1 lower-left, o2 upper-left, o3 upper-right,
    o4 lower-right. For S, left ports face west and right ports face east.
    For L, lower ports face south and upper ports face north.
    horizontal_outputs=True provides horizontal output fanouts, retaining
    the arms and coupling straights. output_bend_type="l" uses two full-size
    Euler L bends per output; "s" uses one smooth S. The L pair has twice
    the vertical offset of the smooth S at the same q. This option
    requires arm_bend_type="s" for an L MZI; native S MZIs already face east.
    """
    if not isinstance(bend_type, str) or bend_type.strip().lower() not in ("s", "l"):
        raise ValueError("bend_type must be 's' or 'l'")
    kind = bend_type.strip().lower()
    if not isinstance(arm_bend_type, str) or arm_bend_type.strip().lower() not in ('s', 'l'):
        raise ValueError("arm_bend_type must be 's' or 'l'")
    arm_kind = arm_bend_type.strip().lower()
    if not isinstance(output_bend_type,str) or output_bend_type.strip().lower() not in ('l','s'):
        raise ValueError('output_bend_type must be l or s')
    output_kind = output_bend_type.strip().lower()
    if horizontal_outputs and kind == 'l' and arm_kind != 's':
        raise ValueError('Horizontal L-MZI outputs require arm_bend_type="s"')
    for name, value in (
        ("arm_length", arm_length),
        ("coupling_length", coupling_length),
        ("wg_width", wg_width),
        ("q", q),
    ):
        if not isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not isfinite(coupler_gap) or coupler_gap < 0:
        raise ValueError("coupler_gap must be finite and non-negative")

    c = gf.Component()
    xs = waveguide_xs(wg_width)
    coupler = directional_coupler(
        bend_type=kind,
        straight_length=coupling_length,
        wg_width=wg_width,
        bend_q=q,
        gap=coupler_gap,
        hierarchical=hierarchical,
    )
    _check_bend_selection(coupler, kind, str(__file__))

    extra_bend_length = 0.0
    if kind == "s":
        dc1 = c << coupler
        dc2 = c << coupler
        # Orient the S couplers left-to-right, preserving upper/lower order.
        for dc in (dc1, dc2):
            if dc.ports["o3"].dx < dc.ports["o2"].dx:
                dc.dmirror(p1=(0.0, 0.0), p2=(0.0, 1.0))
    else:
        bend = euler_l_bend(
            q=q, width=wg_width, theta_deg=45.0, handedness="left",
        )
        arm_factory = (_l_coupler_with_s_arm_ports if arm_kind == 's'
                       else _l_coupler_with_arm_ports)
        first = arm_factory(coupler, bend, side="output")
        dc1 = c << first
        second = (arm_factory(coupler, bend, side="input", horizontal_outputs=True,
                              output_bend_type=output_kind)
                  if horizontal_outputs else arm_factory(coupler, bend, side="input"))
        dc2 = c << second
        if horizontal_outputs:
            for key,value in second.info.model_dump().items():
                if key.startswith('output_transition_'):
                    c.info[key] = value
            c.info['output_transition_length_um'] = (c.info['output_transition_euler_bend_count']*
                q*15*sqrt(2*pi)+c.info['output_transition_s_length_um'])
        if arm_kind == 's':
            # Each S crosses the old bare-coupler boundary at its midpoint.
            # The two half-S extensions together contribute one S length.
            extra_bend_length = float(first.info['arm_s_bend_length'])
            c.info['arm_s_bend_length'] = extra_bend_length
            c.info['arm_s_bend_minimum_radius_um'] = first.info['arm_s_bend_minimum_radius_um']
        else:
            extra_bend_length = 2.0 * float(bend.info["centerline_length"])

    # Both arms reference the SAME straight cell, guaranteeing equal lengths.
    straight = gf.components.straight(length=arm_length, cross_section=xs)
    upper_arm = c << straight
    lower_arm = c << straight
    upper_arm.connect("o1", dc1.ports["o3"])
    lower_arm.connect("o1", dc1.ports["o4"])

    # Translate dc2 to the lower arm, preserving its chosen reflection.
    # Its upper input must then coincide with the upper arm as well.
    dc2.dmove((
        float(lower_arm.ports["o2"].dx - dc2.ports["o1"].dx),
        float(lower_arm.ports["o2"].dy - dc2.ports["o1"].dy),
    ))

    tolerance = max(float(c.kcl.dbu), 1e-6)
    _check_mzi_join(upper_arm.ports["o1"], dc1.ports["o3"], tolerance)
    _check_mzi_join(lower_arm.ports["o1"], dc1.ports["o4"], tolerance)
    _check_mzi_join(dc2.ports["o1"], lower_arm.ports["o2"], tolerance)
    _check_mzi_join(dc2.ports["o2"], upper_arm.ports["o2"], tolerance)

    c.add_port("o1", port=dc1.ports["o1"])
    c.add_port("o2", port=dc1.ports["o2"])
    c.add_port("o3", port=dc2.ports["o3"])
    c.add_port("o4", port=dc2.ports["o4"])

    upper_length = hypot(
        float(upper_arm.ports["o2"].dx - upper_arm.ports["o1"].dx),
        float(upper_arm.ports["o2"].dy - upper_arm.ports["o1"].dy),
    )
    lower_length = hypot(
        float(lower_arm.ports["o2"].dx - lower_arm.ports["o1"].dx),
        float(lower_arm.ports["o2"].dy - lower_arm.ports["o1"].dy),
    )
    c.info["upper_arm_straight_length"] = upper_length
    c.info["lower_arm_straight_length"] = lower_length
    c.info['arm_straights'] = [
        {'arm':name, 'start_x_um':float(arm.ports['o1'].dx),
         'end_x_um':float(arm.ports['o2'].dx),
         'start_y_um':float(arm.ports['o1'].dy),
         'end_y_um':float(arm.ports['o2'].dy), 'length_um':length}
        for name,arm,length in [('upper',upper_arm,upper_length),('lower',lower_arm,lower_length)]]
    c.info['horizontal_outputs'] = horizontal_outputs
    c.info["arm_length_difference"] = upper_length - lower_length
    c.info["bend_type"] = kind.upper()
    c.info["requested_arm_straight_length"] = arm_length
    c.info["added_l_bends_per_arm"] = 2 if kind == "l" and arm_kind == 'l' else 0
    c.info['arm_transition_bend_type'] = ('S_QUINTIC' if arm_kind == 's' else 'L_PAIR') if kind == 'l' else 'S_EULER'
    c.info['arm_s_bends_per_arm'] = 2 if kind == 'l' and arm_kind == 's' else 0
    # Sampled centerline lengths between the bare couplers; excludes the
    # geometry already inside each bare coupler.
    c.info["upper_arm_total_length"] = upper_length + extra_bend_length
    c.info["lower_arm_total_length"] = lower_length + extra_bend_length
    c.info["coupling_length"] = coupling_length
    c.info["coupler_gap"] = coupler_gap
    c.info["waveguide_width"] = wg_width
    c.info["q"] = q
    if not hierarchical:
        c.flatten()
    _check_bend_selection(c, kind, __file__, horizontal_outputs=horizontal_outputs)
    return c
