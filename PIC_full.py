from __future__ import annotations

import argparse
import csv
import json
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from math import factorial, hypot, isfinite, pi, sqrt
from pathlib import Path

import gdsfactory as gf
import klayout.db as kdb
import numpy as np

from pic_components.technology import LAYERS
from pic_components.bends import euler_l_bend
from pic_components.mzi import mzi_equal_arms
from pic_components.edge_coupler import edge_coupler, linear_taper

# -------- MZI inputs: selected arm transitions use smooth S bends --------
BEND_TYPE = "l"                  # "l" or "s"; S receives vertical port adapters.
ARM_BEND_TYPE = "s"              # Internal MZI arm transitions.
OUTPUT_BEND_TYPE = "l"           # Last MZI outputs: "l" = two Euler Ls; "s" = smooth S.
WG_WIDTH = 1.0
COUPLER_GAP = 0.8
Q = 5.0
ARM_LENGTH = 400.0
COUPLING_LENGTH = 80.0
WIDTH_TAPER_LENGTH = 20.0
OUTPUT_LAYER = LAYERS.WG

# -------- Compact optical layout --------
ROUTING_Q = 5.0                  # Peripheral Euler bends; separate from MZI Q.
INPUT_X = 700.0                 # Lower splitter's left port column.
# USER INPUTS: the two short black horizontal sections in the marked picture.
# Lengths are tangent-to-tangent STRAIGHTS only, excluding both end bends.
SHORT_STAGE_STRAIGHT_LENGTH = 700  # H1: between SW1 and SW2 (um).
LONG_STAGE_STRAIGHT_LENGTH = 700   # H2: between SW2 and SW3 (um).
# The long bottom black straight is derived automatically; do not set it.
# USER INPUTS: the six black VERTICAL straight sections (um, excluding bends).
SPLITTER_BOTTOM_STRAIGHT_LENGTH = 75  # V1: splitter down to bottom-bus bend.
# V1 is shorter by 223.818 um after removing the combiner-entry step.
# This keeps the default MZI positions fixed while raising the bottom bus.
SPLITTER_TO_SW1_STRAIGHT_LENGTH = 600 # V2: splitter up to SW1, port to port.
SHORT_UPPER_VERTICAL_LENGTH = 2500   # V3: SW1 up to short-loop top bend.
SHORT_LOWER_VERTICAL_LENGTH = 300.0      # V4: SW1 down to short-loop bottom bend.
LONG_UPPER_VERTICAL_LENGTH = 3000    # V5: SW2 up to long-loop top bend.
LONG_LOWER_VERTICAL_LENGTH = 300.0       # V6: SW2 down to long-loop bottom bend.
# Loop heights and MZI row positions are derived from these lengths.
BOTTOM_BUS_Y = 900          # Reference output height before die centering.
EDGE_FREE_PLANE = 325.0        # Tip spans 325..350; taper spans 350..532.
CONNECT_EDGE_TIPS = True

# -------- MZI heaters and top pads --------
ARM_OVERLAY_LAYER = LAYERS.HEATER
ARM_OVERLAY_WIDTH = 7.0
CONTACT_PAD_LAYER = LAYERS.ROUTING
TOP_PAD_COUNT = 8               # Two per MZI, left-to-right.
CONTACT_PAD_SIZE = 350.0
CONTACT_PAD_PITCH = 575.0
CONTACT_PAD_CORNER_RADIUS = 100.0
TOP_PAD_BOTTOM_Y = 6680.0
ROUTE_HEATERS = True
ELECTRICAL_TRACE_WIDTH = 40.0
ELECTRICAL_TRACE_CLEARANCE = 20.0
ELECTRICAL_ROUTE_PITCH = 80.0
HEATER_CONTACT_TAPER_LENGTH = 40.0
HEATER_CONTACT_OVERLAP = 5.0

# -------- Die, trench and output --------
ADD_CHIP_LAYERS = True
CHIP_WIDTH = 8500.0
CHIP_HEIGHT = 7400.0
CHIP_SIZE_LAYER = LAYERS.CHIP_OUTLINE
CHIP_OUTLINE_WIDTH = 1.0
DEEP_TRENCH_LAYER = LAYERS.ANT_EDGE_TRENCH
DEEP_TRENCH_WIDTH = 350.0
OUTPUT_DIR = Path(__file__).resolve().parent / "gds_output"
SHOW_LAYOUT = True
SHOW_FLAT_GDS = True             # Display direct-edit polygons by default.
MAKE_PREVIEW = True
PRINT_WAVEGUIDE_LENGTHS = True
PRINT_PATH_LENGTHS = True        # L0-L4, in the order of the five marked pictures.

@dataclass(frozen=True)
class MZISettings:
    bend_type: str = "l"
    wg_width: float = 1.0
    coupler_gap: float = 0.8
    q: float = 5.0
    arm_length: float = 100.0
    coupling_length: float = 80.0
    arm_bend_type: str = "s"

@lru_cache(maxsize=16)
def _euler_quarter_profile(q: float, width: float, npoints: int):
    """Compute a 90-degree Euler polygon from q and width, without vertex data.

    For the first half, curvature grows linearly with arc length and
    x/y are integrals of cos(theta)/sin(theta) with ds=A/sqrt(2*theta) dtheta.
    Reflection gives the second half. The first tangent points east and the
    final tangent north. A=q*15 um, theta=0..pi/4, as in the custom bends.
    """
    if not isfinite(q) or q <= 0 or not isfinite(width) or width <= 0:
        raise ValueError("Peripheral bend q and width must be finite and positive")
    if npoints < 3:
        raise ValueError("Peripheral bends need at least three samples per half")
    theta = np.linspace(0.0, pi/4, npoints)
    scale = q * 15.0 / sqrt(2.0)
    x = np.zeros_like(theta)
    y = np.zeros_like(theta)
    # Twelve terms are sufficient at theta <= pi/4 to machine precision.
    for k in range(12):
        x += scale * (-1)**k * theta**(2*k+0.5) / ((2*k+0.5)*factorial(2*k))
        y += scale * (-1)**k * theta**(2*k+1.5) / ((2*k+1.5)*factorial(2*k+1))
    span = float(x[-1] + y[-1])
    xi = x - width/2 * np.sin(theta)
    yi = y + width/2 * np.cos(theta)
    xo = x + width/2 * np.sin(theta)
    yo = y - width/2 * np.cos(theta)
    inner = np.column_stack((np.r_[xi, span-yi[-2::-1]],
                              np.r_[yi, span-xi[-2::-1]]))
    outer = np.column_stack((np.r_[xo, span-yo[-2::-1]],
                              np.r_[yo, span-xo[-2::-1]]))
    return np.vstack((inner, outer[::-1])), span

def _port_point(port):
    return float(port.dx), float(port.dy)

def _check_join(a, b, dbu):
    if hypot(a.dx-b.dx, a.dy-b.dy) > dbu + 1e-9:
        raise ValueError("A new MZI adapter does not meet its port")
    if abs((a.orientation-b.orientation) % 360 - 180) > 1e-6:
        raise ValueError("A new MZI adapter faces the wrong direction")
    if abs(a.dwidth-b.dwidth) > dbu or a.layer != b.layer:
        raise ValueError("MZI adapter widths/layers do not match")

def _cell_name(name):
    """Keep readable, unique names within the traditional 32-character limit."""
    clean=re.sub(r'[^A-Za-z0-9_]', '_', name)
    return clean if len(clean)<=32 else clean[:25]+'_'+hashlib.sha1(clean.encode()).hexdigest()[:6]


def _named_shape(parent, name, layer, shape):
    """Place geometry in a separately editable child cell."""
    child=gf.Component(_cell_name(name))
    child.kdb_cell.shapes(gf.get_layer(layer)).insert(shape)
    parent << child
    return child


def _independent_component(source, name):
    """Clone every instance recursively so edits affect only that occurrence."""
    target=gf.Component(_cell_name(name))
    def copy_into(src,dst,prefix):
        dst.copy_shapes(src)
        for i,instance in enumerate(src.each_inst(),1):
            child_source=instance.cell
            original=child_source.name.lower()
            kind=next((label for token,label in (
                ('heater_upper','HEATER_UPPER'),('heater_lower','HEATER_LOWER'),
                ('heater','HEATER'),('core','OPTICAL'),('directional_coupler','DC'),('_dc','DC'),('straight','STRAIGHT'),
                ('bend','BEND'),('taper','TAPER')) if token in original),'PART')
            child_name=_cell_name(f'{prefix}_{kind}{i}')
            child=gf.Component(child_name)
            copy_into(child_source,child.kdb_cell,child_name)
            dst.insert(kdb.CellInstArray(child.cell_index(),instance.cplx_trans,
                                        instance.a,instance.b,instance.na,instance.nb))
    copy_into(source.kdb_cell,target.kdb_cell,name)
    target.add_ports(source.ports)
    target.info.update(source.info.model_dump())
    return target


def _core_on_layer(device, layer):
    """Copy the MZI core with its child cells, remapping only the optical layer."""
    c = _independent_component(device,'MZI_CORE_OUTPUT_TEMPLATE' if device.info.get('horizontal_outputs') else 'MZI_CORE_TEMPLATE')
    source_layer = device.ports["o1"].layer
    if any(p.layer != source_layer for p in device.ports):
        raise ValueError("MZI ports must all use one waveguide layer")
    destination=gf.get_layer(layer)
    if destination!=source_layer:
        def remap(cell):
            shapes=cell.shapes(source_layer)
            cell.shapes(destination).insert(shapes)
            shapes.clear()
            for instance in cell.each_inst(): remap(instance.cell)
        remap(c.kdb_cell)
        for p in c.ports: p.layer=destination
    return c

def _mzi_arm_straights(device):
    """Read arm endpoints if supplied, otherwise measure the MZI geometry.

    Clip the optical geometry to the central equal-arm span and require
    exactly two rectangular guides. Positions follow the built geometry.
    """
    info = device.info.model_dump()
    records = info.get("arm_straights")
    if records is not None:
        return [dict(record) for record in records]

    dbu = float(device.kcl.dbu)
    upper_length = float(info["upper_arm_straight_length"])
    lower_length = float(info["lower_arm_straight_length"])
    if upper_length <= 0 or abs(upper_length-lower_length) > 1e-8:
        raise ValueError("Layer-11 placement requires two equal positive MZI arm lengths")
    length_units = round(upper_length/dbu)
    # Identical input/output couplers put the straight span midway between
    # the two external port columns, for both the prepared L and S layouts.
    mid_x = (float(device.ports["o1"].dx)+float(device.ports["o4"].dx))/2
    left = round((mid_x-upper_length/2)/dbu)
    right = left+length_units
    source_layer = device.ports["o1"].layer
    core = kdb.Region(device.kdb_cell.begin_shapes_rec(source_layer))
    bounds = core.bbox()
    window = kdb.Region(kdb.Box(left, bounds.bottom-1, right, bounds.top+1))
    sections = list((core & window).merged().each())
    if len(sections) != 2:
        raise ValueError("Could not identify the two straight MZI arms for layer 11")
    sections.sort(key=lambda polygon: polygon.bbox().top, reverse=True)
    records = []
    width_units = round(float(device.ports["o1"].dwidth)/dbu)
    for name, polygon in zip(("upper", "lower"), sections):
        box = polygon.bbox()
        if (box.left != left or box.right != right
                or abs(box.height()-width_units) > 1
                or not (kdb.Region(polygon) ^ kdb.Region(box)).is_empty()):
            raise ValueError("The central MZI sections are not the expected straight arms")
        records.append({
            "arm": name, "start_x_um": left*dbu, "end_x_um": right*dbu,
            "start_y_um": (box.bottom+box.top)*dbu/2,
            "end_y_um": (box.bottom+box.top)*dbu/2,
            "length_um": length_units*dbu,
        })
    return records

def _add_mzi_arm_overlays(component, device, optical_layer, overlay_layer, width):
    """Place heater strips on the two actual MZI straight arms."""
    if overlay_layer is None:
        return []
    if gf.get_layer(overlay_layer) in (device.ports["o1"].layer, gf.get_layer(optical_layer)):
        raise ValueError("The arm overlay must use a separate layer from the waveguides")
    if not isfinite(width) or width <= 0:
        raise ValueError("arm_overlay_width must be finite and positive")
    dbu = float(component.kcl.dbu)
    half_width = width/2
    if abs(round(half_width/dbu)*dbu-half_width) > 1e-8:
        raise ValueError(f"arm_overlay_width must be a multiple of {2*dbu} um")
    arms = _mzi_arm_straights(device)
    if len(arms) != 2 or {arm["arm"] for arm in arms} != {"upper", "lower"}:
        raise ValueError("The MZI must supply one upper and one lower straight arm")
    for arm in arms:
        x1, y1 = arm["start_x_um"], arm["start_y_um"]
        x2, y2 = arm["end_x_um"], arm["end_y_um"]
        length = hypot(x2-x1, y2-y1)
        if length <= 0 or abs(length-arm["length_um"]) > 1e-8:
            raise ValueError("MZI arm endpoints and length do not agree")
        nx, ny = -(y2-y1)/length*half_width, (x2-x1)/length*half_width
        heater=gf.Component(('OUT_ARM_HEATER_' if device.info.get('horizontal_outputs') else 'ARM_HEATER_')+arm['arm'].upper())
        heater.add_polygon([
            (x1+nx, y1+ny), (x2+nx, y2+ny),
            (x2-nx, y2-ny), (x1-nx, y1-ny),
        ], layer=overlay_layer)
        component << heater
    return arms

def _mzi_interface(settings, layer, taper_length,
                   arm_overlay_layer=LAYERS.HEATER, arm_overlay_width=7.0,
                   horizontal_outputs=False):
    """Expose vertical interfaces, or east-facing outputs for the last switch."""
    device = mzi_equal_arms(
        bend_type=settings.bend_type, wg_width=settings.wg_width,
        coupler_gap=settings.coupler_gap, q=settings.q,
        arm_length=settings.arm_length, coupling_length=settings.coupling_length,
        hierarchical=True,
        arm_bend_type=settings.arm_bend_type,
        horizontal_outputs=horizontal_outputs,
        output_bend_type=OUTPUT_BEND_TYPE,
    )
    c = gf.Component()
    ref = c << _core_on_layer(device, layer)
    arm_straights = _add_mzi_arm_overlays(
        c, device, layer, arm_overlay_layer, arm_overlay_width,
    )
    ports = {name: ref.ports[name] for name in ("o1","o2","o3","o4")}
    kind = settings.bend_type.strip().lower()
    interface_taper_length = 0.0
    if kind == "s":
        # The reference has vertical terminals. Add four actual Euler L
        # adapters outside the S-MZI, preserving the S directional couplers.
        bend = euler_l_bend(q=settings.q, width=settings.wg_width, layer=layer)
        for name, attached, free in (("o1","o1","o2"),("o2","o2","o1"),
                                     ("o3","o1","o2"),("o4","o2","o1")):
            if horizontal_outputs and name in ('o3','o4'):
                continue
            turn = c << bend
            turn.connect(attached, ports[name])
            _check_join(turn.ports[attached], ports[name], c.kcl.dbu)
            ports[name] = turn.ports[free]

    if abs(settings.wg_width - 1.0) > float(c.kcl.dbu) / 2:
        if not isfinite(taper_length) or taper_length <= 0:
            raise ValueError("width_taper_length must be positive")
        taper = linear_taper(taper_length, settings.wg_width, 1.0, layer)
        interface_taper_length = hypot(
            float(taper.ports["o2"].dx - taper.ports["o1"].dx),
            float(taper.ports["o2"].dy - taper.ports["o1"].dy),
        )
        for name, port in ports.copy().items():
            transition = c << taper
            transition.connect("o1", port)
            _check_join(transition.ports["o1"], port, c.kcl.dbu)
            ports[name] = transition.ports["o2"]

    for name, angle in (("o1",270),("o2",90),("o3",90),("o4",270)):
        if horizontal_outputs and name in ('o3','o4'):
            angle = 0
        if abs((ports[name].orientation-angle+180) % 360-180) > 1e-6:
            raise ValueError(f"{name} has the wrong reference-interface orientation")
        c.add_port(name, port=ports[name])
    if abs(c.ports["o1"].dx-c.ports["o2"].dx) > c.kcl.dbu:
        raise ValueError("The two left MZI interfaces do not align")
    if abs(c.ports["o3"].dx-c.ports["o4"].dx) > c.kcl.dbu:
        raise ValueError("The two right MZI interfaces do not align")
    c.info.update(device.info.model_dump())
    c.info["arm_straights"] = arm_straights
    c.info["arm_overlay_layer"] = list(arm_overlay_layer) if arm_overlay_layer is not None else None
    c.info["arm_overlay_width_um"] = arm_overlay_width if arm_overlay_layer is not None else 0.0
    c.info["arm_overlay_count"] = len(arm_straights)
    # Nominal length of ONE full selected MZI path, from external interface
    # to external interface. Keep MZI q separate from the fixed routing bends.
    quarter_length = settings.q * 15.0 * sqrt(2*pi)
    smooth_length = 0.0
    if kind == "l":
        if settings.arm_bend_type.strip().lower() == 's':
            euler_length = 2.0 * quarter_length
            smooth_length = 2.0 * float(device.info['arm_s_bend_length'])
        else:
            euler_length = 6.0 * quarter_length
    else:
        # Four S bends (theta=22.5 deg per Euler segment) + two L adapters.
        s_bend_length = 2.0 * settings.q * 15.0 * sqrt(pi)
        euler_length = 4.0 * s_bend_length + 2.0 * quarter_length
    if horizontal_outputs:
        euler_length -= quarter_length
        if kind == 'l':
            euler_length += int(device.info['output_transition_euler_bend_count'])*quarter_length
            smooth_length += float(device.info['output_transition_s_length_um'])
    coupling_length = round(settings.coupling_length / c.kcl.dbu) * c.kcl.dbu
    c.info["nominal_path_length_parts"] = {
        "straight_um": float(device.info["upper_arm_straight_length"]) + 2.0 * coupling_length,
        "euler_um": euler_length,
        "smooth_s_um": smooth_length,
        "taper_um": 2.0 * interface_taper_length,
    }
    # The marked paths start at the downstream end of the splitter output
    # coupling straight, before its outer bend and optional port adapter/taper.
    c.info['output_coupler_to_interface_length_parts'] = {
        'euler_um': (int(device.info['output_transition_euler_bend_count'])*quarter_length if kind=='l' else s_bend_length) if horizontal_outputs else
                    (quarter_length if kind=='l' else quarter_length+s_bend_length),
        'smooth_s_um': float(device.info['output_transition_s_length_um']) if horizontal_outputs and kind=='l' else 0.0,
        'taper_um': interface_taper_length,
    }
    return c

def _network_groups(region, probes, dbu):
    """Return which optical edge guides share the same continuous polygon."""
    polygons = list(region.merged().each())
    groups = {}
    for name, point in probes.items():
        p = kdb.Point(round(point[0]/dbu), round(point[1]/dbu))
        hits = [i for i,poly in enumerate(polygons) if poly.inside(p)]
        if len(hits) != 1:
            raise ValueError(f"Reference routing continuity failed at {name}")
        groups.setdefault(hits[0], []).append(name)
    return sorted(tuple(sorted(names)) for names in groups.values())

def point(port):
    return (float(port.dx), float(port.dy))


def region(component, layer):
    return kdb.Region(component.kdb_cell.begin_shapes_rec(gf.get_layer(layer))).merged()


def box_region(bounds, dbu):
    return kdb.Region(kdb.DBox(*bounds).to_itype(dbu))


class OpticalRouter:
    """Explicit Manhattan centerlines with the supplied Euler quarter profile.

    Every corner consumes one bend span on each neighboring segment. Routes
    that are too short fail instead of silently shrinking the bend radius.
    """
    def __init__(self, component, layer, q):
        self.c, self.layer, self.q = component, layer, q
        self.dbu = float(component.kcl.dbu)
        profile, true_span = _euler_quarter_profile(q, 1.0, 50)
        self.span = round(true_span / self.dbu) * self.dbu
        # Only align the numerical endpoint to the 1 nm GDS grid (<0.5 nm).
        self.profile = profile / true_span * self.span
        self.bend_length = q * 15.0 * sqrt(2*pi) * self.span / true_span
        self.records = []

    def route(self, name, waypoints, start_port=None, end_port=None):
        p = np.rint(np.asarray(waypoints, dtype=float)/self.dbu)*self.dbu
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
            raise ValueError(f'{name}: expected finite Nx2 waypoints')
        # Remove redundant collinear internal waypoints.
        clean = [p[0]]
        for i in range(1, len(p)-1):
            a, b = p[i]-clean[-1], p[i+1]-p[i]
            if abs(a[0]*b[1]-a[1]*b[0]) < 1e-8 and np.dot(a,b) > 0:
                continue
            clean.append(p[i])
        clean.append(p[-1])
        p = np.array(clean)
        lengths = np.linalg.norm(np.diff(p,axis=0),axis=1)
        if np.any(lengths < self.dbu/2):
            raise ValueError(f'{name}: duplicate waypoints')
        directions = np.diff(p,axis=0)/lengths[:,None]
        if any(np.count_nonzero(np.abs(u)>1e-8) != 1 for u in directions):
            raise ValueError(f'{name}: only axis-aligned routing is supported')
        for a,b in zip(directions,directions[1:]):
            if abs(np.dot(a,b)) > 1e-8:
                raise ValueError(f'{name}: corners must turn 90 degrees')
        for port, endpoint, outward in ((start_port,p[0],directions[0]),
                                        (end_port,p[-1],-directions[-1])):
            if port is not None:
                expected = np.array([np.cos(np.deg2rad(port.orientation)),
                                     np.sin(np.deg2rad(port.orientation))])
                if np.linalg.norm(np.asarray(point(port))-endpoint) > self.dbu*1.01:
                    raise ValueError(f'{name}: endpoint is not at its component port')
                if np.linalg.norm(outward-expected) > 1e-7:
                    raise ValueError(f'{name}: route approaches its port backwards')
                if abs(port.dwidth-1.0)>self.dbu or port.layer!=gf.get_layer(self.layer):
                    raise ValueError(f'{name}: port width or layer mismatch')
        r = kdb.Region()
        pieces=[]
        for i,(a,b,u,length) in enumerate(zip(p,p[1:],directions,lengths)):
            take_start = self.span if i else 0.0
            take_end = self.span if i < len(lengths)-1 else 0.0
            available = length-take_start-take_end
            if available < -self.dbu*1.01:
                raise ValueError(f'{name}: segment {i} is {abs(available):.3f} um too short for Euler bends')
            if available > self.dbu/2:
                a, b = a+take_start*u, b-take_end*u
                n = np.array([-u[1],u[0]])*0.5
                poly = np.array([a+n,b+n,b-n,a-n])
                shape=kdb.Polygon([kdb.Point(int(x),int(y)) for x,y in np.rint(poly/self.dbu)])
                r.insert(shape)
                pieces.append((f'STRAIGHT_{i+1:02d}',shape))
        for i in range(1,len(p)-1):
            u,v = directions[i-1],directions[i]
            start = p[i]-self.span*u
            xy = start+self.profile[:,0,None]*u+self.profile[:,1,None]*v
            shape=kdb.Polygon([kdb.Point(int(x),int(y)) for x,y in np.rint(xy/self.dbu)])
            r.insert(shape)
            pieces.append((f'BEND_{i:02d}',shape))
        r.merge()
        if r.count()!=1 or any(p.holes() for p in r.each()):
            raise ValueError(f'{name}: disconnected or self-intersecting route polygons')
        route_cell=gf.Component(_cell_name('ROUTE_'+name.upper()))
        part_prefix=route_cell.name.replace('ROUTE_','R_').replace('_STAGE','').replace('_REFERENCE','')
        for label,shape in pieces:
            _named_shape(route_cell,part_prefix+'_'+label,self.layer,shape)
        self.c << route_cell
        length = float(sum(lengths)+(len(p)-2)*(self.bend_length-2*self.span))
        self.records.append({'name':name,'length_um':length,'bends':len(p)-2,
                             'waypoints_um':p.tolist()})
        return length


def _edge(component, name, coordinate, side='left'):
    """Place one edge coupler normal to the left or right trench wall."""
    template=edge_coupler(connect_tip=CONNECT_EDGE_TIPS,layer=OUTPUT_LAYER,hierarchical=True)
    ec = component << _independent_component(template,'EC_'+name.upper())
    if side=='right':
        free_x=CHIP_WIDTH-DEEP_TRENCH_WIDTH+25.0
        ec.drotate(180)
        ec.dmove((free_x,coordinate))
        endpoint=(free_x-207.0,coordinate)
        tip=(free_x-25.0,coordinate-0.175,free_x,coordinate+0.175)
    elif side=='left':
        ec.dmove((EDGE_FREE_PLANE,coordinate))
        endpoint=(EDGE_FREE_PLANE+207.0,coordinate)
        tip=(EDGE_FREE_PLANE,coordinate-0.175,EDGE_FREE_PLANE+25.0,coordinate+0.175)
    else:
        raise ValueError('Edge side must be left or right')
    component.add_port(name,port=ec.ports['o1'])
    if CONNECT_EDGE_TIPS and hypot(ec.ports['o2'].dx-endpoint[0],ec.ports['o2'].dy-endpoint[1])>component.kcl.dbu:
        raise ValueError(f'{name}: edge taper does not meet its routing endpoint')
    return endpoint,tip,ec.ports['o2'] if CONNECT_EDGE_TIPS else None


def add_pads(c, heaters):
    if TOP_PAD_COUNT != 8:
        raise ValueError('Four MZIs require eight top pads')
    if CONTACT_PAD_PITCH <= CONTACT_PAD_SIZE or not 0<=CONTACT_PAD_CORNER_RADIUS<=CONTACT_PAD_SIZE/2:
        raise ValueError('Invalid pad pitch/size/corner radius')
    dbu=float(c.kcl.dbu)
    size=round(CONTACT_PAD_SIZE/dbu)
    poly=kdb.Polygon(kdb.Box(0,0,size,size)).round_corners(0,round(CONTACT_PAD_CORNER_RADIUS/dbu),64)
    # Align the pad array to the actual heater/arm centers. Output fanout
    # length must not move existing pads or their electrical connections.
    centers=[(h['bounds_um'][0]+h['bounds_um'][2])/2 for h in heaters if h['name'].endswith('.upper')]
    mid=(centers[0]+centers[-1])/2
    first=round((mid-(TOP_PAD_COUNT-1)*CONTACT_PAD_PITCH/2)/dbu)*dbu
    pads=[]
    def add(name,cx,cy,side):
        _named_shape(c,'PAD_'+name,CONTACT_PAD_LAYER,poly.transformed(
            kdb.Trans(round((cx-CONTACT_PAD_SIZE/2)/dbu),round((cy-CONTACT_PAD_SIZE/2)/dbu))))
        pads.append({'name':name,'center_x_um':cx,'center_y_um':cy,
                     'side':side,'size_um':CONTACT_PAD_SIZE,'connection':'spare'})
    for i in range(TOP_PAD_COUNT):
        add(f'TOP_{i+1:02d}',first+i*CONTACT_PAD_PITCH,TOP_PAD_BOTTOM_Y+CONTACT_PAD_SIZE/2,'top')
    return pads


def add_electrical_routes(c,pads,heaters):
    """Eight top-pad connections for the four MZI upper-arm heaters.

    Only the upper arm of each MZI is powered, matching the original script.
    Each terminal has a separate pad; there is no common metal return.
    """
    if not ROUTE_HEATERS:
        return []
    dbu=float(c.kcl.dbu)
    w=ELECTRICAL_TRACE_WIDTH; gap=ELECTRICAL_TRACE_CLEARANCE
    taper=HEATER_CONTACT_TAPER_LENGTH; overlap=HEATER_CONTACT_OVERLAP
    pitch=ELECTRICAL_ROUTE_PITCH
    if min(w,gap,taper,overlap,pitch)<=0 or pitch < w+gap:
        raise ValueError('Invalid electrical widths/clearance/pitch')
    layer=gf.get_layer(CONTACT_PAD_LAYER)
    pad_shapes=list(region(c,CONTACT_PAD_LAYER).each())
    boxes={h['name']:box_region(h['bounds_um'],dbu) for h in heaters}
    selected=[next(h for h in heaters if h['name']==f'MZI_{i}.upper') for i in range(1,5)]
    terminal_specs=[]
    for i,h in enumerate(selected):
        for side in ('left','right'):
            terminal_specs.append((pads[2*i+(side=='right')],h,side))
    routes=[]; nets=[]; trace_shapes=[]
    for i,(pad,h,side) in enumerate(terminal_specs):
        x1,y1,x2,y2=h['bounds_um']; cx=(x1+x2)/2; cy=(y1+y2)/2
        if x2-x1<=2*overlap+gap:
            raise ValueError(f"{h['name']}: insufficient unmetalized heater length")
        px,py=pad['center_x_um'],pad['center_y_um']
        left=side=='left'; edge=x1 if left else x2; tip=edge+(overlap if left else -overlap)
        wide=edge+(-taper if left else taper)
        lane=edge+(-(taper+w) if left else taper+w)
        rank=i if lane<=px else 7-i
        fan=TOP_PAD_BOTTOM_Y-pitch*(rank+1)
        if fan<=cy+pitch:
            raise ValueError('Raise top pads or lower switches to fit heater fanout')
        points=[(px,TOP_PAD_BOTTOM_Y+w/2),(px,fan),(lane,fan),(lane,cy),(wide,cy)]
        contact=[(wide,cy-w/2),(edge,y1),(tip,y1),(tip,y2),(edge,y2),(wide,cy+w/2)]
        intended=box_region((min(edge,tip),y1,max(edge,tip),y2),dbu)
        path=kdb.Region(kdb.Path([kdb.Point(round(x/dbu),round(y/dbu)) for x,y in points],round(w/dbu)).polygon())
        poly=kdb.Region(kdb.Polygon([kdb.Point(round(x/dbu),round(y/dbu)) for x,y in contact]))
        center=kdb.Point(round(px/dbu),round(py/dbu))
        matches=[p for p in pad_shapes if p.inside(center)]
        if len(matches)!=1: raise ValueError('Cannot identify electrical pad')
        net=(path+poly+kdb.Region(matches[0])).merged()
        if net.count()!=1 or any(p.holes() for p in net.each()):
            raise ValueError(f"Disconnected or self-intersecting net {pad['name']}")
        for name,heater in boxes.items():
            actual=net&heater
            if name==h['name']:
                if not (actual^intended).is_empty(): raise ValueError(f'{name}: wrong contact extent')
            elif not actual.is_empty(): raise ValueError(f"{pad['name']} contacts unintended heater {name}")
        for other_name,other in nets:
            if not (net.sized(round(gap/dbu))&other).is_empty():
                raise ValueError(f"Electrical clearance violation: {pad['name']} and {other_name}")
        for p in pad_shapes:
            if not p.inside(center) and not (net.sized(round(gap/dbu))&kdb.Region(p)).is_empty():
                raise ValueError(f"{pad['name']} touches an unrelated pad")
        nets.append((pad['name'],net))
        trace_shapes.append((pad['name'],(path+poly).merged()))
        routes.append({'pad':pad['name'],'heater':h['name'],'terminal':side,
                       'centerline_um':points,'contact_overlap_um':overlap})
        pad['connection']=f"{h['name']}.{side}"
    # Insert only the trace/contact geometry here. The pad already has its
    # own cell; duplicating it in the trace would prevent independent edits.
    for name,metal in trace_shapes: _named_shape(c,'TRACE_'+name,CONTACT_PAD_LAYER,metal)
    if region(c,CONTACT_PAD_LAYER).count()!=len(pads):
        raise ValueError('Pad nets merged unexpectedly')
    return routes


def add_die(c, tips):
    if not ADD_CHIP_LAYERS: return
    dbu=float(c.kcl.dbu)
    t=DEEP_TRENCH_WIDTH
    # Align the left/bottom inner trench edges to every tip/taper junction.
    inner=EDGE_FREE_PLANE+25.0
    retained=box_region((inner,inner,CHIP_WIDTH-t,CHIP_HEIGHT-t),dbu)
    tipregion=kdb.Region()
    for tip in tips: tipregion+=box_region(tip,dbu)
    layers=(OUTPUT_LAYER,ARM_OVERLAY_LAYER,CONTACT_PAD_LAYER)
    for layer in set(layers):
        physical=region(c,layer)
        if layer==OUTPUT_LAYER: physical-=tipregion
        outside=physical-retained
        if not outside.is_empty():
            raise ValueError(f'Layer {layer} extends outside retained die interior: {outside.bbox().to_dtype(dbu)}; increase CHIP_WIDTH/HEIGHT or adjust layout')
    outer=box_region((inner-t,inner-t,CHIP_WIDTH,CHIP_HEIGHT),dbu)
    trench=outer-retained
    if not ((trench&region(c,OUTPUT_LAYER))^tipregion).is_empty():
        raise ValueError('Trench must intersect only the 25 um edge tips')
    _named_shape(c,'DEEP_TRENCH',DEEP_TRENCH_LAYER,trench)
    border=box_region((0,0,CHIP_WIDTH,CHIP_HEIGHT),dbu)-box_region((CHIP_OUTLINE_WIDTH,CHIP_OUTLINE_WIDTH,
                        CHIP_WIDTH-CHIP_OUTLINE_WIDTH,CHIP_HEIGHT-CHIP_OUTLINE_WIDTH),dbu)
    _named_shape(c,'CHIP_OUTLINE',CHIP_SIZE_LAYER,border)


def reported_path_lengths(report):
    """L0-L4 follow the five annotated paths, including their curved sections.

    Start at the splitter output coupling-straight EXIT. End at the wide
    input of the right-edge taper: reference output for L0, lower SW3 output
    for L1-L4. The former bottom combiner has been removed. Tapers and edge
    tips are excluded. L1-L4 each traverse one equal-length arm in
    each of SW1, SW2, SW3, including both coupling straights of every switch.
    All values are nominal geometric centerline lengths, not n_eff*L.
    """
    routes={item['name']:item for item in report['routing']}
    mzi_parts=report['mzi_geometry']['nominal_path_length_parts']
    last_mzi_parts=report['last_mzi_geometry']['nominal_path_length_parts']
    lead_parts=report['mzi_geometry']['output_coupler_to_interface_length_parts']
    mzi_length=sum(mzi_parts.values())
    lead_length=sum(lead_parts.values())
    def term(name,length,kind):
        if not isfinite(length) or length<0:
            raise ValueError(f'Invalid path-length contribution: {name}')
        return {'name':name,'kind':kind,'length_um':float(length)}
    def route(name): return term(name,routes[name]['length_um'],'routed_waveguide')
    def switch(index):
        return term(f'SW{index}_complete_MZI',sum(last_mzi_parts.values()) if index==3 else mzi_length,'mzi_traversal')
    def splitter_lead(branch): return term(f'splitter_{branch}_output_lead',lead_length,'coupler_exit_to_port')
    paths={}
    def add(label,description,short_choice,long_choice,parts):
        total=sum(part['length_um'] for part in parts)
        paths[label]={'description':description,'short_stage':short_choice,
                      'long_stage':long_choice,'length_um':total,'length_mm':total/1000,
                      'components':parts}
    add('L0','Bottom reference path','bypass','bypass',[
        splitter_lead('lower'),route('bottom_reference_bus'),
    ])
    for label,short_choice,long_choice in [('L1','lower','lower'),('L2','upper','lower'),
                                          ('L3','lower','upper'),('L4','upper','upper')]:
        add(label,f'{short_choice.capitalize()} short-stage / {long_choice} long-stage route',
            short_choice,long_choice,[
                splitter_lead('upper'),route('splitter_to_switch_1'),switch(1),
                route(f'short_stage_{short_choice}'),switch(2),
                route(f'long_stage_{long_choice}'),switch(3),route('switch_3_lower_output'),
            ])
    for item in paths.values():
        item['difference_from_L0_um']=item['length_um']-paths['L0']['length_um']
    short_delta=report['stage_lengths_um']['short_difference']
    long_delta=report['stage_lengths_um']['long_difference']
    values={label:item['length_um'] for label,item in paths.items()}
    for measured,expected in [(values['L2']-values['L1'],short_delta),
                              (values['L3']-values['L1'],long_delta),
                              (values['L4']-values['L1'],short_delta+long_delta)]:
        if abs(measured-expected)>1e-8:
            raise ValueError('L0-L4 report is inconsistent with the selected stage routes')
    return {
        'basis':'Nominal geometric waveguide centerline length; includes straight and curved sections.',
        'reference_planes':{
            'start':'Downstream end of splitter output coupling straight (lower branch for L0; upper for L1-L4).',
            'end':'Wide input of right-edge taper: edge_174 (reference) for L0; edge_161 (lower SW3 output) for L1-L4. The upper SW3 output edge_186 has equal geometric length to edge_161.',
        },
        'excluded':'Splitter internal arms/output coupling straight; all edge-coupler tapers and tips. No bottom combiner remains.',
        'endpoint_revision':'Output reference planes changed after deleting the bottom combiner; totals are not directly comparable with reports ending at its coupling entrance.',
        'method':'Analytic Euler arclength, quadrature for fitted S bends, and grid-snapped straight dimensions. This is not refractive-index-weighted optical path length or group delay.',
        'mzi_note':('Each switch contributes one arm and both coupling straights. SW3 output transition: '+
            report['last_mzi_geometry'].get('output_transition_type','S_EULER')+
            '. The two arms have equal length; no transverse distance is assigned to directional-coupler transfer.'),
        'splitter_output_lead_parts_um':lead_parts,
        'one_switch_mzi_parts_um':mzi_parts,
        'last_switch_mzi_parts_um':last_mzi_parts,
        'paths':paths,
    }


def write_path_length_reports(out,report):
    """Write a compact CSV and a readable report with definitions/breakdowns."""
    data=report['reported_path_lengths'];paths=data['paths']
    with (out/'path_lengths_L0_L4.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.writer(handle)
        writer.writerow(['path','short_stage','long_stage','length_um','length_mm','difference_from_L0_um'])
        for label,item in paths.items():
            writer.writerow([label,item['short_stage'],item['long_stage'],
                             f"{item['length_um']:.6f}",f"{item['length_mm']:.9f}",
                             f"{item['difference_from_L0_um']:.6f}"])
    lines=['# L0-L4 waveguide lengths','',data['basis'],
           'Labels follow the five supplied pictures in order. Values recalculate on every run.','',
           '| Path | Short stage | Long stage | Length (µm) | Length (mm) |',
           '|---|---|---|---:|---:|']
    for label,item in paths.items():
        lines.append(f"| {label} | {item['short_stage']} | {item['long_stage']} | {item['length_um']:.3f} | {item['length_mm']:.6f} |")
    lines+=['','## Reference planes','',
            'Start: '+data['reference_planes']['start'],
            '', 'End: '+data['reference_planes']['end'],
            '', 'Excluded: '+data['excluded'],
            '',data['endpoint_revision'],'',data['method'],'',data['mzi_note'],
            '', '## Component contributions', '']
    for label,item in paths.items():
        lines += [f'### {label}: {item["description"]}','',
                  '| Component | Length (µm) |','|---|---:|']
        lines += [f"| {part['name']} | {part['length_um']:.3f} |" for part in item['components']]
        lines += [f"| **Total** | **{item['length_um']:.3f}** |",'']
    (out/'path_lengths_L0_L4.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def rebuild_chip(settings=None, *, short_straight_length=None, long_straight_length=None,
                 splitter_bottom_length=None, splitter_to_sw1_length=None,
                 short_upper_vertical_length=None, short_lower_vertical_length=None,
                 long_upper_vertical_length=None, long_lower_vertical_length=None):
    """Build the connected layout from user-selected horizontal/vertical straights.

    None uses the editable main inputs. Values are in um, exclude the bend
    footprints, and are snapped to the construction grid. Downstream MZIs and
    the output access lengths follow the derived spacing. The bottom bus ends
    at the right edge taper; its horizontal length is calculated automatically.
    """
    if settings is None:
        settings=MZISettings(BEND_TYPE,WG_WIDTH,COUPLER_GAP,Q,ARM_LENGTH,COUPLING_LENGTH,ARM_BEND_TYPE)
    if settings.coupler_gap<=0: raise ValueError('Positive coupler gap required')
    vertical_inputs={
        'splitter_bottom':('V1','SPLITTER_BOTTOM_STRAIGHT_LENGTH',
            SPLITTER_BOTTOM_STRAIGHT_LENGTH if splitter_bottom_length is None else splitter_bottom_length,
            'bottom_reference_bus'),
        'splitter_to_sw1':('V2','SPLITTER_TO_SW1_STRAIGHT_LENGTH',
            SPLITTER_TO_SW1_STRAIGHT_LENGTH if splitter_to_sw1_length is None else splitter_to_sw1_length,
            'splitter_to_switch_1'),
        'short_upper':('V3','SHORT_UPPER_VERTICAL_LENGTH',
            SHORT_UPPER_VERTICAL_LENGTH if short_upper_vertical_length is None else short_upper_vertical_length,
            'short_stage_upper'),
        'short_lower':('V4','SHORT_LOWER_VERTICAL_LENGTH',
            SHORT_LOWER_VERTICAL_LENGTH if short_lower_vertical_length is None else short_lower_vertical_length,
            'short_stage_lower'),
        'long_upper':('V5','LONG_UPPER_VERTICAL_LENGTH',
            LONG_UPPER_VERTICAL_LENGTH if long_upper_vertical_length is None else long_upper_vertical_length,
            'long_stage_upper'),
        'long_lower':('V6','LONG_LOWER_VERTICAL_LENGTH',
            LONG_LOWER_VERTICAL_LENGTH if long_lower_vertical_length is None else long_lower_vertical_length,
            'long_stage_lower'),
    }
    for label,name,value,route in vertical_inputs.values():
        if not isfinite(value) or value<=0:
            raise ValueError(f'{name} ({label}) must be finite and positive (um)')
    requested_short=SHORT_STAGE_STRAIGHT_LENGTH if short_straight_length is None else short_straight_length
    requested_long=LONG_STAGE_STRAIGHT_LENGTH if long_straight_length is None else long_straight_length
    for name,value in [('SHORT_STAGE_STRAIGHT_LENGTH',requested_short),
                       ('LONG_STAGE_STRAIGHT_LENGTH',requested_long)]:
        if not isfinite(value) or value<0:
            raise ValueError(f'{name} must be finite and non-negative (um)')
    c=gf.Component('compact_four_mzi_pic')
    dbu=float(c.kcl.dbu)
    if abs(dbu-0.001)>1e-12: raise ValueError('Use a PDK with 0.001 um construction grid')
    vertical={name:round(values[2]/dbu)*dbu for name,values in vertical_inputs.items()}
    for name,value in vertical.items():
        if value<dbu:
            raise ValueError(f'{vertical_inputs[name][1]} must round to at least {dbu:g} um')
    core=_mzi_interface(settings,OUTPUT_LAYER,WIDTH_TAPER_LENGTH,ARM_OVERLAY_LAYER,ARM_OVERLAY_WIDTH)
    output_core=_mzi_interface(settings,OUTPUT_LAYER,WIDTH_TAPER_LENGTH,
                               ARM_OVERLAY_LAYER,ARM_OVERLAY_WIDTH,horizontal_outputs=True)
    width=float(core.ports['o4'].dx-core.ports['o1'].dx)
    height=float(core.ports['o2'].dy-core.ports['o1'].dy)
    router=OpticalRouter(c,OUTPUT_LAYER,ROUTING_Q); d=router.span
    if not isfinite(BOTTOM_BUS_Y): raise ValueError('BOTTOM_BUS_Y must be finite')
    bottom_bus=round(BOTTOM_BUS_Y/dbu)*dbu
    row1=bottom_bus+d+vertical['splitter_bottom']
    row2=row1+height+vertical['splitter_to_sw1']
    row_top=row2+height
    short_top=row_top+d+vertical['short_upper']
    long_top=row_top+d+vertical['long_upper']
    short_straight=round(requested_short/dbu)*dbu
    long_straight=round(requested_long/dbu)*dbu
    # Each horizontal link also needs one bend footprint at each end.
    short_gap=short_straight+2*d
    long_gap=long_straight+2*d
    xs=[INPUT_X,INPUT_X+width,INPUT_X+2*width+short_gap,
        INPUT_X+3*width+short_gap+long_gap]
    switches=[]; heaters=[]; placements=[]
    for i,(x,y) in enumerate(zip(xs,[row1,row2,row2,row2]),1):
        cell_name=f'MZI_{i}_SPLITTER' if i==1 else f'MZI_{i}_SW{i-1}'
        selected_core=output_core if i==4 else core
        ref=c<<_independent_component(selected_core,cell_name)
        ref.dmove((x-selected_core.ports['o1'].dx,y-selected_core.ports['o1'].dy))
        switches.append(ref)
        placements.append({'name':f'MZI_{i}','role':'splitter' if i==1 else f'switch_{i-1}',
                           'ports_um':{p.name:point(p) for p in ref.ports}})
        dx=ref.ports['o1'].dx-selected_core.ports['o1'].dx; dy=ref.ports['o1'].dy-selected_core.ports['o1'].dy
        for arm in selected_core.info['arm_straights']:
            x1=arm['start_x_um']+dx; x2=arm['end_x_um']+dx; y0=arm['start_y_um']+dy
            heaters.append({'name':f"MZI_{i}.{arm['arm']}",'length_um':arm['length_um'],
                            'bounds_um':[x1,y0-ARM_OVERLAY_WIDTH/2,x2,y0+ARM_OVERLAY_WIDTH/2]})
    a,b,m,z=switches
    short_low=row2-d-vertical['short_lower']
    long_low=row2-d-vertical['long_lower']
    def P(ref,name): return point(ref.ports[name])
    def R(name,pts,start=None,end=None): return router.route(name,pts,start,end)
    R('splitter_to_switch_1',[P(a,'o3'),P(b,'o1')],a.ports['o3'],b.ports['o1'])
    R('short_stage_upper',[P(b,'o3'),(b.ports['o3'].dx,short_top),
                           (m.ports['o2'].dx,short_top),P(m,'o2')],b.ports['o3'],m.ports['o2'])
    R('short_stage_lower',[P(b,'o4'),(b.ports['o4'].dx,short_low),
                           (m.ports['o1'].dx,short_low),P(m,'o1')],b.ports['o4'],m.ports['o1'])
    left=m.ports['o3'].dx; right=z.ports['o2'].dx
    # Remove the marked inner fold. A horizontal 1 um waveguide now joins
    # the tangent endpoints of the two retained outer Euler bends.
    R('long_stage_upper',[P(m,'o3'),(left,long_top),
        (right,long_top),P(z,'o2')],m.ports['o3'],z.ports['o2'])
    R('long_stage_lower',[P(m,'o4'),(left,long_low),(right,long_low),P(z,'o1')],m.ports['o4'],z.ports['o1'])
    # Three direct right-facing outputs: no final combiner or bottom detour.
    tips=[]; probes={}; endpoints={}; edge_ports={}; output_records=[]
    def edge(name,coordinate,side='left'):
        end,tip,port=_edge(c,name,coordinate,side)
        endpoints[name]=end; edge_ports[name]=port; tips.append(tip)
        probes[name]=(end[0]+(0.5 if side=='left' else -0.5),end[1])
        return end
    for name,port in [('edge_117',a.ports['o2']),('edge_119',a.ports['o1']),('edge_116',b.ports['o2'])]:
        up=port.orientation==90
        y=port.dy+(d if up else -d)
        end=edge(name,y)
        R(name+'_access',[end,(port.dx,y),point(port)],start=edge_ports[name],end=port)
    bottom_input=edge('edge_174',bottom_bus,'right')
    R('bottom_reference_bus',[P(a,'o4'),(a.ports['o4'].dx,bottom_bus),bottom_input],
      a.ports['o4'],edge_ports['edge_174'])
    output_records.append({'name':'reference','edge_port':'edge_174',
                           'route':'bottom_reference_bus','edge_taper_input_um':bottom_input})
    for name,port_name,route_name in [
        ('edge_161','o4','switch_3_lower_output'),
        ('edge_186','o3','switch_3_upper_output')]:
        port=z.ports[port_name]
        if abs(port.orientation)>1e-6: raise ValueError('Last-switch outputs must face east')
        end=edge(name,port.dy,'right')
        if end[0]<=port.dx: raise ValueError('No room for direct output: increase CHIP_WIDTH or reduce stage spacing')
        R(route_name,[point(port),end],port,edge_ports[name])
        output_records.append({'name':port_name,'edge_port':name,'route':route_name,
                               'mzi_port_um':point(port),'edge_taper_input_um':end})
    # The four black-marked passive reference loops and their eight edge
    # couplers have been removed. Three continuous optical tracks remain.
    expected=sorted([('edge_116','edge_186'),('edge_117','edge_161'),
                     ('edge_119','edge_174')])
    optical=region(c,OUTPUT_LAYER)
    actual=_network_groups(optical,probes,dbu)
    if actual!=expected:
        raise ValueError(f'Optical connectivity changed. Expected {expected}, got {actual}')
    expected_count=len(expected) if CONNECT_EDGE_TIPS else len(expected)+len(tips)
    if optical.count()!=expected_count:
        raise ValueError(f'Dangling or disconnected optical geometry: {optical.count()} polygons, expected {expected_count}')
    pads=add_pads(c,heaters)
    electrical=add_electrical_routes(c,pads,heaters)
    add_die(c,tips)
    lengths={r['name']:r['length_um'] for r in router.records}
    route_records={r['name']:r for r in router.records}
    def straight_section(name,axis,occurrence=0):
        """Measure a horizontal (0) or vertical (1) tangent-to-tangent straight."""
        points=np.array(route_records[name]['waypoints_um'])
        matches=[]
        for i,(start,end) in enumerate(zip(points,points[1:])):
            if abs(start[1-axis]-end[1-axis])>dbu/2:
                continue
            direction=np.zeros(2); direction[axis]=np.sign(end[axis]-start[axis])
            a=start+direction*(d if i else 0.0)
            b=end-direction*(d if i<len(points)-2 else 0.0)
            a=np.rint(a/dbu)*dbu; b=np.rint(b/dbu)*dbu
            matches.append({'route':name,'start_um':a.tolist(),'end_um':b.tolist(),
                            'length_um':round(abs(b[axis]-a[axis])/dbu)*dbu})
        if (axis==0 and len(matches)!=1) or occurrence>=len(matches):
            raise ValueError(f'{name}: missing or ambiguous selected straight section')
        return matches[occurrence]
    selected={
        'short_stage':{**straight_section('short_stage_lower',0),
            'label':'H1','parameter':'SHORT_STAGE_STRAIGHT_LENGTH','requested_um':requested_short},
        'long_stage':{**straight_section('long_stage_lower',0),
            'label':'H2','parameter':'LONG_STAGE_STRAIGHT_LENGTH','requested_um':requested_long},
        'bottom':{**straight_section('bottom_reference_bus',0),'label':'Bottom','automatic':True},
    }
    for name,value in [('short_stage',short_straight),('long_stage',long_straight)]:
        if abs(selected[name]['length_um']-value)>dbu/2:
            raise ValueError(f'{name}: generated straight does not match its requested length')
    # The right edge is fixed by the die, so the complete reference straight
    # depends on the splitter and die width, not on the two stage gaps.
    predicted_bottom=bottom_input[0]-a.ports['o4'].dx-d
    if abs(selected['bottom']['length_um']-predicted_bottom)>dbu:
        raise ValueError('The automatic bottom straight does not match the output edge')
    selected['bottom']['formula']='right_edge_taper_input_x - splitter_output_x - routing_bend_span'
    selected['bottom']['depends_on_stage_gaps']=False
    selected_vertical={}
    for key,(label,parameter,requested,route) in vertical_inputs.items():
        measured=straight_section(route,1)
        if abs(measured['length_um']-vertical[key])>dbu/2:
            raise ValueError(f'{parameter}: generated vertical straight does not match its input')
        selected_vertical[key]={**measured,'label':label,'parameter':parameter,'requested_um':requested}
        if key.startswith(('short_','long_')):
            other=straight_section(route,1,1)
            if abs(other['length_um']-vertical[key])>dbu/2:
                raise ValueError(f'{parameter}: opposite leg does not match the selected length')
            selected_vertical[key]['opposite_leg']=other
    report={'mzi_count':4,'mzi_settings':settings.__dict__,'chip_size_um':[CHIP_WIDTH,CHIP_HEIGHT],
      'mzi_geometry':core.info.model_dump(),'last_mzi_geometry':output_core.info.model_dump(),
      'direct_outputs':output_records,
      'selected_horizontal_straights_um':selected,
      'selected_vertical_straights_um':selected_vertical,
      'derived_stage_gaps_um':{'short':short_gap,'long':long_gap,'bend_span':d},
      'derived_vertical_positions_um':{'splitter_lower_ports_y':row1,'switch_lower_ports_y':row2,
          'short_loop_top_y':short_top,'long_loop_top_y':long_top,
          'short_loop_bottom_y':short_low,'long_loop_bottom_y':long_low},
      'construction_grid_um':dbu,'placements':placements,'routing':router.records,
      'stage_lengths_um':{
          'short_upper':lengths['short_stage_upper'],'short_lower':lengths['short_stage_lower'],
          'short_difference':lengths['short_stage_upper']-lengths['short_stage_lower'],
          'long_upper':lengths['long_stage_upper'],'long_lower':lengths['long_stage_lower'],
          'long_difference':lengths['long_stage_upper']-lengths['long_stage_lower']},
      'edge_groups':actual,'edge_couplers':len(tips),'heaters':heaters,'pads':pads,
      'removed_reference_loops':['outer_monitor_pair','right_monitor_pair','bottom_left_U','bottom_right_U'],
      'removed_edge_couplers':['edge_156','edge_163','edge_187','edge_191','edge_193','edge_199','edge_195','edge_197'],
      'removed_electrical_components':([f'PAD_RIGHT_{i:02d}' for i in range(1,11)]+
          ['TRACE_RIGHT_09','TRACE_RIGHT_10','HEATER_WG_172']),
      'removed_output_components':['BOTTOM_COMBINER','ROUTE_WG_172',
          'ROUTE_EDGE_161_EXTENSION','ROUTE_EDGE_174_EXTENSION'],
      'bottom_reference_connection':{'type':'direct_to_right_edge_coupler',
          'join_um':bottom_input,'bus_y_um':bottom_bus},
      'upper_connection':{'type':'horizontal_straight_between_outer_bends',
          'start_um':[left+d,long_top],
          'end_um':[right-d,long_top],
          'straight_length_um':right-left-2*d,'width_um':1.0},
      'electrical_routes':electrical,'spare_pads':[p['name'] for p in pads if p['connection']=='spare'],
      'layers':{'waveguide':OUTPUT_LAYER,'heater':ARM_OVERLAY_LAYER,'routing':CONTACT_PAD_LAYER,
                'trench':DEEP_TRENCH_LAYER,'outline':CHIP_SIZE_LAYER},
      'checks':{'optical_edge_pairs':True,'optical_polygon_count':expected_count,
                'selected_straight_lengths':True,'selected_vertical_straight_lengths':True,
                'automatic_bottom_straight':True,
                'electrical_contacts_and_clearance':bool(ROUTE_HEATERS),'trench_tip_only':bool(ADD_CHIP_LAYERS)},
      'coordinate_note':'Report uses construction coordinates; exported GDS subtracts half chip width and height.',
      'gds_translation_um':[-CHIP_WIDTH/2,-CHIP_HEIGHT/2]}
    report['reported_path_lengths']=reported_path_lengths(report)
    report['checks']['reported_path_length_consistency']=True
    # Translate the entire mask set as one rigid unit. Ports follow the same
    # transform; use a parent so component geometry and layer registration agree.
    top=gf.Component('SiNx_PIC_Compact_4MZIs')
    ref=top<<c
    ref.dmove((-CHIP_WIDTH/2,-CHIP_HEIGHT/2))
    top.add_ports(ref.ports)
    # Remove just the chip-wide wrapper. The top level now directly contains
    # the individual MZIs, routes, pads, etc., rather than one giant instance.
    top.kdb_cell.flatten(1,False)
    top.info['mzi_count']=4
    top.info['layout']='compact_serial_delay_stages'
    top.info['chip_width_um']=CHIP_WIDTH
    top.info['chip_height_um']=CHIP_HEIGHT
    return top,report


def write_layer_properties(path):
    import xml.etree.ElementTree as ET
    root=ET.Element('layer-properties')
    for layer,name,color,dither in [
        (OUTPUT_LAYER,'Waveguides','#b14cdb','I0'),
        (ARM_OVERLAY_LAYER,'Heaters','#ef3038','I0'),
        (CONTACT_PAD_LAYER,'Metal and pads','#3b62cb','I3'),
        (DEEP_TRENCH_LAYER,'Deep trench','#b9ab59','I1'),
        (CHIP_SIZE_LAYER,'Chip outline','#666666','I1')]:
        prop=ET.SubElement(root,'properties')
        for key,value in {'name':name,'source':f'{layer[0]}/{layer[1]}@1','frame-color':color,
          'fill-color':color,'dither-pattern':dither,'visible':'true','valid':'true','width':'1'}.items():
            ET.SubElement(prop,key).text=value
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


def export_gds(top,path,report):
    options=kdb.SaveLayoutOptions(); options.gds2_max_vertex_count=4000
    top.write_gds(path,with_metadata=False,save_options=options)
    # Read the written file, then compare every layer against constructed masks.
    check=kdb.Layout(); check.read(str(path))
    if abs(check.dbu-top.kcl.dbu)>1e-12: raise ValueError('Export changed database unit')
    for index in top.kcl.layout.layer_indices():
        expected=kdb.Region(top.kdb_cell.begin_shapes_rec(index)).merged()
        if expected.is_empty(): continue
        info=top.kcl.layout.get_info(index)
        other=check.find_layer(info.layer,info.datatype)
        if other is None: raise ValueError(f'Missing exported layer {info}')
        actual=kdb.Region(check.top_cell().begin_shapes_rec(other)).merged()
        if not (actual^expected).is_empty(): raise ValueError(f'Export geometry differs on {info}')
    if ADD_CHIP_LAYERS:
        bbox=check.top_cell().bbox().to_dtype(check.dbu)
        if abs(bbox.left+bbox.right)>check.dbu or abs(bbox.bottom+bbox.top)>check.dbu:
            raise ValueError('Die is not centered after export')
    report['checks']['gds_layer_readback']=True
    parent_counts={}
    for cell in check.each_cell():
        for instance in cell.each_inst():
            key=instance.cell_index
            parent_counts[key]=parent_counts.get(key,0)+instance.na*instance.nb if instance.is_regular_array() else parent_counts.get(key,0)+1
    if any(count>1 for count in parent_counts.values()):
        raise ValueError('The editable GDS unexpectedly shares a child cell between instances')
    names=['MZI_1_SPLITTER','MZI_2_SW1','MZI_3_SW2','MZI_4_SW3']
    for name in names:
        cell=check.cell(name)
        if cell is None or not list(cell.each_inst()):
            raise ValueError(f'Editable MZI hierarchy missing: {name}')
    for name in report['removed_electrical_components']:
        if check.cell(name) is not None:
            raise ValueError(f'Removed right-side electrical component still present: {name}')
    report['checks']['right_side_electrical_removed']=True
    for name in report['removed_output_components']:
        if check.cell(name) is not None:
            raise ValueError(f'Removed output component still present: {name}')
    if any(cell.name.startswith('COMB_') for cell in check.each_cell()):
        raise ValueError('Removed combiner geometry is still present')
    report['checks']['combiner_and_bottom_output_routes_removed']=True
    report['checks']['independent_cell_instances']=True
    chip=check.top_cell()
    if check.cell('compact_four_mzi_pic') is not None:
        raise ValueError('The chip-wide wrapper must not be present in the exported GDS')
    report['checks']['chip_wrapper_removed']=True
    report['hierarchy']={'top':check.top_cell().name,'chip':chip.name,
                         'chip_children':[i.cell.name for i in chip.each_inst()],
                         'cell_count':check.cells()}
    report['gdsfactory_version']=gf.__version__
    return check


def export_flat_gds(hierarchical_layout,path,report):
    """Export a one-cell layout containing every individual polygon directly.

    Keep polygons separate rather than merging whole waveguide networks.
    There are no cell instances to enter, so polygon editing is available at
    the top level when the file is opened in KLayout Editor mode.
    """
    flat=kdb.Layout()
    flat.dbu=hierarchical_layout.dbu
    cell=flat.create_cell('SiNx_PIC_Flat_Edit')
    source=hierarchical_layout.top_cell()
    for index in hierarchical_layout.layer_indices():
        shapes=kdb.Region(source.begin_shapes_rec(index))
        if shapes.is_empty(): continue
        info=hierarchical_layout.get_info(index)
        target_layer=flat.layer(info.layer,info.datatype)
        cell.shapes(target_layer).insert(shapes)
    options=kdb.SaveLayoutOptions(); options.gds2_max_vertex_count=4000
    flat.write(str(path),options)
    readback=kdb.Layout(); readback.read(str(path))
    if readback.cells()!=1 or list(readback.top_cell().each_inst()):
        raise ValueError('Flat edit export must contain one cell and no instances')
    for index in hierarchical_layout.layer_indices():
        expected=kdb.Region(source.begin_shapes_rec(index)).merged()
        if expected.is_empty(): continue
        info=hierarchical_layout.get_info(index)
        target=readback.find_layer(info.layer,info.datatype)
        if target is None: raise ValueError(f'Missing flat-export layer {info}')
        actual=kdb.Region(readback.top_cell().begin_shapes_rec(target)).merged()
        if not (expected^actual).is_empty():
            raise ValueError(f'Flat export changed geometry on layer {info}')
    report['checks']['flat_gds_readback']=True
    report['flat_export']={'file':path.name,'cell_count':1,'instance_count':0,
                          'shape_count':sum(readback.top_cell().shapes(i).size() for i in readback.layer_indices())}
    return readback


def make_previews(layout,out,report):
    """Plot polygons from the exported GDS; no hand-drawn substitute geometry."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    raw_polygons={}
    for layer in [OUTPUT_LAYER,ARM_OVERLAY_LAYER,CONTACT_PAD_LAYER,CHIP_SIZE_LAYER]:
        index=layout.find_layer(*layer)
        if index is None: continue
        reg=kdb.Region(layout.top_cell().begin_shapes_rec(index)).merged()
        raw_polygons[layer]=[np.array([(p.x*layout.dbu/1000,p.y*layout.dbu/1000) for p in poly.each_point_hull()]) for poly in reg.each()]
    style={OUTPUT_LAYER:('#703ab4',0.48,1),ARM_OVERLAY_LAYER:('#eb453e',0.45,1),
           CONTACT_PAD_LAYER:('#bdcbee',0.25,0.8)}
    def draw(ax,metal=True):
        for layer in [CONTACT_PAD_LAYER,OUTPUT_LAYER,ARM_OVERLAY_LAYER]:
            if layer==CONTACT_PAD_LAYER and not metal: continue
            color,lw,alpha=style[layer]
            ax.add_collection(PolyCollection(raw_polygons.get(layer,[]),facecolors=color,
                edgecolors=color,linewidths=lw,alpha=alpha,rasterized=False))
        ax.set_aspect('equal'); ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
        ax.grid(alpha=.12); ax.set_axisbelow(True)
        for spine in ax.spines.values(): spine.set_color('#ccd1db')
    fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained')
    for ax,metal,title in zip(axes,[False,True],['Optical layout and heaters','Complete mask layout']):
        draw(ax,metal)
        ax.set_xlim(-CHIP_WIDTH/2000,CHIP_WIDTH/2000)
        ax.set_ylim(-CHIP_HEIGHT/2000,CHIP_HEIGHT/2000)
        ax.set_title(title,fontweight='bold')
    for placement in report['placements']:
        p=placement['ports_um']; x=(p['o1'][0]+p['o4'][0])/2-CHIP_WIDTH/2
        y=p['o2'][1]-CHIP_HEIGHT/2
        label='Splitter' if placement['name']=='MZI_1' else placement['name'].replace('MZI_','SW ')
        # User-facing switches are 1, 2, 3; MZI_1 is the splitter.
        if placement['name']!='MZI_1': label=f"SW {int(placement['name'][-1])-1}"
        axes[0].text(x/1000,(y+135)/1000,label,ha='center',fontsize=9,color='#333')
    axes[0].legend(handles=[Patch(color='#703ab4',label='Waveguides'),Patch(color='#eb453e',label='Heaters')],
                   loc='upper left',frameon=False)
    fig.suptitle(f'Compact PIC | 4 MZIs | {CHIP_WIDTH/1000:g} × {CHIP_HEIGHT/1000:g} mm',fontsize=17,fontweight='bold')
    fig.savefig(out/'compact_pic_preview.png',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    draw(ax,False)
    ax.set_xlim(-CHIP_WIDTH/2000+0.25,CHIP_WIDTH/2000-0.15)
    ax.set_ylim(-CHIP_HEIGHT/2000+0.25,report['placements'][1]['ports_um']['o2'][1]/1000-CHIP_HEIGHT/2000+0.45)
    ax.set_title('Lower section: MZIs and three direct right-edge outputs',fontweight='bold')
    fig.savefig(out/'compact_pic_detail.png',dpi=200)
    # Dimension labels identify exactly the three marked straight sections.
    for key,item in report['selected_horizontal_straights_um'].items():
        a=(np.array(item['start_um'])+report['gds_translation_um'])/1000
        b=(np.array(item['end_um'])+report['gds_translation_um'])/1000
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#151515',lw=2,
                marker='|',markersize=8,zorder=5)
        label=f"{item['label']} = {item['length_um']:.3f} µm"
        if key=='bottom': label+=' (automatic)'
        ax.annotate(label,((a[0]+b[0])/2,a[1]),xytext=(0,-18),
                    textcoords='offset points',ha='center',va='top',fontsize=9,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.9},zorder=6)
    ax.set_title('Selectable stage straights and automatic reference-output straight',fontweight='bold')
    fig.savefig(out/'compact_pic_lengths.png',dpi=200)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,9),layout='constrained')
    draw(ax,False)
    ax.set_xlim(-CHIP_WIDTH/2000+0.25,CHIP_WIDTH/2000-0.15)
    ax.set_ylim(-CHIP_HEIGHT/2000+0.25,CHIP_HEIGHT/2000-0.4)
    for item in report['selected_vertical_straights_um'].values():
        a=(np.array(item['start_um'])+report['gds_translation_um'])/1000
        b=(np.array(item['end_um'])+report['gds_translation_um'])/1000
        ax.plot([a[0],b[0]],[a[1],b[1]],color='#151515',lw=2,
                marker='_',markersize=8,zorder=5)
        tall=item['length_um']>1200
        value=f"{item['length_um']:.3f}".rstrip('0').rstrip('.')
        ax.annotate(f"{item['label']} = {value} µm",
                    (a[0],(a[1]+b[1])/2),xytext=(-10,0),
                    textcoords='offset points',ha='right',va='center',
                    rotation=90 if tall else 0,fontsize=9,
                    bbox={'facecolor':'white','edgecolor':'none','alpha':0.9},zorder=6)
    ax.set_title('Six selectable vertical straights (bend lengths excluded)',fontweight='bold')
    fig.savefig(out/'compact_pic_vertical_lengths.png',dpi=200)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-show',action='store_true',help='Do not open KLayout')
    parser.add_argument('--no-preview',action='store_true',help='Do not write PNG previews')
    parser.add_argument('--short-straight',type=float,default=None,metavar='UM',
                        help='H1 horizontal straight between SW1 and SW2; excludes bends')
    parser.add_argument('--long-straight',type=float,default=None,metavar='UM',
                        help='H2 horizontal straight between SW2 and SW3; excludes bends')
    vertical_options=[
        ('--splitter-bottom','splitter_bottom_length','V1: splitter down to bottom-bus bend'),
        ('--splitter-to-sw1','splitter_to_sw1_length','V2: splitter up to SW1'),
        ('--short-upper','short_upper_vertical_length','V3: short upper loop vertical straight'),
        ('--short-lower','short_lower_vertical_length','V4: short lower loop vertical straight'),
        ('--long-upper','long_upper_vertical_length','V5: long upper loop vertical straight'),
        ('--long-lower','long_lower_vertical_length','V6: long lower loop vertical straight'),
    ]
    group=parser.add_argument_group('vertical straight lengths (um, excluding bends)')
    for flag,dest,description in vertical_options:
        group.add_argument(flag,dest=dest,type=float,default=None,metavar='UM',help=description)
    args=parser.parse_args()
    gf.gpdk.PDK.activate(); gf.clear_cache()
    top,report=rebuild_chip(short_straight_length=args.short_straight,
                            long_straight_length=args.long_straight,
                            **{dest:getattr(args,dest) for flag,dest,description in vertical_options})
    out=Path(OUTPUT_DIR); out.mkdir(parents=True,exist_ok=True)
    path=out/'SiNx_PIC_Compact_4MZIs.gds'
    layout=export_gds(top,path,report)
    flat_path=out/'SiNx_PIC_FLAT_EDIT.gds'
    export_flat_gds(layout,flat_path,report)
    write_layer_properties(out/'sinx_compact_layers.lyp')
    if MAKE_PREVIEW and not args.no_preview: make_previews(layout,out,report)
    write_path_length_reports(out,report)
    (out/'compact_layout_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Generated: {path}')
    print(f'Direct polygon editing: {flat_path} (one cell, no child instances)')
    print(f'Four MZIs with {report["mzi_geometry"]["arm_transition_bend_type"]} arm transitions; '
          f'{report["edge_couplers"]} edge couplers; {len(report["pads"])} contact pads.')
    print('MZI_1: lower splitter; MZI_2/MZI_3/MZI_4: three switches in one row.')
    print('Checks: optical continuity, isolated metal nets, intended heater contacts, die/trench fit, GDS readback.')
    print('Stage lengths (um):',report['stage_lengths_um'])
    chosen=report['selected_horizontal_straights_um']
    print(f"Selected straights: H1={chosen['short_stage']['length_um']:.3f} um, "
          f"H2={chosen['long_stage']['length_um']:.3f} um; "
          f"bottom (automatic)={chosen['bottom']['length_um']:.3f} um")
    print('Selected vertical straights (um):',
          {v['label']:round(v['length_um'],3) for v in report['selected_vertical_straights_um'].values()})
    if PRINT_PATH_LENGTHS:
        print('L0-L4: nominal waveguide centerline lengths, including bends (picture order).')
        for label,item in report['reported_path_lengths']['paths'].items():
            print(f"{label}: {item['length_um']:12.3f} um = {item['length_mm']:.6f} mm  | {item['description']}")
        print('Endpoints: splitter output coupling-section exit to right edge-coupler taper input (tapers excluded).')
    if PRINT_WAVEGUIDE_LENGTHS:
        for r in report['routing']: print(f"{r['name']:30s} {r['length_um']:12.3f} um")
    if SHOW_LAYOUT and not args.no_show:
        try: gf.show(flat_path if SHOW_FLAT_GDS else path)
        except Exception as error: print(f'GDS saved. Open it manually in KLayout: {error}')


if __name__=='__main__':
    main()
