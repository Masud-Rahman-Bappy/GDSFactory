"""Custom Euler L/S bends and smooth quintic S bends for fitted transitions.

q is the dimensionless scale factor; width and returned coordinates are um.
Activate a PDK in the calling script before creating a component.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, factorial, isfinite, pi, sqrt
from typing import Literal

import gdsfactory as gf
import numpy as np

from .technology import LAYERS

Handedness = Literal["left", "right"]

@dataclass(frozen=True)
class BendGeometry:
    """Numerical geometry used to build a two-port bend cell."""

    centerline: np.ndarray
    polygon: np.ndarray
    width_in: float
    width_out: float



def smooth_width_transition(
    m: float, theta: float | np.ndarray, theta_max: float
) -> np.ndarray:
    """Return a smooth width change from zero to ``m`` over an angle range."""
    theta_array = np.asarray(theta, dtype=float)
    if theta_max <= 0:
        raise ValueError("theta_max must be positive")
    if np.any(theta_array < 0):
        raise ValueError("theta must be non-negative")
    r = (np.sqrt(theta_array) - 0.5 * sqrt(theta_max)) / (
        0.5 * sqrt(theta_max)
    )
    return m * (0.5 + 15.0 / 16.0 * r - 5.0 / 8.0 * r**3 + 3.0 / 16.0 * r**5)



def _validate_common(
    q: float, width: float, npoints: int, k_taylor: int, f: float
) -> None:
    if q <= 0:
        raise ValueError("q must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if npoints < 3:
        raise ValueError("npoints must be at least 3")
    if k_taylor < 0:
        raise ValueError("k_taylor must be non-negative")
    if not 0.0 <= f <= 1.0:
        raise ValueError("f must be between 0 and 1")



def _euler_series(
    theta: np.ndarray, scale: float, k_taylor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the common x/y Euler-bend Taylor series."""
    x = np.zeros_like(theta, dtype=float)
    y = np.zeros_like(theta, dtype=float)
    for k in range(k_taylor + 1):
        x_term = (
            scale / sqrt(2.0) * (-1.0) ** k * theta ** (2 * k + 0.5)
            / ((2 * k + 0.5) * factorial(2 * k))
        )
        y_term = (
            scale / sqrt(2.0) * (-1.0) ** k * theta ** (2 * k + 1.5)
            / ((2 * k + 1.5) * factorial(2 * k + 1))
        )
        x += x_term
        y += y_term
        if k >= 8 and max(np.max(np.abs(x_term)), np.max(np.abs(y_term))) < 1e-15:
            break
    return x, y



def _paired_transform(
    x: np.ndarray, y: np.ndarray, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the anchored x2/y2 transform shared by all scripts."""
    x_pair = (
        -x * np.sin(delta) + y * np.cos(delta)
        + x[-1] * (1.0 + np.sin(delta)) - y[-1] * np.cos(delta)
    )
    y_pair = (
        x * np.cos(delta) + y * np.sin(delta)
        + y[-1] * (1.0 - np.sin(delta)) - x[-1] * np.cos(delta)
    )
    return x_pair, y_pair



def _simple_pair_geometry(
    q: float,
    width: float,
    m: float,
    theta_deg: float,
    npoints: int,
    k_taylor: int,
    f: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return centerline and boundaries for the common paired section."""
    _validate_common(q, width, npoints, k_taylor, f)
    if theta_deg <= 0:
        raise ValueError("theta_deg must be positive")
    theta_max = np.deg2rad(theta_deg)
    theta = np.linspace(0.0, theta_max, npoints)
    x1, y1 = _euler_series(theta, q * 15.0, k_taylor)
    g = 1.0 - f
    dw = smooth_width_transition(m, theta, theta_max)
    x1i = x1 - (0.5 * width + g * dw) * np.sin(theta)
    x1o = x1 + (0.5 * width - f * dw) * np.sin(theta)
    y1i = y1 + (0.5 * width + g * dw) * np.cos(theta)
    y1o = y1 - (0.5 * width - f * dw) * np.cos(theta)
    delta = 2.0 * theta_max - 3.0 * pi / 2.0
    x2, y2 = _paired_transform(x1, y1, delta)
    x2i, y2i = _paired_transform(x1i, y1i, delta)
    x2o, y2o = _paired_transform(x1o, y1o, delta)
    centerline = np.column_stack((np.r_[x1, x2[-2::-1]], np.r_[y1, y2[-2::-1]]))
    inner = np.column_stack((np.r_[x1i, x2i[-2::-1]], np.r_[y1i, y2i[-2::-1]]))
    outer = np.column_stack((np.r_[x1o, x2o[-2::-1]], np.r_[y1o, y2o[-2::-1]]))
    return centerline, inner, outer



def _mirror_x(points: np.ndarray) -> np.ndarray:
    result = points.copy()
    result[:, 0] *= -1.0
    return result



def _orientation(outside: np.ndarray, inside: np.ndarray) -> float:
    vector = outside - inside
    return degrees(atan2(vector[1], vector[0])) % 360.0



def _snap_cardinal(angle: float, tolerance: float = 1.0) -> float:
    """Snap a sampled end tangent to its intended 0/90/180/270-degree axis."""
    target = (round(angle / 90.0) * 90.0) % 360.0
    error = (angle - target + 180.0) % 360.0 - 180.0
    return target if abs(error) <= tolerance else angle



def _component_from_geometry(
    geometry: BendGeometry, layer: gf.typings.LayerSpec, bend_type: str
) -> gf.Component:
    """Build a GDSFactory polygon, ports, and metadata from numerical data."""
    if geometry.centerline.ndim != 2 or geometry.centerline.shape[1] != 2:
        raise ValueError("centerline must be an N-by-2 array")
    if geometry.polygon.ndim != 2 or geometry.polygon.shape[1] != 2:
        raise ValueError("polygon must be an N-by-2 array")
    if not np.isfinite(geometry.centerline).all() or not np.isfinite(geometry.polygon).all():
        raise ValueError("bend geometry contains a non-finite coordinate")
    c = gf.Component()
    c.add_polygon(geometry.polygon, layer=layer)
    cl = geometry.centerline
    c.add_port(name="o1", center=cl[0], width=geometry.width_in,
               orientation=_snap_cardinal(_orientation(cl[0], cl[1])),
               layer=layer, port_type="optical")
    c.add_port(name="o2", center=cl[-1], width=geometry.width_out,
               orientation=_snap_cardinal(_orientation(cl[-1], cl[-2])),
               layer=layer, port_type="optical")
    segment_lengths = np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1]))
    span = np.ptp(geometry.polygon, axis=0)
    c.info["bend_type"] = bend_type
    c.info["centerline_length"] = float(np.sum(segment_lengths))
    c.info["port_separation"] = float(np.linalg.norm(cl[-1] - cl[0]))
    c.info["x_span"] = float(span[0])
    c.info["y_span"] = float(span[1])
    return c



def _l_geometry(
    q: float, width: float, m: float, theta_deg: float, npoints: int,
    k_taylor: int, f: float, handedness: Handedness
) -> BendGeometry:
    centerline, inner, outer = _simple_pair_geometry(
        q, width, m, theta_deg, npoints, k_taylor, f
    )
    polygon = np.vstack((inner, outer[::-1]))
    if handedness == "right":
        centerline, polygon = _mirror_x(centerline), _mirror_x(polygon)
    elif handedness != "left":
        raise ValueError("handedness must be 'left' or 'right'")
    # The width transition rises toward the join and returns to W0 at o2.
    return BendGeometry(centerline, polygon, width, width)



def _s_geometry(
    q: float, width: float, m: float, theta_deg: float, npoints: int,
    k_taylor: int, f: float
) -> BendGeometry:
    c1, inner1, outer1 = _simple_pair_geometry(
        q, width, m, theta_deg, npoints, k_taylor, f
    )
    pivot = c1[-1]
    c2 = -c1 + 2.0 * pivot
    inner2 = -inner1 + 2.0 * pivot
    outer2 = -outer1 + 2.0 * pivot
    polygon = np.vstack((inner1[:-1], outer2[::-1], inner2[:-1], outer1[::-1]))
    centerline = np.vstack((c1, c2[-2::-1]))
    return BendGeometry(centerline, polygon, width, width)



@gf.cell
def euler_l_bend(
    q: float = 5.0, width: float = 1.0, m: float = 0.0,
    theta_deg: float = 45.0, npoints: int = 50, k_taylor: int = 500,
    f: float = 0.0, handedness: Handedness = "left",
    layer: gf.typings.LayerSpec = LAYERS.WG,
) -> gf.Component:
    """Return a 90-degree Euler-style bend with selectable handedness."""
    return _component_from_geometry(
        _l_geometry(q, width, m, theta_deg, npoints, k_taylor, f, handedness),
        layer, "L"
    )



@gf.cell
def euler_s_bend(
    q: float = 5.0, width: float = 1.0, m: float = 0.0,
    theta_deg: float = 22.5, npoints: int = 50, k_taylor: int = 500,
    f: float = 0.0, layer: gf.typings.LayerSpec = LAYERS.WG,
) -> gf.Component:
    """Return a point-reflected Euler-style S bend."""
    return _component_from_geometry(
        _s_geometry(q, width, m, theta_deg, npoints, k_taylor, f), layer, "S"
    )


@gf.cell
def smooth_s_bend(
    length: float = 223.818,
    offset: float = 223.818,
    width: float = 1.0,
    npoints: int = 401,
    layer: gf.typings.LayerSpec = LAYERS.WG,
) -> gf.Component:
    """A constant-width S bend fitted to two parallel horizontal ports.

    Centerline: x=L*t, y=offset*(10*t**3-15*t**4+6*t**5), 0<=t<=1.
    Both endpoint slopes and curvatures are zero. This is a smooth quintic
    S bend, distinct from the existing fixed-angle Euler S-bend factory.
    Rotation/reflection of a reference provides the vertical versions.
    """
    if not all(isfinite(v) for v in (length,offset,width)) or length<=0 or width<=0:
        raise ValueError('S-bend length/width must be positive; offset must be finite')
    if not isinstance(npoints,int) or npoints<51:
        raise ValueError('Use at least 51 centerline samples for the S bend')
    t=np.linspace(0.0,1.0,npoints)
    f=t**3*(10.0-15.0*t+6.0*t*t)
    slope=(offset/length)*30.0*t*t*(1.0-t)**2
    second=(offset/length**2)*(60.0*t-180.0*t*t+120.0*t**3)
    curvature=np.abs(second)/(1.0+slope*slope)**1.5
    max_curvature=float(np.max(curvature))
    if max_curvature*width/2>=1:
        raise ValueError('S bend is too tight for its width')
    centerline=np.column_stack((length*t,offset*f))
    normals=np.column_stack((-slope,np.ones_like(t)))/np.sqrt(1.0+slope*slope)[:,None]
    inner=centerline+width/2*normals
    outer=centerline-width/2*normals
    polygon=np.vstack((inner,outer[::-1]))
    c=gf.Component()
    c.add_polygon(polygon,layer=layer)
    c.add_port('o1',center=(0,0),width=width,orientation=180,layer=layer,port_type='optical')
    c.add_port('o2',center=(length,offset),width=width,orientation=0,layer=layer,port_type='optical')
    nodes,weights=np.polynomial.legendre.leggauss(128)
    u=(nodes+1)/2
    derivative=offset*30.0*u*u*(1-u)**2
    c.info['centerline_length']=float(np.dot(weights,np.hypot(length,derivative))/2)
    c.info['bend_type']='S_QUINTIC'
    c.info['length_um']=length
    c.info['offset_um']=offset
    c.info['width_um']=width
    c.info['minimum_radius_um']=float(1/max_curvature) if max_curvature else None
    c.info['endpoint_curvature']=0.0
    return c
