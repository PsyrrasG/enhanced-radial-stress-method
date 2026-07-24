"""
Enhanced Radial Stress Method — numerical core for elliptical web openings.

This module is the elliptical counterpart of :mod:`enhanced_rsm.core`. It
provides the geometry factory and the two force-equilibrium kernels whose
expressions depend on the shape of the opening; the remaining kernels are
geometry-independent and are re-used unchanged from the circular core.

Why the geometry differs
------------------------
For a circular opening the inclined planes are defined by equal increments of
the angle to the vertical, and the normal to the edge at any point passes
through the centre of the opening. Neither holds for an ellipse. The normal at a
point P on the perimeter meets the vertical centre-line at a point A that moves
with P, so the inclined plane must be constructed from the local geometry; and
the perimeter of an ellipse has no closed-form expression, so points cannot be
placed by a simple angular increment.

Points are therefore distributed at equal **arc length** around the quarter
perimeter. The total arc length is obtained by numerical integration and the
parameter increments by a fourth-order Runge-Kutta scheme. Equal arc-length
spacing is what allows a direct comparison with the equally spaced nodes of a
finite element mesh.

Conventions
-----------
The unit, quadrant and sign conventions are those of :mod:`enhanced_rsm.core`.
The critical location is reported as a **point index** (0 to 90) and its
coordinates on the perimeter, rather than as an angle.

Relationship to the circular case
---------------------------------
The expressions here are the general ones. Setting ``a_b_ratio = 1`` gives
``yA = 0`` and a constant ``r_thP = h_o / 2``, at which point they reduce
algebraically to the circular expressions of :mod:`enhanced_rsm.core`. That
equivalence is exercised as a test.
"""

from typing import NamedTuple

import numpy as np
from scipy.integrate import quad

# Geometry-independent kernels, shared with the circular core.
from .core import (
    THETAS,
    calculate_zep,
    limit_web_plasticity,
    compute_elastic_stress,
    find_optimal_r,
    redistribute_moment,
)

# Point indices around the quarter perimeter, 0 to 90.
NPOINTS = np.arange(0, 91, 1)

__all__ = [
    "NPOINTS",
    "EllipticalSectionProperties",
    "build_elliptical_section_properties",
    "perform_rsm",
    "calculate_moment_capacity",
    "solve_elastic_limit",
    "point_coordinates",
    "perimeter_coordinates_360",
    # Re-exported unchanged from the circular core.
    "calculate_zep",
    "limit_web_plasticity",
    "compute_elastic_stress",
    "find_optimal_r",
    "redistribute_moment",
]


# ---------------------------------------------------------------------------
# Perimeter discretisation
# ---------------------------------------------------------------------------
def _arc_length_derivative(t, a, b):
    """Derivative of arc length with respect to the ellipse parameter.

    ``ds/dt = sqrt(a^2 sin^2 t + b^2 cos^2 t)``.

    Parameters
    ----------
    t : float or numpy.ndarray
        Ellipse parameter (radians).
    a, b : float
        Semi-axes of the ellipse (mm).

    Returns
    -------
    float or numpy.ndarray
        ``ds/dt`` at ``t``.
    """
    return np.sqrt((a * np.sin(t))**2 + (b * np.cos(t))**2)


def _total_arc_length(a, b):
    """Arc length of the quarter ellipse, by numerical integration.

    Parameters
    ----------
    a, b : float
        Semi-axes of the ellipse (mm).

    Returns
    -------
    float
        Quarter-perimeter arc length (mm).
    """
    result, _ = quad(_arc_length_derivative, 0, np.pi / 2, args=(a, b))
    return result


def _rk4_step(t, ds, a, b):
    """Advance the ellipse parameter by one equal-arc-length step.

    Integrates ``dt/ds = 1 / sqrt(a^2 sin^2 t + b^2 cos^2 t)`` over a step of
    arc length ``ds`` using a fourth-order Runge-Kutta scheme.

    Parameters
    ----------
    t : float
        Current ellipse parameter (radians).
    ds : float
        Arc-length increment (mm).
    a, b : float
        Semi-axes of the ellipse (mm).

    Returns
    -------
    float
        The parameter at the next point.
    """
    k1 = 1 / _arc_length_derivative(t, a, b)
    k2 = 1 / _arc_length_derivative(t + 0.5 * ds * k1, a, b)
    k3 = 1 / _arc_length_derivative(t + 0.5 * ds * k2, a, b)
    k4 = 1 / _arc_length_derivative(t + ds * k3, a, b)
    return t + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def _equally_spaced_points(a, b, n):
    """Generate ``n`` points at equal arc-length spacing on a quarter ellipse.

    Parameters
    ----------
    a, b : float
        Semi-axes of the ellipse (mm).
    n : int
        Number of points, including both ends.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_points, y_points)``, each of shape ``(n,)``.
    """
    delta_s = _total_arc_length(a, b) / (n - 1)

    t_values = [0.0]
    for _ in range(n - 1):
        t_values.append(_rk4_step(t_values[-1], delta_s, a, b))

    x_points = a * np.cos(t_values)
    y_points = b * np.sin(t_values)
    return x_points, y_points


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------
class EllipticalSectionProperties(NamedTuple):
    """Geometry and section properties for an elliptical opening.

    Field names shared with :class:`enhanced_rsm.core.SectionProperties` are
    deliberately identical, so that the geometry-independent kernels operate on
    either without modification. The additional fields describe the elliptical
    geometry.

    All array attributes are indexed by perimeter point and have shape ``(91,)``;
    the ``*_exp`` variants are reshaped to ``(91, 1)`` for broadcasting against a
    trailing load or plasticity axis.

    Attributes
    ----------
    h, h_o, b_f, t_w, t_f, r, f_y : float
        Section depth, opening height (the vertical axis), flange width, web
        thickness, flange thickness, root radius (mm) and yield strength (MPa).
    a_b_ratio : float
        Ratio of the horizontal to the vertical axis of the opening.
    a_semi, b_semi : float
        Semi-axes of the ellipse (mm); ``b_semi`` is the vertical semi-axis.
    ellipse_type : str
        ``'Horizontal'`` when the horizontal axis is the longer one, otherwise
        ``'Vertical'``.
    x_points, y_points : numpy.ndarray
        Coordinates of the perimeter points, measured from the centre of the
        opening (mm), ordered from the vertical centre-line to the horizontal.
    thP_rad, thP_deg : numpy.ndarray
        Angle between the inclined plane at each point and the vertical
        centre-line, in radians and degrees.
    yA : numpy.ndarray
        Ordinate at which the normal at each point P meets the vertical
        centre-line (mm).
    r_thP : numpy.ndarray
        Distance from that intersection to the perimeter point P (mm). 
        This is the elliptical generalisation of the radius of a circular opening.
    d_T_th, h_T_th, t_f_th, A_T_th, A_f_th, I_T_th, c_th : numpy.ndarray
        Inclined-Tee web-outstand depth, total depth, inclined flange thickness,
        Tee area, flange area, second moment of area and neutral-axis position.
    thP_rad_exp, yA_exp, r_thP_exp, d_T_th_exp, h_T_th_exp, t_f_th_exp, A_T_th_exp, A_f_th_exp, I_T_th_exp, c_th_exp : numpy.ndarray
        The ``(91, 1)`` broadcasting variants of the above.
    d_T_o, A_T_o, c_o, z_o : float
        Section properties at the vertical centre-line.
    I_beam : float
        Second moment of area of the perforated section at the centre-line.

    Notes
    -----
    As in the circular core, the pure-shear resistance and the trial-shear bound
    are forces rather than section properties and are returned separately by
    :func:`build_elliptical_section_properties`.
    """

    h: float
    h_o: float
    b_f: float
    t_w: float
    t_f: float
    r: float
    f_y: float

    a_b_ratio: float
    a_semi: float
    b_semi: float
    ellipse_type: str

    x_points: np.ndarray
    y_points: np.ndarray

    thP_rad: np.ndarray
    thP_deg: np.ndarray
    yA: np.ndarray
    r_thP: np.ndarray

    d_T_th: np.ndarray
    h_T_th: np.ndarray
    t_f_th: np.ndarray
    A_T_th: np.ndarray
    A_f_th: np.ndarray
    I_T_th: np.ndarray
    c_th: np.ndarray

    thP_rad_exp: np.ndarray
    yA_exp: np.ndarray
    r_thP_exp: np.ndarray
    d_T_th_exp: np.ndarray
    h_T_th_exp: np.ndarray
    t_f_th_exp: np.ndarray
    A_T_th_exp: np.ndarray
    A_f_th_exp: np.ndarray
    I_T_th_exp: np.ndarray
    c_th_exp: np.ndarray

    d_T_o: float
    A_T_o: float
    c_o: float
    z_o: float
    I_beam: float


def build_elliptical_section_properties(h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y):
    """Compute the inclined-Tee section properties at every perimeter point.

    The quarter perimeter is discretised at equal arc-length intervals, the
    inclined plane at each point is constructed from the local normal, and the
    resulting Tee properties are evaluated.

    Parameters
    ----------
    h : float
        Overall section depth (mm).
    h_o : float
        Opening height, the vertical axis of the ellipse (mm).
    a_b_ratio : float
        Ratio of the horizontal to the vertical axis. A value of 1 gives a
        circular opening.
    b_f : float
        Flange width (mm).
    t_w : float
        Web thickness (mm).
    t_f : float
        Flange thickness (mm).
    r : float
        Root radius (mm).
    f_y : float
        Yield strength (MPa).

    Returns
    -------
    props : EllipticalSectionProperties
        The populated, immutable geometry container.
    V_T_Rd : float
        Tee pure-shear resistance at the centre-line (kN).
    V_Rd : float
        Total pure-shear resistance at the centre-line (kN).
    max_Ved : float
        Upper bound of the trial-shear range (kN), ``floor(V_Rd)``.
    """
    if a_b_ratio <= 0:
        raise ValueError("'a_b_ratio' must be greater than zero.")

    # Ellipse axes: h_o is the vertical axis, scaled by a_b_ratio horizontally.
    b_o = h_o
    a_o = a_b_ratio * b_o
    a_semi = a_o / 2
    b_semi = b_o / 2
    ellipse_type = "Horizontal" if a_semi > b_semi else "Vertical"

    # Equal arc-length points on the quarter perimeter, ordered from the
    # vertical centre-line round to the horizontal, matching the angle
    # convention of the circular case.
    x_points, y_points = _equally_spaced_points(a_semi, b_semi, len(NPOINTS))
    x_points = np.flip(x_points)
    y_points = np.flip(y_points)

    # Pin the two ends exactly onto the axes.
    x_points[0], y_points[0] = 0.0, b_semi
    x_points[-1], y_points[-1] = a_semi, 0.0

    # Point A: where the normal to the ellipse at P meets the vertical
    # centre-line. For a circle this is the centre (yA = 0).
    yA = y_points * (1 - (a_semi**2) / (b_semi**2))

    # Angle between the inclined plane and the vertical centre-line.
    cos_thP = np.abs(y_points - yA) / np.sqrt(x_points**2 + (y_points - yA)**2)
    thP_rad = np.arccos(cos_thP)
    thP_deg = np.rad2deg(thP_rad)

    # Distance from A to P: the generalisation of the circular radius.
    r_thP = np.abs(y_points - yA) / np.cos(thP_rad)

    # Inclined-Tee properties at every point.
    if ellipse_type == "Horizontal":
        d_T_th = (0.5 * h - t_f + np.abs(yA)) / np.cos(thP_rad) - r_thP
    else:
        d_T_th = (0.5 * h - t_f - np.abs(yA)) / np.cos(thP_rad) - r_thP

    t_f_th = t_f / np.cos(thP_rad)
    h_T_th = d_T_th + t_f_th
    A_T_th = t_w * d_T_th + b_f * t_f_th
    A_f_th = b_f * t_f_th
    I_T_th = (
        d_T_th**4 * t_w**2
        + t_f_th**4 * b_f**2
        + 4 * d_T_th**3 * t_w * b_f * t_f_th
        + 4 * t_f_th**3 * d_T_th * t_w * b_f
        + 6 * d_T_th**2 * t_f_th**2 * b_f * t_w
    ) / (12 * (b_f * t_f_th + d_T_th * t_w))
    c_th = (
        0.5 * d_T_th**2 * t_w
        + d_T_th * t_f_th * b_f
        + 0.5 * t_f_th**2 * b_f
    ) / (t_w * d_T_th + t_f_th * b_f)

    # Centre-line values.
    d_T_o = 0.5 * h - b_semi
    d_w_o = (h - h_o - 2 * t_f) / 2
    A_T_o = t_f * b_f + d_w_o * t_w
    c_o = (
        0.5 * d_w_o**2 * t_w + d_w_o * t_f * b_f + 0.5 * t_f**2 * b_f
    ) / (t_w * d_w_o + t_f * b_f)
    z_o = 0.5 * (h - h_o) - c_o

    # Perforated-section second moment of area at the centre-line.
    A_f = b_f * t_f
    I_beam = 0.5 * A_f * (h - t_f)**2 + t_w * ((h - 2 * t_f)**3 - h_o**3) / 12

    # Pure-shear resistance at the centre-line; bounds the trial-shear range.
    h_T = (h - h_o) / 2
    d_T = h_T - t_f
    V_T_Rd = (t_w * d_T + 0.5 * t_f * (2 * r + t_w)) * 0.577 * f_y * 1e-3  # kN
    V_Rd = 2 * V_T_Rd  # kN
    max_Ved = np.floor(V_Rd)  # kN

    props = EllipticalSectionProperties(
        h=h, h_o=h_o, b_f=b_f, t_w=t_w, t_f=t_f, r=r, f_y=f_y,
        a_b_ratio=a_b_ratio, a_semi=a_semi, b_semi=b_semi,
        ellipse_type=ellipse_type,
        x_points=x_points, y_points=y_points,
        thP_rad=thP_rad, thP_deg=thP_deg, yA=yA, r_thP=r_thP,
        d_T_th=d_T_th, h_T_th=h_T_th, t_f_th=t_f_th, A_T_th=A_T_th,
        A_f_th=A_f_th, I_T_th=I_T_th, c_th=c_th,
        thP_rad_exp=thP_rad[:, None], yA_exp=yA[:, None],
        r_thP_exp=r_thP[:, None],
        d_T_th_exp=d_T_th[:, None], h_T_th_exp=h_T_th[:, None],
        t_f_th_exp=t_f_th[:, None], A_T_th_exp=A_T_th[:, None],
        A_f_th_exp=A_f_th[:, None], I_T_th_exp=I_T_th[:, None],
        c_th_exp=c_th[:, None],
        d_T_o=d_T_o, A_T_o=A_T_o, c_o=c_o, z_o=z_o, I_beam=I_beam,
    )
    return props, V_T_Rd, V_Rd, max_Ved


# ---------------------------------------------------------------------------
# Geometry-dependent kernels
# ---------------------------------------------------------------------------
def perform_rsm(sector, Ved, N_T, props, vectorised=True):
    """Enhanced-RSM force equilibrium on the inclined plane at each point.

    The elliptical form of the equilibrium relations. The lever arm is measured
    from the point A at which the local normal meets the vertical centre-line,
    which moves with the perimeter point P; for a circular opening, A is the centre
    and the expressions reduce to those of :func:`enhanced_rsm.core.perform_rsm`.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    Ved : numpy.ndarray or float
        Applied vertical shear (N). A 1-D array when ``vectorised`` is True,
        otherwise a scalar.
    N_T : numpy.ndarray or float
        Axial force in the Tee due to global bending (N), matching ``Ved``.
    props : EllipticalSectionProperties
        Elliptical section properties.
    vectorised : bool, optional
        If True, broadcast over a trailing load axis, giving results of shape
        ``(91, len(Ved))``; otherwise the result is ``(91,)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_th, V_th, M_th)`` — internal axial force, shear force and moment on
        the inclined plane.
    """
    thP_rad = props.thP_rad
    c_th = props.c_th
    r_thP = props.r_thP
    yA = props.yA
    c_o = props.c_o
    b_semi = props.b_semi

    if vectorised:
        Ved = Ved[np.newaxis, :]
        N_T = N_T[np.newaxis, :]
        thP_rad = thP_rad[:, np.newaxis]
        c_th = c_th[:, np.newaxis]
        r_thP = r_thP[:, np.newaxis]
        yA = yA[:, np.newaxis]

    # Lever arm from A to the Tee centroid, and the offset of A from the centre-line reference.
    lever = r_thP + c_th
    offset = c_o + b_semi + np.abs(yA)

    if sector in (1, 4):
        N_th = 0.5 * Ved * np.sin(thP_rad) + N_T * np.cos(thP_rad)
        V_th = -0.5 * Ved * np.cos(thP_rad) + N_T * np.sin(thP_rad)
        M_th = (
            0.5 * Ved * lever * np.sin(thP_rad)
            - N_T * (offset - lever * np.cos(thP_rad))
        )
    elif sector in (2, 3):
        N_th = -0.5 * Ved * np.sin(thP_rad) + N_T * np.cos(thP_rad)
        V_th = 0.5 * Ved * np.cos(thP_rad) + N_T * np.sin(thP_rad)
        M_th = (
            0.5 * Ved * lever * np.sin(thP_rad)
            + N_T * (offset - lever * np.cos(thP_rad))
        )
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    return N_th, V_th, M_th


def calculate_moment_capacity(sector, z, Ved, N_T, n, props, elastic_mode=True):
    """Elasto-plastic bending resistance, applied moment and their ratio.

    The bending resistance is identical to the circular case; the applied moment
    uses the elliptical lever arm, measured from the point at which the local
    normal meets the vertical centre-line.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    z : numpy.ndarray
        Neutral-axis depth from the outer flange fibre.
    Ved : numpy.ndarray or float
        Applied vertical shear (N).
    N_T : numpy.ndarray or float
        Axial force in the Tee due to global bending (N).
    n : float or numpy.ndarray
        Plasticity strain factor (1 at the elastic limit).
    props : EllipticalSectionProperties
        Elliptical section properties.
    elastic_mode : bool, optional
        Retained for signature parity with the circular core; the resistance
        expression is continuous in ``n`` and requires no branching.

    Returns
    -------
    tuple of numpy.ndarray
        ``(M_Rd, M_Rd_fl, M_Rd_tot, M_Ed, r)`` — web bending resistance, flange
        contribution, total resistance, applied moment and the ratio
        ``M_Ed / M_Rd_tot``.
    """
    A_f_exp = props.A_f_th_exp
    h_T_exp = props.h_T_th_exp
    t_f_exp = props.t_f_th_exp
    thP_rad_exp = props.thP_rad_exp
    r_thP_exp = props.r_thP_exp
    yA_exp = props.yA_exp
    t_w = props.t_w
    f_y = props.f_y
    c_o = props.c_o
    b_semi = props.b_semi

    # Elasto-plastic bending resistance in terms of the neutral-axis depth.
    M_Rd = (
        n * A_f_exp * ((z - 0.5 * t_f_exp)**2 / (h_T_exp - z)) * f_y
        + (n * t_w * (z - t_f_exp)**3) / (3 * (h_T_exp - z)) * f_y
        + (t_w * (3 * n**2 - 1)) / (6 * n**2) * (h_T_exp - z)**2 * f_y
    )
    # Additional resistance term from the flange.
    M_Rd_fl = n / (h_T_exp - z) * A_f_exp * t_f_exp**2 * f_y / 12
    M_Rd_tot = M_Rd + M_Rd_fl

    # Lever arm from A to the neutral axis of the inclined Tee.
    lever = r_thP_exp + h_T_exp - z
    offset = c_o + b_semi + np.abs(yA_exp)

    if sector in (1, 4):
        M_Ed = (
            -N_T * (offset - lever * np.cos(thP_rad_exp))
            + 0.5 * Ved * lever * np.sin(thP_rad_exp)
        )
    elif sector in (2, 3):
        M_Ed = (
            N_T * (offset - lever * np.cos(thP_rad_exp))
            + 0.5 * Ved * lever * np.sin(thP_rad_exp)
        )
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    r = M_Ed / M_Rd_tot
    return M_Rd, M_Rd_fl, M_Rd_tot, M_Ed, r


def solve_elastic_limit(sector, V_ed_values, N_T_values, props):
    """Elastic limit of one quadrant: the shear at which its first point yields.

    Vectorises over the trial-shear axis. For every perimeter point P and every
    trial shear force the elastic moment ratio is evaluated; the point P that reaches
    yield at the lowest shear is the critical point, and that shear force is the
    quadrant's elastic limit.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    V_ed_values : numpy.ndarray
        Trial shear forces (N).
    N_T_values : numpy.ndarray
        Axial force in the Tee for each trial shear (N).
    props : EllipticalSectionProperties
        Elliptical section properties.

    Returns
    -------
    tuple
        ``(V_ed_critical, point_critical, s_edge)`` — the elastic-limit shear (N), 
        the index of the critical perimeter point and the quadrant's
        edge-stress distribution at that shear (MPa).
    """
    N_th, _, M_th = perform_rsm(sector, V_ed_values, N_T_values, props, vectorised=True)
    z = calculate_zep(sector, N_th, 1, props, elastic_mode=True)
    *_, r = calculate_moment_capacity(
        sector, z, V_ed_values, N_T_values, 1, props, elastic_mode=True
    )

    _, (V_idxs, _, _) = find_optimal_r(sector, r, props, elastic_mode=True)

    V_ed_at_limit = V_ed_values[V_idxs]
    point_critical = int(np.argmin(V_ed_at_limit))
    V_ed_critical = float(V_ed_at_limit[point_critical])
    V_ed_critical_idx = V_idxs[point_critical]

    N_crit = N_th[NPOINTS, V_ed_critical_idx]
    M_crit = M_th[NPOINTS, V_ed_critical_idx]
    s_edge = compute_elastic_stress(sector, N_crit, M_crit, props)

    return V_ed_critical, point_critical, s_edge


def point_coordinates(props, index, quadrant):
    """Coordinates of a perimeter point, mirrored into the given quadrant.

    Parameters
    ----------
    props : EllipticalSectionProperties
        Elliptical section properties.
    index : int
        Index of the point on the quarter perimeter, 0 to 90.
    quadrant : int
        Quadrant, 1 to 4.

    Returns
    -------
    tuple of float
        ``(x, y)`` in mm from the centre of the opening.
    """
    x = float(props.x_points[index])
    y = float(props.y_points[index])
    signs = {1: (-1, +1), 2: (+1, +1), 3: (+1, -1), 4: (-1, -1)}
    sx, sy = signs[quadrant]
    return sx * x, sy * y


def perimeter_coordinates_360(props):
    """Perimeter point coordinates around the whole opening.

    Mirrors the quarter-perimeter points into the four quadrants in the same
    order as the edge-stress distributions, so that a stress value and its
    location correspond index for index. Required for plotting stresses around
    an elliptical opening, where the location cannot be inferred from an angle.

    Parameters
    ----------
    props : EllipticalSectionProperties
        Elliptical section properties.

    Returns
    -------
    tuple of numpy.ndarray
        ``(x_360, y_360)``, each of shape ``(360,)``, in mm from the centre of
        the opening.
    """
    x, y = props.x_points, props.y_points

    x_Q1, y_Q1 = -x, +y
    x_Q2, y_Q2 = +x, +y
    x_Q3, y_Q3 = +x, -y
    x_Q4, y_Q4 = -x, -y

    x_360 = np.concatenate([x_Q1[-2::-1], x_Q2[1:], x_Q3[-2::-1], x_Q4[1:]])
    y_360 = np.concatenate([y_Q1[-2::-1], y_Q2[1:], y_Q3[-2::-1], y_Q4[1:]])
    return x_360, y_360
