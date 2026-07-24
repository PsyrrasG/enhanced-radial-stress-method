"""
Enhanced RSM — Elastic Limit Mode (LMS) - EL1 for elliptical openings.

The elliptical counterpart of :mod:`enhanced_rsm.el1`. The analysis is the same:
the shear force at which the most critical inclined plane first reaches yield,
which occurs on the lower moment side. Only the geometry differs, and that is
handled by :mod:`enhanced_rsm.core_elliptical`.

The critical location is reported as the index of a point on the perimeter,
together with its coordinates, rather than as an angle. The perimeter
coordinates are returned alongside the edge stresses so that a stress and its
location correspond index for index.
"""

import time
from typing import NamedTuple, Tuple

import numpy as np

from .core_elliptical import (
    build_elliptical_section_properties,
    perform_rsm,
    compute_elastic_stress,
    solve_elastic_limit,
    point_coordinates,
    perimeter_coordinates_360,
)


class EL1EllipticalResult(NamedTuple):
    """Result of an EL1 analysis for an elliptical opening.

    Attributes
    ----------
    duration : float
        Wall-clock analysis time (s).
    a_b_ratio : float
        Ratio of the horizontal to the vertical axis of the opening.
    ellipse_type : str
        ``'Horizontal'`` or ``'Vertical'``, according to which axis is longer.
    V_ed_EL1 : float
        Elastic-limit shear force on the lower moment side (kN).
    M_ed_EL1 : float
        Corresponding global moment at the centre-line (kNm).
    point_critical_Q1, point_critical_Q2, point_critical_Q3, point_critical_Q4 : int
        Index of the critical perimeter point in each quadrant, from the
        vertical centre-line (0) to the horizontal (90).
    coords_critical_Q1, coords_critical_Q2, coords_critical_Q3, coords_critical_Q4 : tuple of float
        Coordinates ``(x, y)`` of the critical point in each quadrant, in mm
        from the centre of the opening.
    s_edge_max_Q1, s_edge_max_Q2, s_edge_max_Q3, s_edge_max_Q4 : float
        Peak edge stress in each quadrant at the elastic limit (MPa).
    s_edge_360 : numpy.ndarray
        Edge-stress distribution around the opening (MPa), tension positive.
    x_360, y_360 : numpy.ndarray
        Coordinates of the perimeter points corresponding index for index to
        ``s_edge_360`` (mm from the centre of the opening).
    """

    duration: float
    a_b_ratio: float
    ellipse_type: str
    V_ed_EL1: float
    M_ed_EL1: float
    point_critical_Q1: int
    point_critical_Q2: int
    point_critical_Q3: int
    point_critical_Q4: int
    coords_critical_Q1: Tuple[float, float]
    coords_critical_Q2: Tuple[float, float]
    coords_critical_Q3: Tuple[float, float]
    coords_critical_Q4: Tuple[float, float]
    s_edge_max_Q1: float
    s_edge_max_Q2: float
    s_edge_max_Q3: float
    s_edge_max_Q4: float
    s_edge_360: np.ndarray
    x_360: np.ndarray
    y_360: np.ndarray


def run_mode_el1_elliptical(h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y, M_V_Ratio):
    """Run an EL1 (elastic limit) analysis for an elliptical opening.

    Parameters
    ----------
    h : float
        Section depth (mm).
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
    M_V_Ratio : float
        Global moment-to-shear ratio at the opening centre-line (m).

    Returns
    -------
    EL1EllipticalResult
        The elastic-limit results.
    """
    start = time.perf_counter()

    props, _, _, max_Ved = build_elliptical_section_properties(
        h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y
    )

    # Trial shears (N) and the axial force each induces via global bending.
    V_ed_values = np.arange(1e3, max_Ved * 1e3, 1e3)
    M_ed_values = V_ed_values * M_V_Ratio * 1e3
    N_T_values = M_ed_values * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam

    # Upper Tee — Q1 (LMS): the elastic limit.
    V_ed_EL1_UT, point_Q1, s_edge_Q1 = solve_elastic_limit(
        1, V_ed_values, N_T_values, props
    )
    M_ed_EL1_UT = V_ed_EL1_UT * (M_V_Ratio * 1e3)
    N_T_EL1_UT = M_ed_EL1_UT * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam
    s_edge_max_Q1 = float(s_edge_Q1[point_Q1])

    # Upper Tee — Q2 (HMS) at the elastic-limit shear.
    N_Q2, _, M_Q2 = perform_rsm(2, V_ed_EL1_UT, N_T_EL1_UT, props, vectorised=False)
    s_edge_Q2 = compute_elastic_stress(2, N_Q2, M_Q2, props)
    point_Q2 = int(np.argmax(s_edge_Q2))
    s_edge_max_Q2 = float(s_edge_Q2[point_Q2])

    # Bottom Tee — Q4 (LMS).
    V_ed_EL1_BT, point_Q4, s_edge_Q4 = solve_elastic_limit(
        4, V_ed_values, N_T_values, props
    )
    s_edge_max_Q4 = float(s_edge_Q4[point_Q4])

    # Bottom Tee — Q3 (HMS) at the elastic-limit shear.
    N_Q3, _, M_Q3 = perform_rsm(3, V_ed_EL1_UT, N_T_EL1_UT, props, vectorised=False)
    s_edge_Q3 = compute_elastic_stress(3, N_Q3, M_Q3, props)
    point_Q3 = int(np.argmax(np.abs(s_edge_Q3)))
    s_edge_max_Q3 = float(s_edge_Q3[point_Q3])

    # Assemble the full edge-stress distribution and the matching coordinates.
    s_edge_360 = np.concatenate([
        s_edge_Q1[-2::-1], s_edge_Q2[1:],
        s_edge_Q3[-2::-1], s_edge_Q4[1:],
    ])
    x_360, y_360 = perimeter_coordinates_360(props)

    duration = time.perf_counter() - start

    return EL1EllipticalResult(
        duration=duration,
        a_b_ratio=a_b_ratio,
        ellipse_type=props.ellipse_type,
        V_ed_EL1=V_ed_EL1_UT / 1e3,
        M_ed_EL1=M_ed_EL1_UT / 1e6,
        point_critical_Q1=point_Q1,
        point_critical_Q2=point_Q2,
        point_critical_Q3=point_Q3,
        point_critical_Q4=point_Q4,
        coords_critical_Q1=point_coordinates(props, point_Q1, 1),
        coords_critical_Q2=point_coordinates(props, point_Q2, 2),
        coords_critical_Q3=point_coordinates(props, point_Q3, 3),
        coords_critical_Q4=point_coordinates(props, point_Q4, 4),
        s_edge_max_Q1=s_edge_max_Q1,
        s_edge_max_Q2=s_edge_max_Q2,
        s_edge_max_Q3=s_edge_max_Q3,
        s_edge_max_Q4=s_edge_max_Q4,
        s_edge_360=s_edge_360,
        x_360=x_360,
        y_360=y_360,
    )


if __name__ == "__main__":
    # Demonstration run (UB 457x152x52, S355, 75%h elliptical opening,
    # axis ratio a/b = 1.5, M/V = 1.333).
    result = run_mode_el1_elliptical(
        h=449.8, h_o=337.35, a_b_ratio=1.5, b_f=152.4, t_w=7.6, t_f=10.9,
        r=10.2, f_y=355, M_V_Ratio=1.333,
    )
    print("Elastic Limit Mode - EL1 (elliptical opening)")
    print(f"  duration        : {result.duration:.3f} s")
    print(f"  a/b ratio       : {result.a_b_ratio}  ({result.ellipse_type})")
    print(f"  V_ed (EL1)      : {result.V_ed_EL1:.0f} kN")
    print(f"  M_ed (EL1)      : {result.M_ed_EL1:.0f} kNm")
    for q in (1, 2, 3, 4):
        pt = getattr(result, f"point_critical_Q{q}")
        x, y = getattr(result, f"coords_critical_Q{q}")
        s_max = getattr(result, f"s_edge_max_Q{q}")
        print(f"  Q{q}  point={pt:>2d}  coords=({x:7.1f},{y:6.1f})  s_edge_max={s_max:7.1f} MPa")
    print(f"  edge-stress distribution: {len(result.s_edge_360)} values")