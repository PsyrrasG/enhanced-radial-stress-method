"""
Enhanced RSM — Mode EL1 (Elastic Limit, lower moment side).

Determines the elastic limit of a perforated section with a circular web
opening: the shear force at which the most critical inclined plane first reaches
yield, which occurs on the lower moment side (LMS). The higher moment side (HMS)
stresses are then evaluated at that shear so the full 360-degree edge-stress
distribution can be assembled.

The numerical kernels (geometry, force equilibrium, neutral-axis solve, edge
stress) are imported from :mod:`enhanced_rsm.core`; this module only orchestrates
them for the EL1 analysis and packages the result.
"""

import time
from typing import NamedTuple

import numpy as np

from .core import (
    build_section_properties,
    perform_rsm,
    compute_elastic_stress,
    solve_elastic_limit,
)


class EL1Result(NamedTuple):
    """Result of an EL1 (elastic limit) analysis.

    Attributes
    ----------
    duration : float
        Wall-clock analysis time (s).
    V_ed_EL1 : float
        Elastic-limit shear force on the lower moment side (kN).
    M_ed_EL1 : float
        Corresponding global moment at the centre-line (kNm).
    theta_critical_Q1, theta_critical_Q2, theta_critical_Q3, theta_critical_Q4 : int
        Critical plane angle in each quadrant (degrees from the vertical).
    s_edge_max_Q1, s_edge_max_Q2, s_edge_max_Q3, s_edge_max_Q4 : float
        Peak edge stress in each quadrant at the elastic limit (MPa).
    s_edge_360 : numpy.ndarray
        Edge-stress distribution around the full opening perimeter (MPa),
        tension positive.
    """

    duration: float
    V_ed_EL1: float
    M_ed_EL1: float
    theta_critical_Q1: int
    theta_critical_Q2: int
    theta_critical_Q3: int
    theta_critical_Q4: int
    s_edge_max_Q1: float
    s_edge_max_Q2: float
    s_edge_max_Q3: float
    s_edge_max_Q4: float
    s_edge_360: np.ndarray


def run_mode_el1(h, h_o, b_f, t_w, t_f, r, f_y, M_V_Ratio):
    """Run an EL1 (elastic limit) analysis.

    Parameters
    ----------
    h : float
        Section depth (mm).
    h_o : float
        Opening diameter (mm).
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
    EL1Result
        The elastic-limit results.
    """
    start = time.perf_counter()

    props, _, _, max_Ved = build_section_properties(h, h_o, b_f, t_w, t_f, r, f_y)

    # Trial shears (N) and the axial force each induces via global bending.
    V_ed_values = np.arange(1e3, max_Ved * 1e3, 1e3)
    M_ed_values = V_ed_values * M_V_Ratio * 1e3
    N_T_values = M_ed_values * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam

    # Upper Tee — Q1 (LMS): the elastic limit.
    V_ed_EL1_UT, theta_Q1, s_edge_Q1 = solve_elastic_limit(1, V_ed_values, N_T_values, props)
    M_ed_EL1_UT = V_ed_EL1_UT * (M_V_Ratio * 1e3)
    N_T_EL1_UT = M_ed_EL1_UT * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam
    s_edge_max_Q1 = float(s_edge_Q1[theta_Q1])

    # Upper Tee — Q2 (HMS) at the elastic-limit shear.
    N_Q2, _, M_Q2 = perform_rsm(2, V_ed_EL1_UT, N_T_EL1_UT, props, vectorised=False)
    s_edge_Q2 = compute_elastic_stress(2, N_Q2, M_Q2, props)
    theta_Q2 = int(np.argmax(s_edge_Q2))
    s_edge_max_Q2 = float(s_edge_Q2[theta_Q2])

    # Bottom Tee — Q4 (LMS).
    V_ed_EL1_BT, theta_Q4, s_edge_Q4 = solve_elastic_limit(4, V_ed_values, N_T_values, props)
    s_edge_max_Q4 = float(s_edge_Q4[theta_Q4])

    # Bottom Tee — Q3 (HMS) at the elastic-limit shear.
    N_Q3, _, M_Q3 = perform_rsm(3, V_ed_EL1_UT, N_T_EL1_UT, props, vectorised=False)
    s_edge_Q3 = compute_elastic_stress(3, N_Q3, M_Q3, props)
    theta_Q3 = int(np.argmax(np.abs(s_edge_Q3)))
    s_edge_max_Q3 = float(s_edge_Q3[theta_Q3])

    # Assemble the full 360-degree edge-stress distribution.
    s_edge_360 = np.concatenate([
        s_edge_Q1[-2::-1], s_edge_Q2[1:],
        s_edge_Q3[-2::-1], s_edge_Q4[1:],
    ])

    duration = time.perf_counter() - start

    return EL1Result(
        duration=duration,
        V_ed_EL1=V_ed_EL1_UT / 1e3,
        M_ed_EL1=M_ed_EL1_UT / 1e6,
        theta_critical_Q1=theta_Q1,
        theta_critical_Q2=theta_Q2,
        theta_critical_Q3=theta_Q3,
        theta_critical_Q4=theta_Q4,
        s_edge_max_Q1=s_edge_max_Q1,
        s_edge_max_Q2=s_edge_max_Q2,
        s_edge_max_Q3=s_edge_max_Q3,
        s_edge_max_Q4=s_edge_max_Q4,
        s_edge_360=s_edge_360,
    )


if __name__ == "__main__":
    # Demonstration run (UB 457x152x52, S355, 75%h circular opening, M/V = 1.333).
    result = run_mode_el1(
        h=449.8, h_o=337.35, b_f=152.4, t_w=7.6, t_f=10.9, r=10.2,
        f_y=355, M_V_Ratio=1.333,
    )
    print("Mode EL1 - Elastic Limit")
    print(f"  duration        : {result.duration:.3f} s")
    print(f"  V_ed (EL1, LMS) : {result.V_ed_EL1:.0f} kN")
    print(f"  M_ed (EL1)      : {result.M_ed_EL1:.0f} kNm")
    print(f"  Q1  theta={result.theta_critical_Q1:>2d} deg  s_edge_max={result.s_edge_max_Q1:7.1f} MPa")
    print(f"  Q2  theta={result.theta_critical_Q2:>2d} deg  s_edge_max={result.s_edge_max_Q2:7.1f} MPa")
    print(f"  Q3  theta={result.theta_critical_Q3:>2d} deg  s_edge_max={result.s_edge_max_Q3:7.1f} MPa")
    print(f"  Q4  theta={result.theta_critical_Q4:>2d} deg  s_edge_max={result.s_edge_max_Q4:7.1f} MPa")
    print(f"  edge-stress distribution: {len(result.s_edge_360)} values")
