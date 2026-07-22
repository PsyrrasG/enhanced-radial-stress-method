"""
Enhanced RSM — Mode EL2 (Elastic Limit, higher moment side).

Determines the shear force at which the higher moment side (HMS) of a circular
web opening first reaches yield. This is the companion to the lower-moment-side
elastic limit returned by mode EL1.

By the time the HMS reaches its elastic limit the lower moment side (LMS) has
already yielded and developed a degree of plasticity, so the analysis also
solves the LMS over the plasticity grid at that same shear and reports how far
it has yielded. If that plasticity reaches 100 per cent the premise of a purely
elastic HMS limit is no longer consistent — the LMS can develop no further
resistance and equilibrium requires redistribution. This module reports that
condition through :attr:`EL2Result.redistribution_required`; deciding what to do
about it is left to the caller.

The numerical kernels are imported from :mod:`enhanced_rsm.core`; this module
only orchestrates them for the EL2 analysis and packages the result.
"""

import time
from typing import NamedTuple

import numpy as np

from .core import (
    build_section_properties,
    perform_rsm,
    calculate_zep,
    limit_web_plasticity,
    calculate_moment_capacity,
    compute_elastic_stress,
    find_optimal_r,
    solve_elastic_limit,
)


class EL2Result(NamedTuple):
    """Result of an EL2 (elastic limit, higher moment side) analysis.

    Attributes
    ----------
    duration : float
        Wall-clock analysis time (s).
    V_ed_EL2 : float
        Elastic-limit shear force on the higher moment side (kN).
    M_ed_EL2 : float
        Corresponding global moment at the centre-line (kNm).
    theta_critical_Q2, theta_critical_Q3 : int
        Critical plane angle on the HMS quadrants (degrees).
    s_edge_max_Q2, s_edge_max_Q3 : float
        Peak edge stress on the HMS quadrants at the elastic limit (MPa).
    theta_critical_Q1, theta_critical_Q4 : int
        Critical plane angle on the LMS quadrants at that shear (degrees).
    pl_ratio_Q1, pl_ratio_Q4 : float
        Proportion of the LMS web already yielded at that shear (per cent).
    n_Q1, n_Q4 : float
        Plasticity strain factor at the LMS critical plane.
    s_edge_max_Q1, s_edge_max_Q4 : float
        Peak edge stress on the LMS quadrants at that shear (MPa).
    redistribution_required : bool
        True when the LMS has fully yielded, so that the HMS cannot reach its
        elastic limit without moment redistribution.
    s_edge_360 : numpy.ndarray
        Edge-stress distribution around the opening (MPa, clipped to +/- f_y).
    """

    duration: float
    V_ed_EL2: float
    M_ed_EL2: float
    theta_critical_Q2: int
    theta_critical_Q3: int
    s_edge_max_Q2: float
    s_edge_max_Q3: float
    theta_critical_Q1: int
    theta_critical_Q4: int
    pl_ratio_Q1: float
    pl_ratio_Q4: float
    n_Q1: float
    n_Q4: float
    s_edge_max_Q1: float
    s_edge_max_Q4: float
    redistribution_required: bool
    s_edge_360: np.ndarray


def _solve_lms_plasticity(sector, Ved, N_T, props, n_expanded, n_values):
    """Plasticity already developed on an LMS quadrant at a prescribed shear.

    Solves the quadrant over the plasticity grid at a fixed shear and returns the
    plane at which the moment ratio converges with the greatest yielded web
    proportion.

    Parameters
    ----------
    sector : int
        LMS quadrant, 1 (upper Tee) or 4 (bottom Tee).
    Ved : float
        Applied shear (N).
    N_T : float
        Axial force in the Tee at that shear (N).
    props : SectionProperties
        Shared section properties.
    n_expanded : numpy.ndarray
        Plasticity grid reshaped for broadcasting, shape ``(1, 1, n)``.
    n_values : numpy.ndarray
        Plasticity grid, shape ``(n,)``.

    Returns
    -------
    tuple
        ``(theta_critical, pl_ratio, n, s_edge, s_edge_max)``.

    Raises
    ------
    RuntimeError
        If the moment ratio does not converge on any plane.
    """
    if sector not in (1, 4):
        raise ValueError("_solve_lms_plasticity() applies to the LMS (Q1, Q4).")

    N_th, _, M_th = perform_rsm(sector, Ved, N_T, props, vectorised=False)
    s_edge = compute_elastic_stress(sector, N_th, M_th, props)
    s_edge_max = float(s_edge[int(np.argmax(np.abs(s_edge)))])

    z = calculate_zep(sector, N_th, n_expanded, props, elastic_mode=False)
    _, pl_ratio = limit_web_plasticity(z, n_expanded, props)
    pl_ratio_values = pl_ratio.squeeze()
    *_, r = calculate_moment_capacity(
        sector, z, Ved, N_T, n_expanded, props, elastic_mode=False
    )
    r_values = r.squeeze()

    converged, (n_idxs, _, pl_ratio_converged) = find_optimal_r(
        sector, r_values, props, elastic_mode=False, pl_ratio_values=pl_ratio_values
    )
    if not converged:
        raise RuntimeError(
            f"Q{sector}: the moment ratio does not converge to any plasticity level."
        )

    theta_critical = int(np.argmax(pl_ratio_converged))
    return (
        theta_critical,
        float(pl_ratio_converged[theta_critical]),
        float(n_values[n_idxs[theta_critical]]),
        s_edge,
        s_edge_max,
    )


def run_mode_el2(h, h_o, b_f, t_w, t_f, r, f_y, M_V_Ratio, max_n=20):
    """Run an EL2 (elastic limit, higher moment side) analysis.

    Parameters
    ----------
    h, h_o, b_f, t_w, t_f, r, f_y : float
        Section depth, opening diameter, flange width, web thickness, flange
        thickness, root radius (mm) and yield strength (MPa).
    M_V_Ratio : float
        Global moment-to-shear ratio at the opening centre-line (m).
    max_n : float, optional
        Upper bound of the plasticity strain factor grid (default 20).

    Returns
    -------
    EL2Result
        The higher-moment-side elastic-limit results.

    Raises
    ------
    RuntimeError
        If the upper and bottom Tees return different elastic limits, which
        would indicate an asymmetric section or a numerical problem.
    """
    start = time.perf_counter()

    props, _, _, max_Ved = build_section_properties(h, h_o, b_f, t_w, t_f, r, f_y)

    V_ed_values = np.arange(1e3, max_Ved * 1e3, 1e3)
    M_ed_values = V_ed_values * M_V_Ratio * 1e3
    N_T_values = M_ed_values * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam

    n_values = np.arange(1.01, max_n + 0.01, 0.01)
    n_expanded = n_values[np.newaxis, np.newaxis, :]

    # Upper Tee — Q2 (HMS): the elastic limit.
    V_ed_EL2_UT, theta_Q2, s_edge_Q2 = solve_elastic_limit(2, V_ed_values, N_T_values, props)
    M_ed_EL2_UT = V_ed_EL2_UT * (M_V_Ratio * 1e3)
    N_T_EL2_UT = M_ed_EL2_UT * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam
    s_edge_max_Q2 = float(s_edge_Q2[theta_Q2])

    # Upper Tee — Q1 (LMS): plasticity already developed at that shear.
    theta_Q1, pl_ratio_Q1, n_Q1, s_edge_Q1, s_edge_max_Q1 = _solve_lms_plasticity(
        1, V_ed_EL2_UT, N_T_EL2_UT, props, n_expanded, n_values
    )

    # Bottom Tee — Q3 (HMS).
    V_ed_EL2_BT, theta_Q3, s_edge_Q3 = solve_elastic_limit(3, V_ed_values, N_T_values, props)
    s_edge_max_Q3 = float(s_edge_Q3[theta_Q3])

    if V_ed_EL2_UT != V_ed_EL2_BT:
        raise RuntimeError(
            "Different EL2 elastic limits obtained for the upper and bottom Tees "
            f"({V_ed_EL2_UT / 1e3:.0f} kN vs {V_ed_EL2_BT / 1e3:.0f} kN)."
        )

    # Bottom Tee — Q4 (LMS).
    theta_Q4, pl_ratio_Q4, n_Q4, s_edge_Q4, s_edge_max_Q4 = _solve_lms_plasticity(
        4, V_ed_EL2_UT, N_T_EL2_UT, props, n_expanded, n_values
    )

    # The LMS cannot develop further resistance once fully yielded: the HMS can
    # then only reach its elastic limit through moment redistribution.
    redistribution_required = bool(pl_ratio_Q1 >= 100.0)

    s_edge_360 = np.clip(
        np.concatenate([
            s_edge_Q1[-2::-1], s_edge_Q2[1:],
            s_edge_Q3[-2::-1], s_edge_Q4[1:],
        ]),
        -f_y, f_y,
    )

    duration = time.perf_counter() - start

    return EL2Result(
        duration=duration,
        V_ed_EL2=V_ed_EL2_UT / 1e3,
        M_ed_EL2=M_ed_EL2_UT / 1e6,
        theta_critical_Q2=theta_Q2,
        theta_critical_Q3=theta_Q3,
        s_edge_max_Q2=s_edge_max_Q2,
        s_edge_max_Q3=s_edge_max_Q3,
        theta_critical_Q1=theta_Q1,
        theta_critical_Q4=theta_Q4,
        pl_ratio_Q1=pl_ratio_Q1,
        pl_ratio_Q4=pl_ratio_Q4,
        n_Q1=n_Q1,
        n_Q4=n_Q4,
        s_edge_max_Q1=s_edge_max_Q1,
        s_edge_max_Q4=s_edge_max_Q4,
        redistribution_required=redistribution_required,
        s_edge_360=s_edge_360,
    )


if __name__ == "__main__":
    # Demonstration run (UB 457x152x52, S355, 75%h circular opening, M/V = 2.333).
    result = run_mode_el2(
        h=449.8, h_o=337.35, b_f=152.4, t_w=7.6, t_f=10.9, r=10.2,
        f_y=355, M_V_Ratio=2.333,
    )
    print("Mode EL2 - Elastic Limit (higher moment side)")
    print(f"  duration        : {result.duration:.3f} s")
    print(f"  V_ed (EL2, HMS) : {result.V_ed_EL2:.0f} kN")
    print(f"  M_ed (EL2)      : {result.M_ed_EL2:.0f} kNm")
    print(f"  Q2  theta={result.theta_critical_Q2:>2d} deg  s_edge_max={result.s_edge_max_Q2:7.1f} MPa")
    print(f"  Q3  theta={result.theta_critical_Q3:>2d} deg  s_edge_max={result.s_edge_max_Q3:7.1f} MPa")
    print(f"  Q1  theta={result.theta_critical_Q1:>2d} deg  pl_ratio={result.pl_ratio_Q1:6.2f}%  n={result.n_Q1:.2f}  s_edge_max={result.s_edge_max_Q1:7.1f} MPa")
    print(f"  Q4  theta={result.theta_critical_Q4:>2d} deg  pl_ratio={result.pl_ratio_Q4:6.2f}%  n={result.n_Q4:.2f}  s_edge_max={result.s_edge_max_Q4:7.1f} MPa")
    print(f"  redistribution required: {result.redistribution_required}")
    print(f"  edge-stress distribution: {len(result.s_edge_360)} values")
