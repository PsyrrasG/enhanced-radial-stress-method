"""
Enhanced RSM — Mode Given Forces.

Interrogates a circular web opening at a prescribed pair of applied forces
rather than searching for a limiting shear. If the applied shear lies below the
elastic limit the response is elastic and the elastic stress state is reported.
Otherwise the section is solved once at those forces — with no shear sweep — to
report the degree of plasticity on both sides of the opening, and moment is
redistributed from the lower moment side (LMS) to the higher moment side (HMS)
if the LMS has fully yielded.

This is a diagnostic mode: it reports the state of the opening under the forces
given. Whether those forces lie within the section's capacity is a separate
question, answered by mode PLCAP; this module flags the conditions it detects
(``total_failure``, ``exceeds_tee_resistance``) but does not stop on them.

The numerical kernels are imported from :mod:`enhanced_rsm.core_elliptical`; the Tee
resistance comes from :mod:`enhanced_rsm.tee_bending_resistance`. The elastic limit is
obtained from :mod:`enhanced_rsm.el1_elliptical` unless supplied explicitly.
"""

import time
from typing import NamedTuple, Optional

import numpy as np

from .core_elliptical import (
    NPOINTS,
    build_elliptical_section_properties,
    perform_rsm,
    calculate_zep,
    limit_web_plasticity,
    calculate_moment_capacity,
    compute_elastic_stress,
    find_optimal_r,
    redistribute_moment,
)
from .tee_bending_resistance import analyze_circular_opening
from .el1_elliptical import run_mode_el1_elliptical


class GivenForcesEllipticalResult(NamedTuple):
    """Result of a Given Forces analysis.

    Plasticity fields are ``None`` when the response is elastic; redistribution
    fields are ``None`` when no redistribution occurred.

    Attributes
    ----------
    duration : float
        Wall-clock analysis time (s).
    V_ed, M_ed : float
        The applied shear (kN) and moment (kNm) interrogated.
    V_ed_EL1 : float
        Elastic limit used for the elastic/plastic decision (kN).
    elastic : bool
        True when the applied shear lies below the elastic limit.
    point_LMS, point_HMS : int
        Critical plane angle on each side (degrees). In the elastic branch these
        are the planes of peak elastic stress.
    pl_ratio_LMS, n_LMS : float or None
        Yielded web proportion (per cent) and strain factor on the LMS.
    pl_ratio_HMS, n_HMS : float or None
        Yielded web proportion (per cent) and strain factor on the HMS.
    redistribution : bool
        True when moment was redistributed from the LMS to the HMS.
    DM_T : float or None
        Moment redistributed (kNm).
    point_HMS_RE, pl_ratio_HMS_RE, n_HMS_RE : int or None, float or None, float or None
        Higher moment side state after redistribution.
    total_failure : bool
        True when the HMS had already reached full plasticity at these forces.
    exceeds_tee_resistance : bool
        True when the redistributed moment exceeds the Tee's plastic bending
        resistance at the opening centre-line, which should not occur below the
        plastic capacity.
    s_edge_360 : numpy.ndarray
        Edge-stress distribution around the opening (MPa). Clipped to +/- f_y in
        the plastic branch; unclipped in the elastic branch.
    """

    duration: float
    V_ed: float
    M_ed: float
    V_ed_EL1: float
    elastic: bool
    point_LMS: int
    point_HMS: int
    pl_ratio_LMS: Optional[float]
    n_LMS: Optional[float]
    pl_ratio_HMS: Optional[float]
    n_HMS: Optional[float]
    redistribution: bool
    DM_T: Optional[float]
    point_HMS_RE: Optional[int]
    pl_ratio_HMS_RE: Optional[float]
    n_HMS_RE: Optional[float]
    total_failure: bool
    exceeds_tee_resistance: bool
    s_edge_360: np.ndarray


def _make_initial_state():
    """Return the per-Tee state dictionary for a Given Forces analysis."""
    return {
        "point_LMS": None,
        "pl_ratio_LMS": None,
        "n_LMS": None,
        "point_HMS": None,
        "pl_ratio_HMS": None,
        "n_HMS": None,
        "redistribution": False,
        "DM_T": None,
        "point_HMS_RE": None,
        "pl_ratio_HMS_RE": None,
        "n_HMS_RE": None,
        "total_failure": False,
        "exceeds_tee_resistance": False,
    }


def _solve_sector_plasticity(sector, N_Q, Ved, N_T, props, n_expanded, n_values):
    """Solve one quadrant over the plasticity grid at a fixed pair of forces.

    Unlike the plastic-capacity sweep the shear is constant here, so the search
    returns the plane with the greatest yielded web proportion at which the
    moment ratio converges.

    Returns
    -------
    tuple
        ``(M_Rd_tot, M_Ed, r_values, pl_ratio_values, info)`` where ``info`` is a
        dict of the critical plane, or ``None`` if the quadrant is still elastic.
    """
    z = calculate_zep(sector, N_Q, n_expanded, props, elastic_mode=False)
    _, pl_ratio = limit_web_plasticity(z, n_expanded, props)
    _, _, M_Rd_tot, M_Ed, r = calculate_moment_capacity(
        sector, z, Ved, N_T, n_expanded, props, elastic_mode=False
    )

    r_values = r.squeeze()
    pl_ratio_values = pl_ratio.squeeze()

    converged, (n_idxs, _, pl_ratio_converged) = find_optimal_r(
        sector, r_values, props, elastic_mode=False, pl_ratio_values=pl_ratio_values
    )

    if converged:
        point = int(np.argmax(pl_ratio_converged))
        info = {
            "sector": sector,
            "point": point,
            "pl_ratio": float(pl_ratio_converged[point]),
            "n": float(n_values[n_idxs[point]]),
        }
    else:
        info = None  # Still elastic in this quadrant.

    return M_Rd_tot, M_Ed, r_values, pl_ratio_values, info


def _compute_tee_state(LMS_sector, HMS_sector, state, Ved, N_T, N_Q_map, props,
                       n_expanded, n_values, M_V_Ratio):
    """Evaluate one Tee (an LMS/HMS pair) at the prescribed forces.

    Mutates ``state`` in place. Redistribution is applied only when the LMS has
    fully yielded; the checks that would terminate a plastic-capacity sweep are
    recorded as flags here rather than stopping the analysis.
    """
    LMS_M_Rd_tot, LMS_M_Ed, _, _, LMS_info = _solve_sector_plasticity(
        LMS_sector, N_Q_map[LMS_sector], Ved, N_T, props, n_expanded, n_values
    )
    HMS_M_Rd_tot, HMS_M_Ed, _, HMS_pl_ratio_values, HMS_info = _solve_sector_plasticity(
        HMS_sector, N_Q_map[HMS_sector], Ved, N_T, props, n_expanded, n_values
    )

    if LMS_info:
        state["point_LMS"] = LMS_info["point"]
        state["pl_ratio_LMS"] = LMS_info["pl_ratio"]
        state["n_LMS"] = LMS_info["n"]

    if HMS_info:
        state["point_HMS"] = HMS_info["point"]
        state["pl_ratio_HMS"] = HMS_info["pl_ratio"]
        state["n_HMS"] = HMS_info["n"]

    if HMS_info and HMS_info["pl_ratio"] >= 100.0:
        # The higher moment side has already failed at these forces.
        state["total_failure"] = True
        return state

    if LMS_info and LMS_info["pl_ratio"] >= 100.0:
        DM_T, _, _, _, _, HMS_r_after = redistribute_moment(
            Ved, LMS_M_Rd_tot, LMS_M_Ed, HMS_M_Rd_tot, HMS_M_Ed
        )
        converged_RE, (HMS_n_idxs, _, HMS_pl_ratio_converged) = find_optimal_r(
            HMS_sector, HMS_r_after, props, elastic_mode=False,
            pl_ratio_values=HMS_pl_ratio_values,
        )

        if converged_RE:
            optimal_DM_T = DM_T[NPOINTS, HMS_n_idxs]
            max_point = int(np.argmax(HMS_pl_ratio_converged))
            max_DM_T = optimal_DM_T[max_point]

            M_T_Rd = analyze_circular_opening(
                d_o=props.h_o, h=props.h, b_f=props.b_f, t_f=props.t_f,
                t_w=props.t_w, r=props.r, f_y=props.f_y,
                moment_shear_ratio=M_V_Ratio, V_Ed=Ved / 1e3,
            ) * 1e6
            if max_DM_T > M_T_Rd:
                # Should not occur below the plastic capacity; recorded, not fatal.
                state["exceeds_tee_resistance"] = True

            state["redistribution"] = True
            state["DM_T"] = float(max_DM_T)
            state["point_HMS_RE"] = max_point
            state["pl_ratio_HMS_RE"] = float(HMS_pl_ratio_converged[max_point])
            state["n_HMS_RE"] = float(n_values[HMS_n_idxs[max_point]])

    return state


def run_mode_given_forces_elliptical(h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y, Ved, Med,
                          Ved_EL1=None, max_n=20):
    """Run a Given Forces analysis at a prescribed shear and moment.

    Parameters
    ----------
    h, h_o, b_f, t_w, t_f, r, f_y : float
        Section depth, opening height (the vertical axis), flange width, web
        thickness, flange thickness, root radius (mm) and yield strength (MPa).
    a_b_ratio : float
        Ratio of the horizontal to the vertical axis of the opening.
    Ved : float
        Applied vertical shear at the opening centre-line (kN).
    Med : float
        Applied global moment at the opening centre-line (kNm).
    Ved_EL1 : float, optional
        Elastic limit (kN). Computed from an EL1 analysis if omitted.
    max_n : float, optional
        Upper bound of the plasticity strain factor grid (default 20).

    Returns
    -------
    GivenForcesEllipticalResult
        The state of the opening at the prescribed forces.
    """
    start = time.perf_counter()

    if Ved <= 0:
        raise ValueError("'Ved' must be greater than zero.")

    # Moment-to-shear ratio implied by the prescribed forces (m).
    M_V_Ratio = Med / Ved

    props, _, _, _ = build_elliptical_section_properties(
        h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y
    )

    if Ved_EL1 is None:
        Ved_EL1 = run_mode_el1_elliptical(
            h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y, M_V_Ratio
        ).V_ed_EL1

    Ved_N = Ved * 1e3
    Med_Nmm = Med * 1e6
    N_T = Med_Nmm * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam

    # Internal forces per quadrant at the prescribed forces.
    forces = {
        sector: perform_rsm(sector, Ved_N, N_T, props, vectorised=False)
        for sector in (1, 2, 3, 4)
    }

    # --- Elastic branch --------------------------------------------------
    if Ved < Ved_EL1:
        s_edge = {
            sector: compute_elastic_stress(sector, forces[sector][0], forces[sector][2], props)
            for sector in (1, 2, 3, 4)
        }
        s_edge_360 = np.concatenate([
            s_edge[1][-2::-1], s_edge[2][1:],
            s_edge[3][-2::-1], s_edge[4][1:],
        ])
        duration = time.perf_counter() - start
        return GivenForcesEllipticalResult(
            duration=duration,
            V_ed=Ved, M_ed=Med, V_ed_EL1=Ved_EL1,
            elastic=True,
            point_LMS=int(np.argmax(np.abs(s_edge[1]))),
            point_HMS=int(np.argmax(np.abs(s_edge[2]))),
            pl_ratio_LMS=None, n_LMS=None,
            pl_ratio_HMS=None, n_HMS=None,
            redistribution=False, DM_T=None,
            point_HMS_RE=None, pl_ratio_HMS_RE=None, n_HMS_RE=None,
            total_failure=False, exceeds_tee_resistance=False,
            s_edge_360=s_edge_360,
        )

    # --- Elasto-plastic branch -------------------------------------------
    n_values = np.arange(1.01, max_n + 0.01, 0.01)
    n_expanded = n_values[np.newaxis, np.newaxis, :]
    N_Q_map = {sector: forces[sector][0] for sector in (1, 2, 3, 4)}

    UT_state = _compute_tee_state(
        1, 2, _make_initial_state(), Ved_N, N_T, N_Q_map, props,
        n_expanded, n_values, M_V_Ratio,
    )
    _ = _compute_tee_state(
        4, 3, _make_initial_state(), Ved_N, N_T, N_Q_map, props,
        n_expanded, n_values, M_V_Ratio,
    )

    s_edge = {
        sector: compute_elastic_stress(sector, forces[sector][0], forces[sector][2], props)
        for sector in (1, 2, 3, 4)
    }
    s_edge_360 = np.clip(
        np.concatenate([
            s_edge[1][-2::-1], s_edge[2][1:],
            s_edge[3][-2::-1], s_edge[4][1:],
        ]),
        -f_y, f_y,
    )

    point_LMS = UT_state["point_LMS"]
    if point_LMS is None:
        point_LMS = int(np.argmax(np.abs(s_edge[1])))
    point_HMS = UT_state["point_HMS"]
    if point_HMS is None:
        point_HMS = int(np.argmax(np.abs(s_edge[2])))

    duration = time.perf_counter() - start

    return GivenForcesEllipticalResult(
        duration=duration,
        V_ed=Ved, M_ed=Med, V_ed_EL1=Ved_EL1,
        elastic=False,
        point_LMS=point_LMS,
        point_HMS=point_HMS,
        pl_ratio_LMS=UT_state["pl_ratio_LMS"],
        n_LMS=UT_state["n_LMS"],
        pl_ratio_HMS=UT_state["pl_ratio_HMS"],
        n_HMS=UT_state["n_HMS"],
        redistribution=UT_state["redistribution"],
        DM_T=(UT_state["DM_T"] / 1e6) if UT_state["DM_T"] is not None else None,
        point_HMS_RE=UT_state["point_HMS_RE"],
        pl_ratio_HMS_RE=UT_state["pl_ratio_HMS_RE"],
        n_HMS_RE=UT_state["n_HMS_RE"],
        total_failure=UT_state["total_failure"],
        exceeds_tee_resistance=UT_state["exceeds_tee_resistance"],
        s_edge_360=s_edge_360,
    )


if __name__ == "__main__":
    # Demonstration runs (UB 457x152x52, S355, 75%h elliptical opening,
    # axis ratio a/b = 1.5).
    section = dict(h=449.8, h_o=337.35, a_b_ratio=1.5, b_f=152.4, t_w=7.6,
                   t_f=10.9, r=10.2, f_y=355)
 
    for Ved, Med, label in [(60, 80, "below the elastic limit"),
                            (100, 133, "above the elastic limit")]:
        result = run_mode_given_forces_elliptical(**section, Ved=Ved, Med=Med)
        print(f"Mode Given Forces - V_ed = {result.V_ed:.0f} kN, "
              f"M_ed = {result.M_ed:.0f} kNm ({label})")
        print(f"  duration      : {result.duration:.3f} s")
        print(f"  V_ed (EL1)    : {result.V_ed_EL1:.0f} kN")
        print(f"  response      : {'elastic' if result.elastic else 'elasto-plastic'}")
        if result.elastic:
            print(f"  LMS point={result.point_LMS}   HMS point={result.point_HMS}")
        else:
            print(f"  LMS  point={result.point_LMS}  pl_ratio={result.pl_ratio_LMS:.2f}%  n={result.n_LMS:.2f}")
            if result.pl_ratio_HMS is not None:
                print(f"  HMS  point={result.point_HMS}  pl_ratio={result.pl_ratio_HMS:.2f}%  n={result.n_HMS:.2f}")
            else:
                print(f"  HMS  point={result.point_HMS}  (still elastic)")
            print(f"  redistribution: {result.redistribution}")
            if result.redistribution:
                print(f"    DM_T = {result.DM_T:.2f} kNm")
                print(f"    HMS after: point={result.point_HMS_RE}  "
                      f"pl_ratio={result.pl_ratio_HMS_RE:.2f}%  n={result.n_HMS_RE:.2f}")
            if result.total_failure:
                print("  WARNING: the higher moment side has already fully yielded.")
            if result.exceeds_tee_resistance:
                print("  WARNING: redistribution exceeds the Tee's bending resistance.")
        print(f"  edge-stress distribution: {len(result.s_edge_360)} values\n")
