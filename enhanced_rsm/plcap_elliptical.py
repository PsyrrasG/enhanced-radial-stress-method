"""
Enhanced RSM — Mode PLCAP (Plastic Capacity with moment redistribution).

Determines the plastic capacity of a perforated section with a circular web
opening. The applied shear is swept upward from the elastic limit; at each step
the inclined Tee is solved over the plasticity grid, the first fully-yielded
plane on the lower moment side (LMS) is captured once, and moment is
redistributed to the higher moment side (HMS) until one of two stop criteria is
met: the HMS reaches full plasticity, or the redistributed moment exceeds the
Tee's plastic bending resistance at the opening centre-line.

The numerical kernels are imported from :mod:`enhanced_rsm.core_elliptical`; the Tee
resistance used for the second stop criterion comes from
:mod:`enhanced_rsm.tee_bending_resistance`. The starting shear (elastic limit) is obtained
from :mod:`enhanced_rsm.el1_elliptical` unless supplied explicitly.
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

# Wall-clock guard for the load sweep (s).
_MAX_SWEEP_SECONDS = 90


class PLCAPEllipticalResult(NamedTuple):
    """Result of a PLCAP (plastic capacity) analysis.

    Fields that depend on redistribution having occurred are ``None`` when it did
    not. All shears are in kN, moments in kNm, stresses in MPa and angles in
    degrees.

    Attributes
    ----------
    duration : float
        Wall-clock analysis time (s).
    V_ed_pl, M_ed_pl : float
        Plastic capacity and corresponding moment without redistribution.
    V_ed_pl_RE, M_ed_pl_RE : float
        Plastic capacity and corresponding moment with redistribution.
    V_increase_pct : float
        Percentage increase in capacity due to redistribution.
    DM_T : float or None
        Maximum moment redistributed from LMS to HMS (kNm).
    stop_criterion : str or None
        Which criterion terminated the redistribution.
    point_LMS, pl_ratio_LMS, n_LMS : int, float, float
        Critical plane on the lower moment side (same with/without
        redistribution).
    point_HMS, pl_ratio_HMS, n_HMS : int or None, float or None, float or None
        Higher moment side state at the plastic capacity, before redistribution.
    point_HMS_RE, pl_ratio_HMS_RE, n_HMS_RE : int or None, float or None, float or None
        Higher moment side state after redistribution.
    s_edge_360, s_edge_360_RE : numpy.ndarray
        Edge-stress distributions around the opening (MPa, clipped to +/- f_y),
        without and with redistribution.
    """

    duration: float
    V_ed_pl: float
    M_ed_pl: float
    V_ed_pl_RE: float
    M_ed_pl_RE: float
    V_increase_pct: float
    DM_T: Optional[float]
    stop_criterion: Optional[str]
    point_LMS: int
    pl_ratio_LMS: float
    n_LMS: float
    point_HMS: Optional[int]
    pl_ratio_HMS: Optional[float]
    n_HMS: Optional[float]
    point_HMS_RE: Optional[int]
    pl_ratio_HMS_RE: Optional[float]
    n_HMS_RE: Optional[float]
    s_edge_360: np.ndarray
    s_edge_360_RE: np.ndarray


def _make_initial_state():
    """Return the per-Tee state dictionary that persists across load steps."""
    return {
        "critical_point_found": False,
        "critical_point_completed": False,
        "critical_pl_level_HMS_captured": False,
        # LMS / critical plane:
        "critical_Ved": None,
        "critical_point": None,
        "critical_pl_ratio": None,
        "critical_n": None,
        "critical_sector": None,
        # HMS plasticity at the critical shear:
        "pl_level_max_HMS": None,
        "point_max_HMS": None,
        "n_max_HMS": None,
        # After redistribution:
        "critical_Ved_RE": None,
        "max_redistribution": None,
        "max_point_HMS_RE": None,
        "max_pl_ratio_HMS_RE": None,
        "max_HMS_optimal_r_value": None,
        "max_n_idx_HMS_RE": None,
        "max_n_HMS_RE": None,
        "redistribution_stop_criterion": None,
    }


def _perform_enhanced_rsm(sector, N_Q, Ved, N_T, props, n_expanded, n_values,
                          critical_point_found, critical_point_completed):
    """Solve one quadrant over the plasticity grid for a fixed shear.

    Branch 1 (used while the critical plane has not yet been found) locates the
    first fully-yielded plane on the LMS. Branch 2 (used once the critical plane
    is found) captures the plasticity level on the HMS at the same shear.

    Returns
    -------
    tuple
        ``(M_Rd_tot, M_Ed, r, critical_point_info, plasticity_level_max_info,
        critical_point_completed, pl_ratio_values)``.
    """
    z = calculate_zep(sector, N_Q, n_expanded, props, elastic_mode=False)
    _, pl_ratio = limit_web_plasticity(z, n_expanded, props)
    _, _, M_Rd_tot, M_Ed, r = calculate_moment_capacity(
        sector, z, Ved, N_T, n_expanded, props, elastic_mode=False
    )

    r_values = r.squeeze()
    pl_ratio_values = pl_ratio.squeeze()

    # Planes/plasticity levels at which the web has fully yielded.
    pl_ratio_ge100_mask = pl_ratio_values >= 100.0
    fully_yielded_mask = pl_ratio_ge100_mask & (r_values >= 1.0)
    n_idxs_fully_yielded = np.argmax(fully_yielded_mask, axis=1)
    pl_ratio_fully_yielded = pl_ratio_values[NPOINTS, n_idxs_fully_yielded]

    # Branch 1 — capture the LMS critical plane (first full yield), once.
    if np.any(fully_yielded_mask) and not critical_point_found:
        critical_point_idx = int(np.argmax(pl_ratio_fully_yielded))
        critical_point_info = {
            "sector": sector,
            "point": critical_point_idx,
            "pl_ratio": pl_ratio_fully_yielded[critical_point_idx],
            "n": n_values[n_idxs_fully_yielded[critical_point_idx]],
        }
    else:
        critical_point_info = None

    # Branch 2 — capture the HMS plasticity level once the critical plane exists.
    if critical_point_found and not critical_point_completed:
        converged, (n_idxs_r_limit, _, pl_ratio_r_converged) = find_optimal_r(
            sector, r_values, props, elastic_mode=False,
            pl_ratio_values=pl_ratio_values,
        )
        if converged:
            point_max = int(np.argmax(pl_ratio_r_converged))
            plasticity_level_max_info = {
                "point": point_max,
                "pl_ratio": pl_ratio_r_converged[point_max],
                "n": n_values[n_idxs_r_limit[point_max]],
            }
        else:
            plasticity_level_max_info = None
        critical_point_completed = True
    else:
        plasticity_level_max_info = None

    return (M_Rd_tot, M_Ed, r, critical_point_info,
            plasticity_level_max_info, critical_point_completed, pl_ratio_values)


def _compute_tee_max_capacity(LMS_sector, HMS_sector, state, Ved, N_T, N_Q_map,
                              props, n_expanded, n_values, M_V_Ratio):
    """Advance one Tee (an LMS/HMS pair) by one load step, with redistribution.

    Mutates ``state`` in place and returns ``(status, stop_criterion, state)``
    where ``status`` is ``'continue'`` or ``'stop'``.
    """
    # --- Lower moment side ---
    (LMS_M_Rd_tot, LMS_M_Ed, _, LMS_critical_point_info, _,
     state["critical_point_completed"], _) = _perform_enhanced_rsm(
        LMS_sector, N_Q_map[LMS_sector], Ved, N_T, props, n_expanded, n_values,
        state["critical_point_found"], state["critical_point_completed"],
    )

    if LMS_critical_point_info:
        state["critical_point_found"] = True
        state["critical_Ved"] = Ved
        state["critical_point"] = LMS_critical_point_info["point"]
        state["critical_pl_ratio"] = LMS_critical_point_info["pl_ratio"]
        state["critical_n"] = LMS_critical_point_info["n"]
        state["critical_sector"] = LMS_sector

    # --- Higher moment side ---
    (HMS_M_Rd_tot, HMS_M_Ed, _, _, HMS_plasticity_level_max_info,
     state["critical_point_completed"], HMS_pl_ratio_values) = _perform_enhanced_rsm(
        HMS_sector, N_Q_map[HMS_sector], Ved, N_T, props, n_expanded, n_values,
        state["critical_point_found"], state["critical_point_completed"],
    )

    if HMS_plasticity_level_max_info and not state["critical_pl_level_HMS_captured"]:
        state["pl_level_max_HMS"] = HMS_plasticity_level_max_info["pl_ratio"]
        state["point_max_HMS"] = HMS_plasticity_level_max_info["point"]
        state["n_max_HMS"] = HMS_plasticity_level_max_info["n"]
        state["critical_pl_level_HMS_captured"] = True

    # --- Moment redistribution (only after the critical plane exists) ---
    if state["critical_point_found"]:
        DM_T, _, _, _, _, HMS_r_after = redistribute_moment(
            Ved, LMS_M_Rd_tot, LMS_M_Ed, HMS_M_Rd_tot, HMS_M_Ed
        )
        converged_RE, (HMS_n_idxs_r_limit, _, HMS_pl_ratio_r_converged) = find_optimal_r(
            HMS_sector, HMS_r_after, props, elastic_mode=False,
            pl_ratio_values=HMS_pl_ratio_values,
        )

        if converged_RE:
            optimal_DM_T = DM_T[NPOINTS, HMS_n_idxs_r_limit]
            max_point = int(np.argmax(HMS_pl_ratio_r_converged))
            max_pl_ratio = HMS_pl_ratio_r_converged[max_point]
            max_n_idx = HMS_n_idxs_r_limit[max_point]
            max_DM_T = optimal_DM_T[max_point]

            # Stop criterion 1: HMS exceeds full plasticity.
            if max_pl_ratio > 100:
                state["redistribution_stop_criterion"] = (
                    f"Reached max. plasticity in Q{HMS_sector}."
                )
                return "stop", state["redistribution_stop_criterion"], state

            # Stop criterion 2: redistributed moment exceeds the Tee resistance.
            M_T_Rd = analyze_circular_opening(
                d_o=props.h_o, h=props.h, b_f=props.b_f, t_f=props.t_f,
                t_w=props.t_w, r=props.r, f_y=props.f_y,
                moment_shear_ratio=M_V_Ratio, V_Ed=Ved / 1e3,
            ) * 1e6
            if max_DM_T > M_T_Rd:
                state["redistribution_stop_criterion"] = (
                    "Exceeded the Tee's plastic bending resistance."
                )
                return "stop", state["redistribution_stop_criterion"], state

            # Both checks passed — persist this step as the new last-valid state.
            state["max_point_HMS_RE"] = max_point
            state["max_pl_ratio_HMS_RE"] = max_pl_ratio
            state["max_n_idx_HMS_RE"] = max_n_idx
            state["max_n_HMS_RE"] = n_values[max_n_idx]
            state["max_redistribution"] = max_DM_T
            state["critical_Ved_RE"] = Ved

    return "continue", None, state


def _edge_stress_360(Ved_value, props, M_V_Ratio, clip=True):
    """Edge-stress distribution around the whole opening at a given shear.

    Parameters
    ----------
    Ved_value : float
        Shear force (N).
    props : EllipticalSectionProperties
        Elliptical section properties.
    M_V_Ratio : float
        Global moment-to-shear ratio (m).
    clip : bool, optional
        Clip the stresses to +/- f_y (as done for the plastic-capacity plots).

    Returns
    -------
    numpy.ndarray
        360-degree edge-stress distribution (MPa).
    """
    Med = Ved_value * (M_V_Ratio * 1e3)
    N_T = Med * (0.5 * props.h_o + props.c_o) * props.A_T_o / props.I_beam

    s_edge = {}
    for sector in (1, 2, 3, 4):
        N_th, _, M_th = perform_rsm(sector, Ved_value, N_T, props, vectorised=False)
        s_edge[sector] = compute_elastic_stress(sector, N_th, M_th, props)

    s_edge_360 = np.concatenate([
        s_edge[1][-2::-1], s_edge[2][1:],
        s_edge[3][-2::-1], s_edge[4][1:],
    ])
    if clip:
        s_edge_360 = np.clip(s_edge_360, -props.f_y, props.f_y)
    return s_edge_360


def run_mode_plcap_elliptical(h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y, M_V_Ratio,
                              max_n=20, min_Ved=None):
    """Run a PLCAP (plastic capacity) analysis.

    Parameters
    ----------
    h, h_o, b_f, t_w, t_f, r, f_y : float
        Section depth, opening height (the vertical axis), flange width, web
        thickness, flange thickness, root radius (mm) and yield strength (MPa).
    a_b_ratio : float
        Ratio of the horizontal to the vertical axis of the opening.
    M_V_Ratio : float
        Global moment-to-shear ratio at the opening centre-line (m).
    max_n : float, optional
        Upper bound of the plasticity strain factor grid (default 20).
    min_Ved : float, optional
        Starting shear for the sweep (N). If omitted, the elastic limit from an
        EL1 analysis is used.

    Returns
    -------
    PLCAPEllipticalResult
        The plastic-capacity results.
    """
    start = time.perf_counter()

    props, _, V_Rd, _ = build_elliptical_section_properties(
        h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y
    )

    n_values = np.arange(1.01, max_n + 0.01, 0.01)
    n_expanded = n_values[np.newaxis, np.newaxis, :]

    # Starting shear: the elastic limit unless supplied.
    if min_Ved is None:
        min_Ved = run_mode_el1_elliptical(
            h, h_o, a_b_ratio, b_f, t_w, t_f, r, f_y, M_V_Ratio
        ).V_ed_EL1 * 1e3

    UT_state = _make_initial_state()   # Upper Tee: Q1 (LMS), Q2 (HMS)
    BT_state = _make_initial_state()   # Bottom Tee: Q4 (LMS), Q3 (HMS)

    Ved = min_Ved
    V_Rd_N = np.floor(V_Rd) * 1e3      # sweep bound (N)

    while Ved <= V_Rd_N:
        Med = Ved * (M_V_Ratio * 1e3)
        N_T = Med * (0.5 * h_o + props.c_o) * props.A_T_o / props.I_beam

        # Internal axial forces per quadrant at this shear.
        N_Q_map = {
            sector: perform_rsm(sector, Ved, N_T, props, vectorised=False)[0]
            for sector in (1, 2, 3, 4)
        }

        # Upper Tee first, so the critical plane is found on the LMS.
        status_UT, _, _ = _compute_tee_max_capacity(
            1, 2, UT_state, Ved, N_T, N_Q_map, props, n_expanded, n_values, M_V_Ratio
        )
        if status_UT == "stop":
            break

        status_BT, _, _ = _compute_tee_max_capacity(
            4, 3, BT_state, Ved, N_T, N_Q_map, props, n_expanded, n_values, M_V_Ratio
        )
        if status_BT == "stop":
            break

        Ved += 1e3
        if time.perf_counter() - start > _MAX_SWEEP_SECONDS:
            raise RuntimeError("Analysis exceeded the expected time limit.")

    duration = time.perf_counter() - start

    # --- Assemble results from the Upper Tee state (matches the reported side) ---
    critical_Ved = UT_state["critical_Ved"]
    if critical_Ved is None:
        raise RuntimeError(
            "The Upper Tee never reached a fully-yielded plane before the sweep "
            "terminated. Check the inputs or the starting shear."
        )

    critical_Ved_RE = UT_state["critical_Ved_RE"]
    if critical_Ved_RE is None:
        # No redistribution step succeeded — fall back to the no-redistribution shear.
        V_increase_pct = 0.0
        critical_Ved_RE = critical_Ved
    else:
        V_increase_pct = (critical_Ved_RE - critical_Ved) / critical_Ved * 100

    critical_Med = critical_Ved * (M_V_Ratio * 1e3)
    critical_Med_RE = critical_Ved_RE * (M_V_Ratio * 1e3)

    # Edge stresses without and with redistribution.
    s_edge_360 = _edge_stress_360(critical_Ved, props, M_V_Ratio)
    s_edge_360_RE = _edge_stress_360(critical_Ved_RE, props, M_V_Ratio)

    # Fallbacks if a critical angle was not established through plasticity.
    point_LMS = UT_state["critical_point"]
    if not point_LMS:
        point_LMS = int(np.argmax(np.abs(s_edge_360)))

    DM_T = UT_state["max_redistribution"]

    return PLCAPEllipticalResult(
        duration=duration,
        V_ed_pl=critical_Ved / 1e3,
        M_ed_pl=critical_Med / 1e6,
        V_ed_pl_RE=critical_Ved_RE / 1e3,
        M_ed_pl_RE=critical_Med_RE / 1e6,
        V_increase_pct=V_increase_pct,
        DM_T=(DM_T / 1e6) if DM_T is not None else None,
        stop_criterion=UT_state["redistribution_stop_criterion"],
        point_LMS=point_LMS,
        pl_ratio_LMS=UT_state["critical_pl_ratio"],
        n_LMS=UT_state["critical_n"],
        point_HMS=UT_state["point_max_HMS"],
        pl_ratio_HMS=UT_state["pl_level_max_HMS"],
        n_HMS=UT_state["n_max_HMS"],
        point_HMS_RE=UT_state["max_point_HMS_RE"],
        pl_ratio_HMS_RE=UT_state["max_pl_ratio_HMS_RE"],
        n_HMS_RE=UT_state["max_n_HMS_RE"],
        s_edge_360=s_edge_360,
        s_edge_360_RE=s_edge_360_RE,
    )


if __name__ == "__main__":
    # Demonstration run (UB 457x152x52, S355, 75%h elliptical opening,
    # axis ratio a/b = 1.5, M/V = 1.333).
    result = run_mode_plcap_elliptical(
        h=449.8, h_o=337.35, a_b_ratio=1.5, b_f=152.4, t_w=7.6, t_f=10.9,
        r=10.2, f_y=355, M_V_Ratio=1.333,
    )
    print("Mode PLCAP - Plastic Capacity (elliptical opening)")
    print(f"  duration            : {result.duration:.3f} s")
    print("  Without redistribution:")
    print(f"    V_ed = {result.V_ed_pl:.0f} kN   M_ed = {result.M_ed_pl:.0f} kNm")
    print(f"    LMS  point={result.point_LMS}  pl_ratio={result.pl_ratio_LMS:.1f}%  n={result.n_LMS:.2f}")
    if result.pl_ratio_HMS is not None:
        print(f"    HMS  point={result.point_HMS}  pl_ratio={result.pl_ratio_HMS:.1f}%  n={result.n_HMS:.2f}")
    print("  With redistribution:")
    print(f"    V_ed = {result.V_ed_pl_RE:.0f} kN   M_ed = {result.M_ed_pl_RE:.0f} kNm   (+{result.V_increase_pct:.1f}%)")
    if result.DM_T is not None:
        print(f"    DM_T = {result.DM_T:.2f} kNm")
    if result.point_HMS_RE is not None:
        print(f"    HMS  point={result.point_HMS_RE}  pl_ratio={result.pl_ratio_HMS_RE:.1f}%  n={result.n_HMS_RE:.2f}")
    if result.stop_criterion:
        print(f"    stop criterion: {result.stop_criterion}")
    print(f"  edge-stress distributions: {len(result.s_edge_360)} values (x2)")
