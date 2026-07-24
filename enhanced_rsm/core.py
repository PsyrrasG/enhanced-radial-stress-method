"""
Enhanced Radial Stress Method (RSM) — shared numerical core (circular openings).

This module contains the geometry, section properties and force-equilibrium
kernels shared by all four analysis modes (EL1, EL2, plastic capacity and given
forces). Each mode script imports from here rather than redefining these
functions, so that the nomenclature, conventions and numerics are identical
across the whole package.

Conventions
-----------
* Units: forces in N, moments in N.mm, stresses in N/mm^2 (MPa), lengths in mm,
  unless a variable name or docstring states otherwise. Shear forces exchanged
  with the user interface are in kN and are converted at the boundary.
* Quadrants: Q1, Q2 belong to the upper Tee and Q4, Q3 to the bottom Tee. Q1/Q4
  are the lower moment side (LMS) and Q2/Q3 the higher moment side (HMS).
* Angles: the opening edge of each quadrant is discretised into 91 planes at
  one-degree increments from 0 to 90 degrees (see ``THETAS``).
* Sign convention for edge stress: tension positive, to match the FE results.

Design
------
The shared geometry (section properties at every plane, plus the derived
scalars) is computed once by :func:`build_section_properties` and returned as an
immutable :class:`SectionProperties`. Every kernel receives this object as its
``props`` argument, which keeps the functions pure (no hidden module state) while
avoiding very long argument lists.
"""

from typing import NamedTuple
import numpy as np

# Plane discretisation: one quadrant into 91 planes (0 deg to 90 deg).
THETAS = np.arange(0, 91, 1)


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------
class SectionProperties(NamedTuple):
    """Geometry and section properties shared by all Enhanced-RSM kernels.

    All array attributes are indexed by plane angle and have shape ``(91,)``;
    the ``*_exp`` variants are the same arrays reshaped to ``(91, 1)`` so they
    broadcast against a trailing load (``Ved``) or plasticity (``n``) axis.

    Attributes
    ----------
    h, h_o, b_f, t_w, t_f, r, f_y : float
        Section depth, opening diameter, flange width, web thickness, flange
        thickness, root radius (mm) and yield strength (MPa).
    th_rad : numpy.ndarray, shape (91,)
        Plane angles in radians.
    th_rad_exp : numpy.ndarray, shape (91, 1)
        ``th_rad`` reshaped for broadcasting.
    d_T_th, h_T_th, t_f_th, A_T_th, A_f_th, I_T_th, c_th : numpy.ndarray
        Inclined-Tee web-outstand depth, total Tee depth, inclined flange
        thickness, Tee area, flange area, second moment of area and
        neutral-axis position, each of shape ``(91,)``.
    d_T_th_exp, h_T_th_exp, t_f_th_exp, A_T_th_exp, A_f_th_exp, I_T_th_exp, c_th_exp : numpy.ndarray
        The ``(91, 1)`` broadcasting variants of the above.
    d_T_o, A_T_o, c_o, z_o : float
        Section properties at the vertical centre-line (theta = 0).
    I_beam : float
        Second moment of area of the perforated section at the centre-line.

    Notes
    -----
    The pure-shear resistance (``V_T_Rd``, ``V_Rd``) and the trial-shear bound
    (``max_Ved``) are resistances/forces rather than section properties, so they
    are returned separately by :func:`build_section_properties` and are not held
    on this record.
    """

    h: float
    h_o: float
    b_f: float
    t_w: float
    t_f: float
    r: float
    f_y: float

    th_rad: np.ndarray
    th_rad_exp: np.ndarray

    d_T_th: np.ndarray
    h_T_th: np.ndarray
    t_f_th: np.ndarray
    A_T_th: np.ndarray
    A_f_th: np.ndarray
    I_T_th: np.ndarray
    c_th: np.ndarray

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


def build_section_properties(h, h_o, b_f, t_w, t_f, r, f_y):
    """Compute the inclined-Tee section properties for every plane.

    This is the single factory for the geometry shared by all modes. It
    evaluates the section properties at 91 planes around the quadrant, the
    corresponding centre-line values, the perforated-section second moment of
    area and the pure-shear resistance used to bound the trial-shear range.

    Parameters
    ----------
    h : float
        Overall section depth (mm).
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

    Returns
    -------
    props : SectionProperties
        The populated, immutable geometry/section-properties container.
    V_T_Rd : float
        Tee pure-shear resistance at the centre-line (kN).
    V_Rd : float
        Total pure-shear resistance at the centre-line (kN).
    max_Ved : float
        Upper bound of the trial-shear range (kN), ``floor(V_Rd)``.
    """
    th_rad = np.deg2rad(THETAS)
    cos_th = np.cos(th_rad)

    # Inclined-Tee properties at every plane (shape (91,)).
    d_T_th = 0.5 * ((h - 2 * t_f) / cos_th - h_o)
    h_T_th = d_T_th + t_f / cos_th
    t_f_th = t_f / cos_th
    A_T_th = t_w * d_T_th + b_f * t_f / cos_th
    A_f_th = b_f * t_f / cos_th
    I_T_th = (
        d_T_th**4 * t_w**2
        + (t_f / cos_th)**4 * b_f**2
        + 4 * d_T_th**3 * t_w * b_f * (t_f / cos_th)
        + 4 * (t_f / cos_th)**3 * d_T_th * t_w * b_f
        + 6 * d_T_th**2 * (t_f / cos_th)**2 * b_f * t_w
    ) / (12 * (b_f * (t_f / cos_th) + d_T_th * t_w))
    c_th = (
        0.5 * d_T_th**2 * t_w
        + (d_T_th + 0.5 * (t_f / cos_th)) * (t_f / cos_th) * b_f
    ) / (t_w * d_T_th + (t_f / cos_th) * b_f)

    # Centre-line values (theta = 0).
    d_T_o = d_T_th[0]
    A_T_o = A_T_th[0]
    c_o = c_th[0]
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

    props = SectionProperties(
        h=h, h_o=h_o, b_f=b_f, t_w=t_w, t_f=t_f, r=r, f_y=f_y,
        th_rad=th_rad, th_rad_exp=th_rad[:, None],
        d_T_th=d_T_th, h_T_th=h_T_th, t_f_th=t_f_th, A_T_th=A_T_th,
        A_f_th=A_f_th, I_T_th=I_T_th, c_th=c_th,
        d_T_th_exp=d_T_th[:, None], h_T_th_exp=h_T_th[:, None],
        t_f_th_exp=t_f_th[:, None], A_T_th_exp=A_T_th[:, None],
        A_f_th_exp=A_f_th[:, None], I_T_th_exp=I_T_th[:, None],
        c_th_exp=c_th[:, None],
        d_T_o=d_T_o, A_T_o=A_T_o, c_o=c_o, z_o=z_o, I_beam=I_beam,
    )
    return props, V_T_Rd, V_Rd, max_Ved


# ---------------------------------------------------------------------------
# Force-equilibrium kernels
# ---------------------------------------------------------------------------
def perform_rsm(sector, Ved, N_T, props, vectorised=True):
    """RSM force equilibrium on the inclined plane of one quadrant.

    Computes the internal axial force, shear force and moment on the inclined
    plane for every plane angle. Works both for a vector of trial shear forces
    (``vectorised=True``, used for the elastic-limit search) and for a single
    shear force (``vectorised=False``, used inside the plastic load sweep).

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    Ved : numpy.ndarray or float
        Applied vertical shear (N). A 1-D array when ``vectorised`` is True,
        otherwise a scalar.
    N_T : numpy.ndarray or float
        Axial force in the Tee due to global bending (N), matching ``Ved``.
    props : SectionProperties
        Shared section properties.
    vectorised : bool, optional
        If True, ``Ved`` and ``N_T`` are broadcast over a trailing load axis and
        the result has shape ``(91, len(Ved))``; otherwise the result is
        ``(91,)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(N_th, V_th, M_th)`` — internal axial force, shear force and moment
        on the inclined plane.
    """
    th_rad = props.th_rad
    c_th = props.c_th
    c_o = props.c_o
    h_o = props.h_o

    if vectorised:
        Ved = Ved[np.newaxis, :]
        N_T = N_T[np.newaxis, :]
        th_rad = th_rad[:, np.newaxis]
        c_th = c_th[:, np.newaxis]

    if sector in (1, 4):
        N_th = 0.5 * Ved * np.sin(th_rad) + N_T * np.cos(th_rad)
        V_th = -0.5 * Ved * np.cos(th_rad) + N_T * np.sin(th_rad)
        M_th = (
            0.5 * Ved * (c_th + 0.5 * h_o) * np.sin(th_rad)
            - N_T * (c_o + 0.5 * h_o - (c_th + 0.5 * h_o) * np.cos(th_rad))
        )
    elif sector in (2, 3):
        N_th = -0.5 * Ved * np.sin(th_rad) + N_T * np.cos(th_rad)
        V_th = 0.5 * Ved * np.cos(th_rad) + N_T * np.sin(th_rad)
        M_th = (
            0.5 * Ved * (c_th + 0.5 * h_o) * np.sin(th_rad)
            + N_T * (c_o + 0.5 * h_o - (c_th + 0.5 * h_o) * np.cos(th_rad))
        )
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    return N_th, V_th, M_th


def calculate_zep(sector, N, n, props, elastic_mode=True):
    """Elasto-plastic neutral-axis position measured from the outer flange fibre.

    Forms and solves the equilibrium relation for the neutral-axis depth of the
    inclined Tee. At the elastic limit (``elastic_mode=True``, ``n = 1``) the
    relation is linear and is solved directly; in the plastic regime it is the
    quadratic ``A z^2 + B z + C = 0`` and the physically admissible (smaller in
    magnitude) root is returned.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    N : numpy.ndarray
        Axial force on the inclined plane (N). In elastic mode it carries the
        trailing load axis; in plastic mode it is ``(91,)`` and is reshaped
        internally to broadcast against ``n``.
    n : float or numpy.ndarray
        Plasticity strain factor. 1 at the elastic limit; an array of trial
        values (already shaped to broadcast) in the plastic regime.
    props : SectionProperties
        Shared section properties.
    elastic_mode : bool, optional
        If True, solve the linear (n = 1) relation; otherwise solve the
        quadratic.

    Returns
    -------
    numpy.ndarray
        Neutral-axis depth ``z`` from the outer flange fibre.
    """
    A_f_exp = props.A_f_th_exp
    h_T_exp = props.h_T_th_exp
    t_f_exp = props.t_f_th_exp
    t_w = props.t_w
    f_y = props.f_y

    if not elastic_mode:
        N = N[:, np.newaxis]

    if sector in (1, 4):
        A = ((n - 1)**2 * t_w) / (2 * n)
        B = n * A_f_exp - (N / f_y) + t_w * (((2 * n - 1) / n) * h_T_exp - n * t_f_exp)
        C = -(
            0.5 * n * t_f_exp * A_f_exp
            - (N * h_T_exp) / f_y
            + (t_w / 2) * (((2 * n - 1) / n) * h_T_exp**2 - n * t_f_exp**2)
        )
    elif sector in (2, 3):
        A = ((n - 1)**2 * t_w) / (2 * n)
        B = n * A_f_exp + (N / f_y) + t_w * (((2 * n - 1) / n) * h_T_exp - n * t_f_exp)
        C = -(
            0.5 * n * t_f_exp * A_f_exp
            + (N * h_T_exp) / f_y
            + (t_w / 2) * (((2 * n - 1) / n) * h_T_exp**2 - n * t_f_exp**2)
        )
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    if elastic_mode:
        # Linear limit (A -> 0): z = -C / B.
        return -C / B

    discriminant = B**2 - 4 * A * C
    z1 = (-B + np.sqrt(discriminant)) / (2 * A)
    z2 = (-B - np.sqrt(discriminant)) / (2 * A)
    return np.where(np.abs(z1) < np.abs(z2), z1, z2)


def limit_web_plasticity(z, n, props):
    """Depth and proportion of the inclined web that has yielded.

    Parameters
    ----------
    z : numpy.ndarray
        Neutral-axis depth from the outer flange fibre.
    n : float or numpy.ndarray
        Plasticity strain factor.
    props : SectionProperties
        Shared section properties.

    Returns
    -------
    tuple of numpy.ndarray
        ``(pl_depth, pl_ratio)`` — yielded web depth and the yielded proportion
        of the web outstand (per cent).
    """
    h_T_exp = props.h_T_th_exp
    t_f_exp = props.t_f_th_exp
    pl_depth = (n - 1) / n * (h_T_exp - z)
    pl_ratio = pl_depth / (h_T_exp - t_f_exp) * 100
    return pl_depth, pl_ratio


def calculate_moment_capacity(sector, z, Ved, N_T, n, props, elastic_mode=True):
    """Elasto-plastic bending resistance, applied moment and their ratio.

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
    props : SectionProperties
        Shared section properties.
    elastic_mode : bool, optional
        Retained for signature parity with the other kernels; the resistance
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
    th_rad_exp = props.th_rad_exp
    t_w = props.t_w
    f_y = props.f_y
    h = props.h
    z_o = props.z_o

    # Elasto-plastic bending resistance in terms of the neutral-axis depth.
    M_Rd = (
        n * A_f_exp * ((z - 0.5 * t_f_exp)**2 / (h_T_exp - z)) * f_y
        + (n * t_w * (z - t_f_exp)**3) / (3 * (h_T_exp - z)) * f_y
        + (t_w * (3 * n**2 - 1)) / (6 * n**2) * (h_T_exp - z)**2 * f_y
    )
    # Additional resistance term from the flange.
    M_Rd_fl = n / (h_T_exp - z) * A_f_exp * t_f_exp**2 * f_y / 12
    M_Rd_tot = M_Rd + M_Rd_fl

    # Applied moment on the inclined plane.
    if sector in (1, 4):
        M_Ed = (
            -N_T * (z * np.cos(th_rad_exp) - z_o)
            + 0.5 * Ved * (0.5 * h * np.tan(th_rad_exp) - z * np.sin(th_rad_exp))
        )
    elif sector in (2, 3):
        M_Ed = (
            N_T * (z * np.cos(th_rad_exp) - z_o)
            + 0.5 * Ved * (0.5 * h * np.tan(th_rad_exp) - z * np.sin(th_rad_exp))
        )
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    r = M_Ed / M_Rd_tot
    return M_Rd, M_Rd_fl, M_Rd_tot, M_Ed, r


def compute_elastic_stress(sector, N_th, M_th, props):
    """Elastic normal stress at the opening edge for every plane.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    N_th : numpy.ndarray
        Axial force on the inclined plane (N), shape ``(91,)``.
    M_th : numpy.ndarray
        Moment on the inclined plane (N.mm), shape ``(91,)``.
    props : SectionProperties
        Shared section properties.

    Returns
    -------
    numpy.ndarray
        Edge stress for each plane (MPa), tension positive.
    """
    A_T_th = props.A_T_th
    I_T_th = props.I_T_th
    c_th = props.c_th

    if sector == 1:
        s_edge_th = -N_th / A_T_th - M_th * c_th / I_T_th
    elif sector == 2:
        s_edge_th = -N_th / A_T_th + M_th * c_th / I_T_th
    elif sector == 3:
        s_edge_th = +N_th / A_T_th - M_th * c_th / I_T_th
    elif sector == 4:
        s_edge_th = +N_th / A_T_th + M_th * c_th / I_T_th
    else:
        raise ValueError("'sector' must be an integer 1-4 (Q1-Q4).")

    return s_edge_th


def find_optimal_r(sector, r_values, props, elastic_mode=True,
                   pl_ratio_values=None, tolerance=0.005):
    """Locate, per plane, the index at which the moment ratio ``r`` reaches 1.

    Two related searches share this function. In elastic mode the search is over
    the trial-shear axis and returns, for each plane, the first shear at which
    ``r`` first reaches 1 (the elastic limit). In plastic mode the search is
    over the plasticity axis ``n`` and returns, for each plane, the plasticity
    level at which ``r`` reaches 1.

    The plastic search uses a symmetric convergence band ``|r - 1| <= tolerance``
    and additionally requires the plane to exhibit a genuine crossing from above
    (some ``r > 1`` in the row), which avoids spurious convergence where ``r``
    merely brushes 1 at the start of the plasticity range. Planes that do not
    truly converge are invalidated so that a downstream ``argmax`` over the
    plasticity level cannot select them.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    r_values : numpy.ndarray
        Moment ratio for every plane and every trial value, shape
        ``(91, n_trials)``.
    props : SectionProperties
        Shared section properties (retained for signature parity).
    elastic_mode : bool, optional
        If True, search the elastic limit over the trial-shear axis; otherwise
        search the plasticity level over the ``n`` axis.
    pl_ratio_values : numpy.ndarray, optional
        Yielded-web proportion for every plane and trial value (plastic mode
        only), shape ``(91, n_trials)``.
    tolerance : float, optional
        Half-width of the plastic convergence band on ``r``.

    Returns
    -------
    tuple
        ``(converged, (idxs, r_values_converged, pl_ratio_r_converged))``.
        ``converged`` is a bool. In elastic mode ``pl_ratio_r_converged`` is
        ``None``. When not converged the inner values are ``None``.
    """
    if elastic_mode:
        r_ge1_mask = r_values >= 1.0
        if not np.any(r_ge1_mask):
            raise RuntimeError(
                f"Elastic limit not reached in Q{sector}; increase the shear range."
            )
        idxs_not_converged = ~np.any(r_ge1_mask, axis=1)
        idxs = np.argmax(r_ge1_mask, axis=1)
        # Planes that never reach yield fall back to the last trial index.
        idxs[idxs_not_converged] = r_values.shape[1] - 1
        r_values_converged = r_values[THETAS, idxs]
        return True, (idxs, r_values_converged, None)

    # Plastic mode: robust band + genuine-crossing guard + invalidation.
    r_converged_mask = (r_values >= 1.0 - tolerance) & (r_values <= 1.0 + tolerance)
    has_overcapacity = np.any(r_values > 1.0, axis=1)
    theta_converged = np.any(r_converged_mask, axis=1) & has_overcapacity

    if not np.any(theta_converged):
        return False, (None, None, None)

    idxs = np.argmax(r_converged_mask, axis=1)
    r_values_converged = r_values[THETAS, idxs].astype(float)
    pl_ratio_r_converged = pl_ratio_values[THETAS, idxs].astype(float)

    # Invalidate planes that did not genuinely converge.
    idxs = idxs.copy()
    idxs[~theta_converged] = -1
    r_values_converged[~theta_converged] = np.nan
    pl_ratio_r_converged[~theta_converged] = -np.inf

    return True, (idxs, r_values_converged, pl_ratio_r_converged)


def solve_elastic_limit(sector, V_ed_values, N_T_values, props):
    """Elastic limit of one quadrant: the shear at which its first plane yields.

    Vectorises over the trial-shear axis. For every plane and every trial shear
    the elastic moment ratio is evaluated; the plane that reaches yield at the
    lowest shear is the critical plane, and that shear is the quadrant's elastic
    limit. Used by EL1 on the lower moment side and by EL2 on the higher moment
    side.

    Parameters
    ----------
    sector : int
        Quadrant, 1 to 4.
    V_ed_values : numpy.ndarray
        Trial shear forces (N).
    N_T_values : numpy.ndarray
        Axial force in the Tee for each trial shear (N).
    props : SectionProperties
        Shared section properties.

    Returns
    -------
    tuple
        ``(V_ed_critical, theta_critical, s_edge)`` — the elastic-limit shear
        (N), the critical plane angle (degrees) and the quadrant's edge-stress
        distribution at that shear (MPa).
    """
    N_th, _, M_th = perform_rsm(sector, V_ed_values, N_T_values, props, vectorised=True)
    z = calculate_zep(sector, N_th, 1, props, elastic_mode=True)
    *_, r = calculate_moment_capacity(
        sector, z, V_ed_values, N_T_values, 1, props, elastic_mode=True
    )

    _, (V_idxs, _, _) = find_optimal_r(sector, r, props, elastic_mode=True)

    V_ed_at_limit = V_ed_values[V_idxs]
    theta_critical = int(np.argmin(V_ed_at_limit))
    V_ed_critical = float(V_ed_at_limit[theta_critical])
    V_ed_critical_idx = V_idxs[theta_critical]

    N_crit = N_th[THETAS, V_ed_critical_idx]
    M_crit = M_th[THETAS, V_ed_critical_idx]
    s_edge = compute_elastic_stress(sector, N_crit, M_crit, props)

    return V_ed_critical, theta_critical, s_edge


def redistribute_moment(Ved, M_Rd_tot_LMS, M_Ed_LMS, M_Rd_tot_HMS, M_Ed_HMS):
    """Redistribute excess moment from the lower to the higher moment side.

    Only positive excess (``M_Ed_LMS - M_Rd_tot_LMS``) is shed from the LMS and
    added to the HMS.

    Parameters
    ----------
    Ved : float
        Applied vertical shear (N), used to express the redistribution as an
        eccentricity.
    M_Rd_tot_LMS, M_Ed_LMS : numpy.ndarray
        Total resistance and applied moment on the lower moment side.
    M_Rd_tot_HMS, M_Ed_HMS : numpy.ndarray
        Total resistance and applied moment on the higher moment side.

    Returns
    -------
    tuple of numpy.ndarray
        ``(DM_T, e, M_Ed_LMS_after, M_Ed_HMS_after, r_LMS_after, r_HMS_after)``.
    """
    M_Rd_tot_LMS = M_Rd_tot_LMS.squeeze()
    M_Ed_LMS = M_Ed_LMS.squeeze()
    M_Rd_tot_HMS = M_Rd_tot_HMS.squeeze()
    M_Ed_HMS = M_Ed_HMS.squeeze()

    DM_T = np.maximum(M_Ed_LMS - M_Rd_tot_LMS, 0)
    e = DM_T / (0.5 * Ved)
    M_Ed_LMS_after = M_Ed_LMS - DM_T
    r_LMS_after = M_Ed_LMS_after / M_Rd_tot_LMS
    M_Ed_HMS_after = M_Ed_HMS + DM_T
    r_HMS_after = M_Ed_HMS_after / M_Rd_tot_HMS

    return DM_T, e, M_Ed_LMS_after, M_Ed_HMS_after, r_LMS_after, r_HMS_after

