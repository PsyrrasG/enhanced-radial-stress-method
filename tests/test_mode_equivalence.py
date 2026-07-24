"""
Mode-level equivalence of the circular and elliptical implementations.

Each of the four analysis modes must return the same result whether it is run
through the circular implementation or through the elliptical one with a unit
axis ratio. This exercises the full orchestration of every mode, not just the
kernels, and so complements the finer-grained checks in
``test_circular_elliptical_equivalence.py``.

Run with:  pytest tests/
"""

import pytest

from enhanced_rsm import (
    run_mode_el1, run_mode_el2, run_mode_plcap, run_mode_given_forces,
)
from enhanced_rsm.el1_elliptical import run_mode_el1_elliptical
from enhanced_rsm.el2_elliptical import run_mode_el2_elliptical
from enhanced_rsm.plcap_elliptical import run_mode_plcap_elliptical
from enhanced_rsm.given_forces_elliptical import run_mode_given_forces_elliptical

# UB 457x152x52, S355, opening of 75 per cent of the section depth.
SECTION = dict(h=449.8, h_o=337.35, b_f=152.4, t_w=7.6, t_f=10.9, r=10.2, f_y=355)
M_V_RATIO = 1.333
RTOL = 1e-9


def test_el1_equivalence():
    """The elastic limit agrees between the two implementations."""
    circular = run_mode_el1(**SECTION, M_V_Ratio=M_V_RATIO)
    elliptical = run_mode_el1_elliptical(a_b_ratio=1.0, **SECTION, M_V_Ratio=M_V_RATIO)

    assert elliptical.V_ed_EL1 == pytest.approx(circular.V_ed_EL1, rel=RTOL)
    assert elliptical.M_ed_EL1 == pytest.approx(circular.M_ed_EL1, rel=RTOL)
    assert elliptical.point_critical_Q1 == circular.theta_critical_Q1
    assert elliptical.point_critical_Q2 == circular.theta_critical_Q2
    assert elliptical.s_edge_max_Q1 == pytest.approx(circular.s_edge_max_Q1, rel=RTOL)


def test_el2_equivalence():
    """The higher-moment-side elastic limit and LMS plasticity agree."""
    circular = run_mode_el2(**SECTION, M_V_Ratio=M_V_RATIO)
    elliptical = run_mode_el2_elliptical(a_b_ratio=1.0, **SECTION, M_V_Ratio=M_V_RATIO)

    assert elliptical.V_ed_EL2 == pytest.approx(circular.V_ed_EL2, rel=RTOL)
    assert elliptical.point_critical_Q2 == circular.theta_critical_Q2
    assert elliptical.pl_ratio_Q1 == pytest.approx(circular.pl_ratio_Q1, rel=RTOL)
    assert elliptical.n_Q1 == pytest.approx(circular.n_Q1, rel=RTOL)
    assert elliptical.redistribution_required == circular.redistribution_required


def test_plcap_equivalence():
    """The plastic capacity, redistribution and stop criterion agree."""
    circular = run_mode_plcap(**SECTION, M_V_Ratio=M_V_RATIO)
    elliptical = run_mode_plcap_elliptical(a_b_ratio=1.0, **SECTION, M_V_Ratio=M_V_RATIO)

    assert elliptical.V_ed_pl == pytest.approx(circular.V_ed_pl, rel=RTOL)
    assert elliptical.V_ed_pl_RE == pytest.approx(circular.V_ed_pl_RE, rel=RTOL)
    assert elliptical.V_increase_pct == pytest.approx(circular.V_increase_pct, rel=RTOL)
    assert elliptical.DM_T == pytest.approx(circular.DM_T, rel=RTOL)
    assert elliptical.point_LMS == circular.theta_LMS
    assert elliptical.stop_criterion == circular.stop_criterion


def test_given_forces_equivalence():
    """The state at a prescribed pair of forces agrees."""
    forces = dict(Ved=100, Med=133)
    circular = run_mode_given_forces(**SECTION, **forces)
    elliptical = run_mode_given_forces_elliptical(a_b_ratio=1.0, **SECTION, **forces)

    assert elliptical.elastic == circular.elastic
    assert elliptical.point_LMS == circular.theta_LMS
    assert elliptical.pl_ratio_LMS == pytest.approx(circular.pl_ratio_LMS, rel=RTOL)
    assert elliptical.n_LMS == pytest.approx(circular.n_LMS, rel=RTOL)
    assert elliptical.redistribution == circular.redistribution


def test_elongation_reduces_capacity():
    """Elongating the opening horizontally lowers the elastic limit."""
    limits = [
        run_mode_el1_elliptical(a_b_ratio=ratio, **SECTION,
                                M_V_Ratio=M_V_RATIO).V_ed_EL1
        for ratio in (1.0, 1.25, 1.5, 2.0)
    ]
    assert limits == sorted(limits, reverse=True)
