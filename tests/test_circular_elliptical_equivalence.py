"""
Equivalence of the circular and elliptical implementations.

An ellipse of unit axis ratio is a circle, so the elliptical core must reproduce
the circular core exactly when ``a_b_ratio = 1``. Because the two are written
independently — the circular core discretises the quadrant by equal angle
increments, the elliptical core by equal arc length with the inclined plane
constructed from the local normal — agreement between them is a genuine check on
both, not a restatement of one in terms of the other.

The point at ninety degrees is excluded throughout. There the inclined plane
becomes horizontal and the Tee depth diverges, so both implementations are
degenerate there by construction; neither ever selects it as the critical plane.

Run with:  pytest tests/
"""

import numpy as np
import pytest

from enhanced_rsm import core as circular
from enhanced_rsm import core_elliptical as elliptical

# UB 457x152x52, S355, opening of 75 per cent of the section depth.
SECTION = dict(h=449.8, h_o=337.35, b_f=152.4, t_w=7.6, t_f=10.9, r=10.2, f_y=355)
M_V_RATIO = 1.333

# Machine precision leaves a little room for the different routes to the same
# quantity; anything larger would indicate a genuine discrepancy.
RTOL = 1e-10

# All perimeter points except the degenerate one at ninety degrees.
VALID = slice(0, 90)


@pytest.fixture(scope="module")
def properties():
    """Circular and unit-ratio elliptical section properties for the same section."""
    props_c, _, _, max_Ved = circular.build_section_properties(**SECTION)
    props_e, _, _, _ = elliptical.build_elliptical_section_properties(
        a_b_ratio=1.0, **SECTION
    )
    return props_c, props_e, max_Ved


@pytest.fixture(scope="module")
def loading(properties):
    """Trial shears and the axial force each induces, shared by the kernel tests."""
    props_c, _, max_Ved = properties
    V_ed_values = np.arange(1e3, max_Ved * 1e3, 1e3)
    M_ed_values = V_ed_values * M_V_RATIO * 1e3
    N_T_values = (
        M_ed_values * (0.5 * SECTION["h_o"] + props_c.c_o)
        * props_c.A_T_o / props_c.I_beam
    )
    return V_ed_values, N_T_values


def test_normal_passes_through_the_centre(properties):
    """For a circle the normal at every point meets the centre-line at the centre."""
    _, props_e, _ = properties
    assert np.allclose(props_e.yA, 0.0, atol=1e-12)


def test_radial_distance_equals_the_radius(properties):
    """The elliptical lever arm reduces to the radius of the circular opening."""
    _, props_e, _ = properties
    assert np.allclose(
        props_e.r_thP[VALID], SECTION["h_o"] / 2, rtol=RTOL
    )


def test_plane_angles_match(properties):
    """Equal arc-length spacing on a circle gives equal angle increments."""
    props_c, props_e, _ = properties
    assert np.allclose(props_e.thP_rad[VALID], props_c.th_rad[VALID], rtol=RTOL,
                       atol=1e-12)


@pytest.mark.parametrize(
    "name", ["d_T_th", "h_T_th", "t_f_th", "A_T_th", "A_f_th", "I_T_th", "c_th"]
)
def test_section_properties_match(properties, name):
    """Every inclined-Tee section property agrees between the two cores."""
    props_c, props_e, _ = properties
    assert np.allclose(
        getattr(props_e, name)[VALID], getattr(props_c, name)[VALID], rtol=RTOL
    )


def test_centreline_properties_match(properties):
    """The centre-line properties agree, though computed by different routes."""
    props_c, props_e, _ = properties
    for name in ("A_T_o", "c_o", "z_o", "I_beam"):
        assert getattr(props_e, name) == pytest.approx(
            getattr(props_c, name), rel=RTOL
        ), name


@pytest.mark.parametrize("sector", [1, 2, 3, 4])
def test_force_equilibrium_matches(properties, loading, sector):
    """The internal forces on the inclined plane agree in every quadrant."""
    props_c, props_e, _ = properties
    V_ed_values, N_T_values = loading

    forces_c = circular.perform_rsm(sector, V_ed_values, N_T_values, props_c)
    forces_e = elliptical.perform_rsm(sector, V_ed_values, N_T_values, props_e)

    scale = max(np.max(np.abs(f[VALID])) for f in forces_c)
    for got, expected in zip(forces_e, forces_c):
        assert np.allclose(got[VALID], expected[VALID], rtol=RTOL, atol=RTOL * scale)


@pytest.mark.parametrize("sector", [1, 2, 3, 4])
def test_moment_ratio_matches(properties, loading, sector):
    """The elastic moment ratio, which locates first yield, agrees."""
    props_c, props_e, _ = properties
    V_ed_values, N_T_values = loading

    N_c, _, _ = circular.perform_rsm(sector, V_ed_values, N_T_values, props_c)
    z_c = circular.calculate_zep(sector, N_c, 1, props_c, elastic_mode=True)
    *_, r_c = circular.calculate_moment_capacity(
        sector, z_c, V_ed_values, N_T_values, 1, props_c, elastic_mode=True
    )

    N_e, _, _ = elliptical.perform_rsm(sector, V_ed_values, N_T_values, props_e)
    z_e = elliptical.calculate_zep(sector, N_e, 1, props_e, elastic_mode=True)
    *_, r_e = elliptical.calculate_moment_capacity(
        sector, z_e, V_ed_values, N_T_values, 1, props_e, elastic_mode=True
    )

    assert np.allclose(r_e[VALID], r_c[VALID], rtol=RTOL)


@pytest.mark.parametrize("sector", [1, 2, 3, 4])
def test_elastic_limit_matches(properties, loading, sector):
    """The elastic limit and the critical location agree in every quadrant."""
    props_c, props_e, _ = properties
    V_ed_values, N_T_values = loading

    V_c, index_c, s_edge_c = circular.solve_elastic_limit(
        sector, V_ed_values, N_T_values, props_c
    )
    V_e, index_e, s_edge_e = elliptical.solve_elastic_limit(
        sector, V_ed_values, N_T_values, props_e
    )

    assert V_e == pytest.approx(V_c, rel=RTOL)
    assert index_e == index_c
    assert np.allclose(s_edge_e[VALID], s_edge_c[VALID], rtol=RTOL)


def test_perimeter_coordinates_lie_on_the_circle(properties):
    """The assembled perimeter coordinates trace a circle of the right radius."""
    _, props_e, _ = properties
    x_360, y_360 = elliptical.perimeter_coordinates_360(props_e)

    assert x_360.shape == (360,)
    assert y_360.shape == (360,)
    assert np.allclose(
        np.hypot(x_360, y_360), SECTION["h_o"] / 2, rtol=RTOL
    )
