"""
Enhanced Radial Stress Method (RSM) for perforated steel beams.

An elasto-plastic extension of the Radial Stress Method for the *Vierendeel*
bending design of steel I-beams with web openings, in accordance with
EN 1993-1-13.

Four analysis modes share a single numerical core:

``el1``
    Elastic limit — the shear force at which the lower moment (LMS) side first yields.
``el2``
    Elastic limit of the higher moment side (HMS), with the plasticity already
    developed on the LMS at that shear force.
``plcap``
    Plastic capacity, with moment redistribution between the two sides.
``given_forces``
    The state of the opening at a prescribed shear and moment pair.

Circular and elliptical openings are handled by parallel implementations sharing
the geometry-independent numerics. The elliptical modes carry an ``_elliptical``
suffix and take an additional ``a_b_ratio`` argument; at ``a_b_ratio = 1`` they
reproduce the circular results exactly, which is verified by the test suite.

Example
-------
>>> from enhanced_rsm import run_mode_el1
>>> result = run_mode_el1(h=449.8, h_o=337.35, b_f=152.4, t_w=7.6,
...                       t_f=10.9, r=10.2, f_y=355, M_V_Ratio=1.333)
>>> round(result.V_ed_EL1)
78
"""

__version__ = "1.0.0"

from .core import (
    THETAS,
    SectionProperties,
    build_section_properties,
    perform_rsm,
    calculate_zep,
    limit_web_plasticity,
    calculate_moment_capacity,
    compute_elastic_stress,
    find_optimal_r,
    solve_elastic_limit,
    redistribute_moment,
)
from .tee_bending_resistance import (
    VierendeelCircularOpening,
    analyze_circular_opening,
)
from .el1 import run_mode_el1, EL1Result
from .core_elliptical import (
    NPOINTS,
    EllipticalSectionProperties,
    build_elliptical_section_properties,
    perimeter_coordinates_360,
)
from .core_elliptical import point_coordinates
from .el1_elliptical import run_mode_el1_elliptical, EL1EllipticalResult
from .el2_elliptical import run_mode_el2_elliptical, EL2EllipticalResult
from .plcap_elliptical import run_mode_plcap_elliptical, PLCAPEllipticalResult
from .given_forces_elliptical import (
    run_mode_given_forces_elliptical, GivenForcesEllipticalResult,
)
from .el2 import run_mode_el2, EL2Result
from .plcap import run_mode_plcap, PLCAPResult
from .given_forces import run_mode_given_forces, GivenForcesResult

__all__ = [
    "__version__",
    # Analysis modes - circular openings
    "run_mode_el1", "EL1Result",
    "run_mode_el2", "EL2Result",
    "run_mode_plcap", "PLCAPResult",
    "run_mode_given_forces", "GivenForcesResult",
    # Analysis modes - elliptical openings
    "run_mode_el1_elliptical", "EL1EllipticalResult",
    "run_mode_el2_elliptical", "EL2EllipticalResult",
    "run_mode_plcap_elliptical", "PLCAPEllipticalResult",
    "run_mode_given_forces_elliptical", "GivenForcesEllipticalResult",
    # Elliptical numerical core
    "NPOINTS", "EllipticalSectionProperties",
    "build_elliptical_section_properties", "perimeter_coordinates_360",
    "point_coordinates",
    # Shared numerical core
    "THETAS", "SectionProperties", "build_section_properties",
    "perform_rsm", "calculate_zep", "limit_web_plasticity",
    "calculate_moment_capacity", "compute_elastic_stress",
    "find_optimal_r", "solve_elastic_limit", "redistribute_moment",
    # Tee resistance
    "VierendeelCircularOpening", "analyze_circular_opening",
]
