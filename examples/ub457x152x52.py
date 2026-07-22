"""
Worked example: UB 457x152x52, S355, circular opening of 75 per cent of the
section depth, at a moment-to-shear ratio of 1.333 m.

Runs all four analysis modes on the same section and prints a summary, so that
the relationship between the elastic limit, the plastic capacity and the state
at a prescribed pair of forces can be seen at a glance.

Run with:  python examples/ub457x152x52.py
"""

from enhanced_rsm import (
    run_mode_el1,
    run_mode_el2,
    run_mode_plcap,
    run_mode_given_forces,
)

SECTION = dict(h=449.8, h_o=337.35, b_f=152.4, t_w=7.6, t_f=10.9, r=10.2, f_y=355)
M_V_RATIO = 1.333


def main():
    print("UB 457x152x52, S355, 75%h circular opening, M/V = 1.333 m")
    print("=" * 62)

    el1 = run_mode_el1(**SECTION, M_V_Ratio=M_V_RATIO)
    print(f"\nEL1  elastic limit (LMS)")
    print(f"  V_ed = {el1.V_ed_EL1:6.0f} kN   M_ed = {el1.M_ed_EL1:6.0f} kNm")
    print(f"  critical plane: Q1 at {el1.theta_critical_Q1} deg "
          f"({el1.s_edge_max_Q1:.0f} MPa)")

    el2 = run_mode_el2(**SECTION, M_V_Ratio=M_V_RATIO)
    print(f"\nEL2  elastic limit (HMS)")
    print(f"  V_ed = {el2.V_ed_EL2:6.0f} kN   M_ed = {el2.M_ed_EL2:6.0f} kNm")
    print(f"  critical plane: Q2 at {el2.theta_critical_Q2} deg "
          f"({el2.s_edge_max_Q2:.0f} MPa)")
    print(f"  LMS already yielded: {el2.pl_ratio_Q1:.1f}% (n = {el2.n_Q1:.2f})")
    print(f"  redistribution required: {el2.redistribution_required}")

    plcap = run_mode_plcap(**SECTION, M_V_Ratio=M_V_RATIO)
    print(f"\nPLCAP  plastic capacity")
    print(f"  without redistribution: V_ed = {plcap.V_ed_pl:6.0f} kN")
    print(f"     LMS at {plcap.theta_LMS} deg, {plcap.pl_ratio_LMS:.1f}% yielded "
          f"(n = {plcap.n_LMS:.2f})")
    print(f"  with redistribution:    V_ed = {plcap.V_ed_pl_RE:6.0f} kN "
          f"(+{plcap.V_increase_pct:.1f}%)")
    if plcap.DM_T is not None:
        print(f"     moment redistributed: {plcap.DM_T:.2f} kNm")
    if plcap.stop_criterion:
        print(f"     stopped by: {plcap.stop_criterion}")

    # Interrogate the section midway between the elastic limit and the capacity.
    V_ed = round((el1.V_ed_EL1 + plcap.V_ed_pl) / 2)
    gf = run_mode_given_forces(**SECTION, Ved=V_ed, Med=V_ed * M_V_RATIO)
    print(f"\nGIVEN FORCES  at V_ed = {gf.V_ed:.0f} kN, M_ed = {gf.M_ed:.0f} kNm")
    print(f"  response: {'elastic' if gf.elastic else 'elasto-plastic'}")
    if not gf.elastic:
        print(f"  LMS at {gf.theta_LMS} deg, {gf.pl_ratio_LMS:.1f}% yielded "
              f"(n = {gf.n_LMS:.2f})")
        print(f"  redistribution: {gf.redistribution}")
    print()


if __name__ == "__main__":
    main()
