"""
Command-line interface for the Enhanced Radial Stress Method.

A single entry point dispatches to the four analysis modes through subcommands::

    enhanced-rsm el1           --h 449.8 ... --mv 1.333
    enhanced-rsm el2           --h 449.8 ... --mv 1.333
    enhanced-rsm plcap         --h 449.8 ... --mv 1.333
    enhanced-rsm given-forces  --h 449.8 ... --ved 100 --med 133

The section geometry and material flags are shared by every subcommand; each
subcommand adds only the arguments specific to it. Results are printed as a
readable summary and can additionally be written to a JSON file with
``--output``.
"""

import argparse
import json
import sys

import numpy as np

from . import __version__
from .el1 import run_mode_el1
from .el2 import run_mode_el2
from .plcap import run_mode_plcap
from .given_forces import run_mode_given_forces


def _section_parser():
    """Return a parent parser holding the shared section and material flags."""
    parser = argparse.ArgumentParser(add_help=False)
    section = parser.add_argument_group("section geometry and material")
    section.add_argument("--h", type=float, required=True,
                         help="section depth (mm)")
    section.add_argument("--ho", type=float, required=True, dest="h_o",
                         help="opening diameter (mm)")
    section.add_argument("--bf", type=float, required=True, dest="b_f",
                         help="flange width (mm)")
    section.add_argument("--tw", type=float, required=True, dest="t_w",
                         help="web thickness (mm)")
    section.add_argument("--tf", type=float, required=True, dest="t_f",
                         help="flange thickness (mm)")
    section.add_argument("--r", type=float, required=True,
                         help="root radius (mm)")
    section.add_argument("--fy", type=float, required=True, dest="f_y",
                         help="yield strength (MPa)")

    output = parser.add_argument_group("output")
    output.add_argument("--output", metavar="FILE",
                        help="write the full results to a JSON file")
    output.add_argument("--stresses", action="store_true",
                        help="also print the edge-stress distribution")
    return parser


def _build_parser():
    """Return the top-level argument parser with one subcommand per mode."""
    shared = _section_parser()
    parser = argparse.ArgumentParser(
        prog="enhanced-rsm",
        description="Enhanced Radial Stress Method for perforated steel beams "
                    "with circular web openings.",
        epilog="Example: enhanced-rsm plcap --h 449.8 --ho 337.35 --bf 152.4 "
               "--tw 7.6 --tf 10.9 --r 10.2 --fy 355 --mv 1.333",
    )
    parser.add_argument("--version", action="version",
                        version=f"enhanced-rsm {__version__}")
    subparsers = parser.add_subparsers(dest="mode", required=True,
                                       metavar="MODE")

    el1 = subparsers.add_parser(
        "el1", parents=[shared], help="elastic limit (lower moment side)",
        description="Determine the shear at which the lower moment side first "
                    "reaches yield.")
    el1.add_argument("--mv", type=float, required=True, dest="M_V_Ratio",
                     help="moment-to-shear ratio at the opening centre-line (m)")

    el2 = subparsers.add_parser(
        "el2", parents=[shared], help="elastic limit (higher moment side)",
        description="Determine the shear at which the higher moment side first "
                    "reaches yield, and the plasticity already developed on the "
                    "lower moment side at that shear.")
    el2.add_argument("--mv", type=float, required=True, dest="M_V_Ratio",
                     help="moment-to-shear ratio at the opening centre-line (m)")
    el2.add_argument("--max-n", type=float, default=20, dest="max_n",
                     help="upper bound of the plasticity strain factor "
                          "(default: 20)")

    plcap = subparsers.add_parser(
        "plcap", parents=[shared], help="plastic capacity with redistribution",
        description="Determine the plastic capacity, with and without moment "
                    "redistribution from the lower to the higher moment side.")
    plcap.add_argument("--mv", type=float, required=True, dest="M_V_Ratio",
                       help="moment-to-shear ratio at the opening centre-line (m)")
    plcap.add_argument("--max-n", type=float, default=20, dest="max_n",
                       help="upper bound of the plasticity strain factor "
                            "(default: 20)")

    gf = subparsers.add_parser(
        "given-forces", parents=[shared],
        help="state of the opening at a prescribed shear and moment",
        description="Report the state of the opening under a prescribed pair of "
                    "applied forces.")
    gf.add_argument("--ved", type=float, required=True, dest="Ved",
                    help="applied shear at the opening centre-line (kN)")
    gf.add_argument("--med", type=float, required=True, dest="Med",
                    help="applied moment at the opening centre-line (kNm)")
    gf.add_argument("--ved-el1", type=float, default=None, dest="Ved_EL1",
                    help="elastic limit (kN); computed automatically if omitted")
    gf.add_argument("--max-n", type=float, default=20, dest="max_n",
                    help="upper bound of the plasticity strain factor "
                         "(default: 20)")

    return parser


def _format_value(value):
    """Render a single result field for the printed summary."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def _print_summary(mode, result, show_stresses=False):
    """Print a readable summary of a mode result."""
    print(f"\nEnhanced RSM - mode '{mode}'")
    print("=" * (22 + len(mode)))
    for name, value in zip(result._fields, result):
        if isinstance(value, np.ndarray):
            if show_stresses:
                print(f"  {name}:")
                for i in range(0, len(value), 12):
                    row = "  ".join(f"{v:7.1f}" for v in value[i:i + 12])
                    print(f"    {i:>3d}: {row}")
            else:
                print(f"  {name:<24}: {len(value)} values "
                      f"(use --stresses to print)")
        else:
            print(f"  {name:<24}: {_format_value(value)}")
    print()


def _to_serialisable(result):
    """Convert a result to plain Python types for JSON output."""
    out = {}
    for name, value in zip(result._fields, result):
        if isinstance(value, np.ndarray):
            out[name] = [float(v) for v in value]
        elif isinstance(value, (np.floating, np.integer)):
            out[name] = value.item()
        else:
            out[name] = value
    return out


def main(argv=None):
    """Parse the command line, run the requested mode and report the result."""
    args = _build_parser().parse_args(argv)
    params = vars(args).copy()
    mode = params.pop("mode")
    output = params.pop("output")
    show_stresses = params.pop("stresses")

    runners = {
        "el1": run_mode_el1,
        "el2": run_mode_el2,
        "plcap": run_mode_plcap,
        "given-forces": run_mode_given_forces,
    }

    try:
        result = runners[mode](**params)
    except (ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    _print_summary(mode, result, show_stresses)

    if output:
        with open(output, "w", encoding="utf-8") as handle:
            json.dump({"mode": mode, "inputs": params,
                       "results": _to_serialisable(result)},
                      handle, indent=2)
        print(f"Results written to {output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
