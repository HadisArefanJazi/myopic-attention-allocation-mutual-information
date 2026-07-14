"""Command-line entry point for mutual-information attention experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from myopic_mi.experiment import ExperimentConfig, run_experiment
from myopic_mi.plotting import save_figures, save_mat_results


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run the myopic mutual-information attention experiment.")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--gamma-start", type=float, default=0.5)
    parser.add_argument("--gamma-stop", type=float, default=2.5)
    parser.add_argument("--gamma-step", type=float, default=0.5)
    parser.add_argument("--gamma-target", type=float, default=0.5)
    parser.add_argument("--grid-step", type=float, default=0.0005)
    parser.add_argument("--lmax", type=float, default=2.0)
    parser.add_argument("--save-mat", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MI experiment from the command line."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    config = ExperimentConfig(
        horizon=args.horizon,
        gamma_start=args.gamma_start,
        gamma_stop=args.gamma_stop,
        gamma_step=args.gamma_step,
        gamma_target=args.gamma_target,
        grid_step=args.grid_step,
        lmax=args.lmax,
    )
    result = run_experiment(config)
    logging.info("Experiment finished for %s gamma value(s).", len(config.gamma_list))
    logging.info("Total trace costs: %s", result.trace_totals)

    if args.save_mat:
        path = save_mat_results(result, args.output_dir / "myopic_MI_exact_draft_results.mat")
        logging.info("Saved MATLAB results to %s.", path)
    if not args.skip_figures:
        figure_dir = save_figures(result, args.output_dir / "figures")
        logging.info("Saved figures to %s.", figure_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
