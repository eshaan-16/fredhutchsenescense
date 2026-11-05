"""Command line entry points.

    python -m senescence.cli demo                      # synthetic end-to-end
    python -m senescence.cli ingest  data/raw/x.xlsx    # validate a real sheet
    python -m senescence.cli sweep   data/raw/x.xlsx    # experiment 1
    python -m senescence.cli panels  data/raw/x.xlsx    # experiment 2
    python -m senescence.cli power   data/raw/x.xlsx --marker p16   # experiment 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from . import montecarlo as mc
from .io import read_sheet
from .panel import load_panel
from .simulate import simulate_panel

RESULTS = Path(__file__).resolve().parents[2] / "results"


def _load(path: str | None, panel):
    if path is None:
        print("no sheet given, using simulated data\n")
        return simulate_panel(panel).drop(columns=["_responder"])
    df, report = read_sheet(path, panel)
    print(report.summary(), "\n")
    return df


def _save(df: pd.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / name
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="senescence")
    ap.add_argument(
        "command",
        choices=["demo", "ingest", "sweep", "panels", "power"],
    )
    ap.add_argument("sheet", nargs="?", default=None)
    ap.add_argument("--panel", default=None, help="path to panel.yaml")
    ap.add_argument("--marker", default=None)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--shift", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    panel = load_panel(args.panel)
    print(
        f"panel: {len(panel.markers)} markers "
        f"({', '.join(panel.names)}), level={panel.level}, "
        f"log_transform={panel.log_transform}"
    )
    print(
        f"controls: negative={list(panel.negative_controls)} "
        f"positive={list(panel.positive_controls)}\n"
    )

    if args.command == "ingest":
        if args.sheet is None:
            ap.error("ingest needs a sheet path")
        _load(args.sheet, panel)
        return 0

    df = _load(args.sheet, panel)

    if args.command in ("demo", "sweep"):
        sens = mc.threshold_sensitivity(df, panel, seed=args.seed)
        summary = mc.stability_summary(sens)
        print("=== experiment 1: threshold sensitivity ===")
        print(summary.to_string(index=False))
        _save(sens, "exp1_threshold_sweep.csv")
        _save(summary, "exp1_stability_summary.csv")

    if args.command in ("demo", "panels"):
        search = mc.panel_rule_search(
            df, panel, n_draws=args.draws, seed=args.seed
        )
        robust = mc.rule_robustness(search)
        print("\n=== experiment 2: panel rules, by median J across cutoffs ===")
        print(robust.head(12).to_string(index=False))
        _save(search, "exp2_rule_search.csv")
        _save(robust, "exp2_rule_robustness.csv")

    if args.command in ("demo", "power"):
        marker = args.marker or panel.names[0]
        pw = mc.power_curve(
            df,
            panel,
            marker,
            true_shift_pp=args.shift,
            n_sims=args.sims,
            seed=args.seed,
        )
        print(
            f"\n=== experiment 3: power for {marker}, "
            f"{args.shift:g} pp shift ==="
        )
        print(
            pw.pivot(
                index="cells_per_well", columns="n_replicates", values="power"
            ).to_string()
        )
        _save(pw, "exp3_power_curve.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
