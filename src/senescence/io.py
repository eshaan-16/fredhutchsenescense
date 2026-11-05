"""Reading the lab's Excel sheets, with validation up front.

Excel from a wet lab arrives with merged header rows, trailing blank rows,
stray totals, and marker columns named slightly differently from last time. The
schema check here exists so those fail loudly at ingest instead of propagating
into a percent-senescent number that looks fine.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from .panel import Panel


@dataclasses.dataclass
class IngestReport:
    """What ingest found, so it can be printed or written to results/."""

    path: str
    n_rows_raw: int
    n_rows_kept: int
    conditions: dict[str, int]
    dropped_nonfinite: dict[str, int]
    warnings: list[str]

    def summary(self) -> str:
        lines = [
            f"ingested {self.path}",
            f"  rows: {self.n_rows_raw} raw -> {self.n_rows_kept} kept",
            "  conditions: "
            + ", ".join(f"{k}={v}" for k, v in sorted(self.conditions.items())),
        ]
        bad = {k: v for k, v in self.dropped_nonfinite.items() if v}
        if bad:
            lines.append(
                "  missing/non-finite values per marker: "
                + ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
            )
        for w in self.warnings:
            lines.append(f"  WARNING {w}")
        return "\n".join(lines)


def read_sheet(
    path: str | Path,
    panel: Panel,
    sheet: str | int = 0,
    header_row: int = 0,
) -> tuple[pd.DataFrame, IngestReport]:
    """Read one sheet and validate it against the panel.

    Returns the tidy frame plus a report. Raises on a missing marker column or
    a missing control, since neither can be worked around downstream.
    """
    path = str(path)
    if path.lower().endswith((".csv", ".tsv")):
        sep = "\t" if path.lower().endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep, header=header_row)
    else:
        df = pd.read_excel(path, sheet_name=sheet, header=header_row)
    n_raw = len(df)
    warnings: list[str] = []

    df.columns = [str(c).strip() for c in df.columns]

    g = panel.grouping
    needed = [m.column for m in panel.markers] + [g.condition_column]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path}: missing required columns {missing}.\n"
            f"columns present: {list(df.columns)}\n"
            f"Fix the names in config/panel.yaml rather than renaming the "
            f"source sheet, so the sheet stays byte-identical to what the lab "
            f"sent."
        )

    for optional in (g.replicate_column, g.batch_column, g.cell_id_column):
        if optional not in df.columns:
            warnings.append(
                f"no {optional!r} column; batch and replicate structure cannot "
                f"be modeled, so experiment 3 will treat all cells as one group"
            )

    # Coerce marker columns to numeric. Text like "n/a" or "<LOD" becomes NaN
    # rather than silently making the whole column dtype=object.
    # Counted AFTER coercion as a total, not as a coercion delta: pandas
    # already maps "n/a", "NA" and friends to NaN while reading, so a delta
    # would under-report. What the analyst needs to know is simply how many
    # cells lack a usable value in each channel.
    dropped: dict[str, int] = {}
    for m in panel.markers:
        df[m.column] = pd.to_numeric(df[m.column], errors="coerce")
        dropped[m.name] = int((~np.isfinite(df[m.column])).sum())

    df[g.condition_column] = df[g.condition_column].astype(str).str.strip()

    # Drop rows with no marker data at all (trailing blanks, totals rows).
    marker_cols = [m.column for m in panel.markers]
    all_nan = df[marker_cols].isna().all(axis=1)
    if all_nan.any():
        warnings.append(f"dropped {int(all_nan.sum())} rows with no marker data")
        df = df.loc[~all_nan].reset_index(drop=True)

    counts = df[g.condition_column].value_counts().to_dict()

    for cond in panel.negative_controls:
        if counts.get(cond, 0) == 0:
            raise ValueError(
                f"{path}: negative control {cond!r} not found in column "
                f"{g.condition_column!r}. Present: {sorted(counts)}. Every "
                f"threshold here is calibrated on the negative control, so "
                f"this cannot proceed."
            )
    if panel.positive_controls and not any(
        counts.get(c, 0) for c in panel.positive_controls
    ):
        warnings.append(
            "no positive-control rows found; control-separation scoring "
            "(experiments 1 and 2) will not be available"
        )

    report = IngestReport(
        path=path,
        n_rows_raw=n_raw,
        n_rows_kept=len(df),
        conditions={str(k): int(v) for k, v in counts.items()},
        dropped_nonfinite=dropped,
        warnings=warnings,
    )
    return df, report


def marker_values(
    df: pd.DataFrame, panel: Panel, marker_name: str, conditions: list[str]
) -> np.ndarray:
    """Finite values of one marker, restricted to the given conditions.

    Applies the panel's log transform so thresholds are set on the scale the
    panel declares. Mixing scales between cutoff and call is a silent error,
    so this is the only place the transform is applied.
    """
    m = panel[marker_name]
    g = panel.grouping
    sel = df[df[g.condition_column].isin(conditions)]
    v = pd.to_numeric(sel[m.column], errors="coerce").to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if panel.log_transform:
        v = np.log(np.clip(v, 0, None) + panel.log_pseudocount)
    return v


def transformed_frame(df: pd.DataFrame, panel: Panel) -> pd.DataFrame:
    """Copy of df with marker columns on the panel's analysis scale."""
    out = df.copy()
    if panel.log_transform:
        for m in panel.markers:
            out[m.column] = np.log(
                np.clip(pd.to_numeric(out[m.column], errors="coerce"), 0, None)
                + panel.log_pseudocount
            )
    return out
