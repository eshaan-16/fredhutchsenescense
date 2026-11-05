"""Synthetic single-cell marker data matching the panel.

Purpose is to make the pipeline runnable and testable before the real sheets
land, and to supply null data for the power analysis. The generative model is
deliberately simple but not naive about the three things that actually break
threshold analyses on real immunofluorescence data:

  1. Log-normal intensities. IF signal is multiplicative and right-skewed.
     Thresholds set on the raw scale sit out in the bright tail.

  2. A mixture, not a shift. An induced well is not "all cells brighter". It is
     a mixture of responders and non-responders, which is why percent-senescent
     is the natural readout and why a mean-based test on well averages is
     underpowered relative to a single-cell call.

  3. Plate effects. A multiplicative per-plate gain, shared across markers
     within a plate. This is what makes a threshold calibrated on plate 1's
     control wrong for plate 3, and it is the single most common reason a
     percent-senescent number fails to replicate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .panel import Panel


def simulate_panel(
    panel: Panel,
    n_cells_per_well: int = 300,
    n_replicates: int = 3,
    n_plates: int = 2,
    responder_fraction: float = 0.45,
    effect_log2: float = 1.8,
    plate_gain_sd: float = 0.18,
    noise_sd: float = 0.45,
    baseline_log: float = 6.0,
    seed: int | None = 0,
) -> pd.DataFrame:
    """Generate a tidy per-cell frame with the panel's columns.

    Parameters that matter most for interpretation:
      responder_fraction  share of cells in a positive-control well that
                          actually enter the senescent state
      effect_log2         log2 fold-change in a responder cell
      plate_gain_sd       SD of the per-plate multiplicative gain, in log space
      noise_sd            per-cell per-marker measurement noise, in log space
    """
    rng = np.random.default_rng(seed)
    g = panel.grouping

    conditions = list(panel.negative_controls) + list(panel.positive_controls)
    if not conditions:
        raise ValueError("panel has no control conditions to simulate")

    plate_gain = rng.normal(0.0, plate_gain_sd, size=n_plates)
    rows = []
    cell_id = 0

    for plate in range(n_plates):
        for cond in conditions:
            is_pos = cond in panel.positive_controls
            for rep in range(n_replicates):
                n = n_cells_per_well
                responder = (
                    rng.random(n) < responder_fraction
                    if is_pos
                    else np.zeros(n, dtype=bool)
                )
                rec: dict[str, np.ndarray | str | int] = {}
                for marker in panel.markers:
                    # sign: an "up" marker rises in responders, "down" falls
                    sign = -1.0 if marker.direction == "down" else 1.0
                    mu = (
                        baseline_log
                        + plate_gain[plate]
                        + sign * effect_log2 * np.log(2.0) * responder
                    )
                    logv = rng.normal(mu, noise_sd)
                    rec[marker.column] = np.exp(logv)

                # Nuclear and cell area both enlarge in senescent cells, which
                # is why they are useful covariates and also why they are not
                # independent of the markers.
                area_shift = 1.0 + 0.55 * responder
                rec["nuclear_area"] = np.exp(
                    rng.normal(np.log(120.0 * area_shift), 0.22)
                )
                rec["cell_area"] = np.exp(
                    rng.normal(np.log(480.0 * area_shift), 0.28)
                )

                frame = pd.DataFrame(rec)
                frame[g.condition_column] = cond
                frame[g.replicate_column] = rep
                frame[g.batch_column] = plate
                frame[g.cell_id_column] = np.arange(cell_id, cell_id + n)
                frame["_responder"] = responder  # ground truth, for tests only
                cell_id += n
                rows.append(frame)

    return pd.concat(rows, ignore_index=True)


def write_example_sheet(path: str, panel: Panel, **kwargs) -> str:
    """Write a simulated panel to .xlsx, shaped like what the lab will send."""
    df = simulate_panel(panel, **kwargs).drop(columns=["_responder"])
    df.to_excel(path, index=False, sheet_name="per_cell")
    return path
