"""End-to-end checks on simulated data, where ground truth is known."""

import numpy as np
import pandas as pd
import pytest

from senescence import montecarlo as mc
from senescence.io import marker_values, read_sheet
from senescence.panel import load_panel
from senescence.simulate import simulate_panel, write_example_sheet
from senescence.thresholds import ThresholdRule, call_positive


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def sim(panel):
    return simulate_panel(panel, n_cells_per_well=200, n_replicates=2, seed=7)


def test_simulation_has_the_panels_columns(panel, sim):
    for m in panel.markers:
        assert m.column in sim.columns
    g = panel.grouping
    for c in (g.condition_column, g.replicate_column, g.batch_column):
        assert c in sim.columns


def test_up_markers_rise_and_down_markers_fall_in_the_positive_control(panel, sim):
    g = panel.grouping
    pos = sim[sim[g.condition_column].isin(panel.positive_controls)]
    neg = sim[sim[g.condition_column].isin(panel.negative_controls)]
    for m in panel.markers:
        if m.direction == "up":
            assert pos[m.column].median() > neg[m.column].median()
        else:
            assert pos[m.column].median() < neg[m.column].median()


def test_recovered_positive_rate_tracks_the_simulated_responder_fraction(panel):
    """A sane cutoff should recover roughly the fraction that was induced."""
    sim = simulate_panel(
        panel, n_cells_per_well=4000, n_replicates=1, n_plates=1,
        responder_fraction=0.40, effect_log2=3.0, plate_gain_sd=0.0, seed=11,
    )
    ctrl = marker_values(sim, panel, "p16", list(panel.negative_controls))
    posv = marker_values(sim, panel, "p16", list(panel.positive_controls))
    cut = ThresholdRule("mad", 2.0).cutoff(ctrl, "up")
    rate = call_positive(posv, cut, "up").mean()
    assert 0.30 < rate < 0.50


def test_threshold_sweep_shape_and_monotonicity(panel, sim):
    sens = mc.threshold_sensitivity(sim, panel, n_bootstrap=20, seed=0)
    assert len(sens) == len(panel.markers) * 63  # 3 families x 21 points
    # within a family, raising k must make the call stricter, never looser
    for (marker, fam), grp in sens.groupby(["marker", "family"]):
        grp = grp.sort_values("k")
        pct = grp["pct_pos_control"].to_numpy()
        assert np.all(np.diff(pct) <= 1e-9), f"{marker}/{fam} not monotone"


def test_stability_summary_reports_a_range(panel, sim):
    sens = mc.threshold_sensitivity(sim, panel, n_bootstrap=0, seed=0)
    summ = mc.stability_summary(sens)
    assert set(summ["marker"]) == set(panel.names)
    assert (summ["pct_pos_range"] >= 0).all()
    assert (summ["best_j"] >= summ["median_j"]).all()


def test_panel_rule_search_prefers_informative_markers(panel, sim):
    search = mc.panel_rule_search(sim, panel, n_draws=400, max_subset=2, seed=0)
    assert len(search) > 0
    assert search["youden_j"].iloc[0] >= search["youden_j"].iloc[-1]
    robust = mc.rule_robustness(search, min_draws=3)
    assert "median_j" in robust.columns
    # the best rule should beat a coin flip on simulated data with a real effect
    assert robust["median_j"].iloc[0] > 0.2


def test_panel_search_without_positive_control_raises(panel, sim):
    import dataclasses
    nopos = dataclasses.replace(panel, positive_controls=())
    with pytest.raises(ValueError, match="needs"):
        mc.panel_rule_search(sim, nopos, n_draws=10)


def test_power_rises_with_cells_and_replicates(panel, sim):
    pw = mc.power_curve(
        sim, panel, "p16",
        cells_per_well=(50, 400), n_replicates=(3, 8),
        true_shift_pp=12.0, n_sims=60, seed=0,
    )
    assert set(pw["power"].between(0, 1)) == {True}
    lo = pw[(pw.cells_per_well == 50) & (pw.n_replicates == 3)]["power"].iloc[0]
    hi = pw[(pw.cells_per_well == 400) & (pw.n_replicates == 8)]["power"].iloc[0]
    assert hi >= lo


def test_zero_effect_power_is_near_alpha(panel, sim):
    """The null check. If this fails, the test is anticonservative."""
    pw = mc.power_curve(
        sim, panel, "p16",
        cells_per_well=(200,), n_replicates=(4,),
        true_shift_pp=0.0, plate_gain_sd=0.1, n_sims=300, seed=5,
    )
    assert pw["power"].iloc[0] < 0.15


def test_ingest_roundtrip_through_xlsx(panel, tmp_path):
    p = write_example_sheet(
        str(tmp_path / "sheet.xlsx"), panel,
        n_cells_per_well=50, n_replicates=2, n_plates=1,
    )
    df, report = read_sheet(p, panel)
    assert report.n_rows_kept == len(df) > 0
    assert set(panel.negative_controls) <= set(report.conditions)
    assert "untreated" in report.conditions


def test_ingest_rejects_a_sheet_missing_a_marker(panel, tmp_path):
    df = simulate_panel(panel, n_cells_per_well=10, n_replicates=1, n_plates=1)
    df = df.drop(columns=[panel["p21"].column])
    path = tmp_path / "bad.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        read_sheet(path, panel)


def test_ingest_rejects_a_sheet_with_no_negative_control(panel, tmp_path):
    df = simulate_panel(panel, n_cells_per_well=10, n_replicates=1, n_plates=1)
    g = panel.grouping
    df = df[~df[g.condition_column].isin(panel.negative_controls)]
    path = tmp_path / "nopos.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="negative control"):
        read_sheet(path, panel)


def test_ingest_coerces_text_junk_to_nan(panel, tmp_path):
    df = simulate_panel(panel, n_cells_per_well=20, n_replicates=1, n_plates=1)
    col = panel["p16"].column
    df[col] = df[col].astype(object)
    df.loc[0, col] = "<LOD"
    df.loc[1, col] = "n/a"
    path = tmp_path / "junk.csv"
    df.to_csv(path, index=False)
    out, report = read_sheet(path, panel)
    assert report.dropped_nonfinite["p16"] == 2
    assert pd.isna(out.loc[0, panel["p16"].column])
