"""The three Monte Carlo experiments.

Experiment 1  threshold sensitivity
    Sweep the cutoff rule across its plausible range and watch what happens to
    percent-senescent and to control separation. The deliverable is not a
    single number but a stability statement: over the whole cutoff space, does
    the effect survive, and how much does the headline figure move. An effect
    that exists only at sd(1.5) is a different scientific claim from one that
    holds from sd(1.0) to percentile(99).

Experiment 2  panel-rule search
    Scan combination rules (AND / OR / m-of-n over marker subsets) crossed with
    cutoff rules, scoring each against the controls by Youden's J. Answers
    which panel definition actually separates induced from uninduced cells,
    rather than inheriting a definition from whichever paper was read last.

Experiment 3  power and noise
    Resample from the control distributions under an assumed effect, with a
    plate-level random gain, and ask how many cells and replicates are needed
    to detect a given shift in percent-senescent. This is the one to produce
    first, because it constrains the experimental design rather than describing
    data already collected.

All three take a seeded Generator and return tidy DataFrames, so results are
reproducible and go straight into plots or a sheet for the lab.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics, rules as rulemod, thresholds as thr
from .panel import Panel


# ---------------------------------------------------------------- experiment 1

def threshold_sensitivity(
    df: pd.DataFrame,
    panel: Panel,
    test_conditions: list[str] | None = None,
    grid: list[thr.ThresholdRule] | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Sweep cutoff rules; bootstrap the control to get a CI on each point.

    The bootstrap resamples the NEGATIVE CONTROL, not the test condition. That
    is the point: the uncertainty being quantified is uncertainty in where the
    line gets drawn, which is what propagates into percent-senescent.
    """
    from .io import marker_values

    rng = np.random.default_rng(seed)
    grid = grid or thr.all_families_grid()
    g = panel.grouping
    neg = list(panel.negative_controls)
    pos = list(panel.positive_controls)

    if test_conditions is None:
        known = set(neg) | set(pos)
        test_conditions = [
            c for c in df[g.condition_column].unique() if c not in known
        ]

    rows = []
    for marker in panel.markers:
        ctrl = marker_values(df, panel, marker.name, neg)
        pos_v = marker_values(df, panel, marker.name, pos) if pos else np.array([])
        test_v = {
            c: marker_values(df, panel, marker.name, [c]) for c in test_conditions
        }

        for rule in grid:
            cut = rule.cutoff(ctrl, marker.direction)

            rec: dict[str, object] = {
                "marker": marker.name,
                "direction": marker.direction,
                "family": rule.family,
                "k": rule.k,
                "rule": str(rule),
                "cutoff": cut,
                "pct_neg_control": metrics.pct_senescent(
                    thr.call_positive(ctrl, cut, marker.direction)
                ),
            }

            if pos_v.size:
                sep = metrics.control_separation(
                    thr.call_positive(pos_v, cut, marker.direction),
                    thr.call_positive(ctrl, cut, marker.direction),
                )
                rec.update(
                    pct_pos_control=sep.pct_pos_control,
                    sensitivity=sep.sensitivity,
                    specificity=sep.specificity,
                    youden_j=sep.youden_j,
                )

            for c, v in test_v.items():
                if v.size:
                    rec[f"pct_{c}"] = metrics.pct_senescent(
                        thr.call_positive(v, cut, marker.direction)
                    )

            # Bootstrap the control to get a CI on the cutoff itself, and
            # propagate it to percent-senescent in the positive control.
            if n_bootstrap and ctrl.size >= 2:
                cuts = np.empty(n_bootstrap)
                pcts = np.empty(n_bootstrap)
                for b in range(n_bootstrap):
                    boot = rng.choice(ctrl, size=ctrl.size, replace=True)
                    cb = rule.cutoff(boot, marker.direction)
                    cuts[b] = cb
                    pcts[b] = (
                        metrics.pct_senescent(
                            thr.call_positive(pos_v, cb, marker.direction)
                        )
                        if pos_v.size
                        else np.nan
                    )
                rec["cutoff_lo"] = float(np.percentile(cuts, 2.5))
                rec["cutoff_hi"] = float(np.percentile(cuts, 97.5))
                if pos_v.size:
                    rec["pct_pos_lo"] = float(np.nanpercentile(pcts, 2.5))
                    rec["pct_pos_hi"] = float(np.nanpercentile(pcts, 97.5))

            rows.append(rec)

    return pd.DataFrame(rows)


def stability_summary(sens: pd.DataFrame) -> pd.DataFrame:
    """Collapse the sweep into one row per marker: how much does it move?

    `pct_pos_range` is the headline. A marker whose positive-control
    percent-senescent spans 20 points across the cutoff grid is not a marker
    you can quote a single percentage for.
    """
    if "youden_j" not in sens.columns:
        return (
            sens.groupby("marker")
            .agg(
                cutoff_min=("cutoff", "min"),
                cutoff_max=("cutoff", "max"),
                n_rules=("rule", "nunique"),
            )
            .reset_index()
        )
    out = (
        sens.groupby("marker")
        .agg(
            best_rule=("youden_j", lambda s: sens.loc[s.idxmax(), "rule"]),
            best_j=("youden_j", "max"),
            worst_j=("youden_j", "min"),
            median_j=("youden_j", "median"),
            pct_pos_min=("pct_pos_control", "min"),
            pct_pos_max=("pct_pos_control", "max"),
            n_rules=("rule", "nunique"),
        )
        .reset_index()
    )
    out["j_range"] = out["best_j"] - out["worst_j"]
    out["pct_pos_range"] = out["pct_pos_max"] - out["pct_pos_min"]
    return out.sort_values("best_j", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- experiment 2

def panel_rule_search(
    df: pd.DataFrame,
    panel: Panel,
    n_draws: int = 2000,
    max_subset: int | None = None,
    exhaustive_cutoffs: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Monte Carlo over (panel rule) x (per-marker cutoff rule).

    Two modes. With `exhaustive_cutoffs=False` (default) each draw samples one
    cutoff rule per marker independently, which explores the realistic case
    where each channel gets thresholded on its own terms. With True, one shared
    cutoff rule is applied to every marker, which is the stricter convention
    some reviewers prefer, and the resulting space is small enough to enumerate.
    """
    from .io import marker_values

    rng = np.random.default_rng(seed)
    neg, pos = list(panel.negative_controls), list(panel.positive_controls)
    if not pos:
        raise ValueError(
            "panel-rule search scores rules by control separation, which needs "
            "a positive control. None is defined in the panel."
        )

    ctrl = {m.name: marker_values(df, panel, m.name, neg) for m in panel.markers}
    posv = {m.name: marker_values(df, panel, m.name, pos) for m in panel.markers}

    names = [m.name for m in panel.markers]
    rule_pool = rulemod.enumerate_rules(names, max_subset=max_subset)

    if exhaustive_cutoffs:
        combos = [
            (pr, cr) for pr in rule_pool for cr in thr.all_families_grid(9)
        ]
        draws = [(pr, {n: cr for n in names}) for pr, cr in combos]
    else:
        draws = []
        for _ in range(n_draws):
            pr = rule_pool[int(rng.integers(len(rule_pool)))]
            draws.append((pr, {n: thr.sample_rule(rng) for n in names}))

    rows = []
    for panel_rule, cut_rules in draws:
        pos_calls, neg_calls, cuts = {}, {}, {}
        ok = True
        for name in panel_rule.markers:
            m = panel[name]
            c, p = ctrl[name], posv[name]
            if c.size < 2 or p.size == 0:
                ok = False
                break
            cut = cut_rules[name].cutoff(c, m.direction)
            cuts[name] = cut
            neg_calls[name] = thr.call_positive(c, cut, m.direction)
            pos_calls[name] = thr.call_positive(p, cut, m.direction)
        if not ok:
            continue

        sep = metrics.control_separation(
            panel_rule.apply(pos_calls), panel_rule.apply(neg_calls)
        )
        rows.append(
            {
                "panel_rule": str(panel_rule),
                "logic": panel_rule.logic,
                "n_markers": panel_rule.n_markers,
                "m": panel_rule.m if panel_rule.logic == "mofn" else None,
                "cutoff_rules": "; ".join(
                    f"{n}={cut_rules[n]}" for n in panel_rule.markers
                ),
                "sensitivity": sep.sensitivity,
                "specificity": sep.specificity,
                "youden_j": sep.youden_j,
                "pct_pos_control": sep.pct_pos_control,
                "pct_neg_control": sep.pct_neg_control,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values("youden_j", ascending=False).reset_index(drop=True)


def rule_robustness(search: pd.DataFrame, min_draws: int = 5) -> pd.DataFrame:
    """Rank panel rules by how they do ACROSS cutoff draws, not at their best.

    A rule whose median J is high is robust to where the lines get drawn. A
    rule whose max J is high but median is poor only works at one hand-picked
    cutoff, and that is the kind of result that fails to replicate.
    """
    grp = search.groupby("panel_rule").agg(
        n_draws=("youden_j", "size"),
        median_j=("youden_j", "median"),
        p10_j=("youden_j", lambda s: float(np.percentile(s, 10))),
        max_j=("youden_j", "max"),
        median_sens=("sensitivity", "median"),
        median_spec=("specificity", "median"),
        n_markers=("n_markers", "first"),
        logic=("logic", "first"),
    )
    grp = grp[grp["n_draws"] >= min_draws]
    return grp.sort_values("median_j", ascending=False).reset_index()


# ---------------------------------------------------------------- experiment 3

def _safe_ttest_p(a: list[float], b: list[float]) -> float:
    """Two-sample p-value that handles the degenerate zero-variance case.

    This is not defensive boilerplate, it is a real situation in this design.
    When the cutoff sits far out in the tail and cells-per-well is small, every
    replicate can return exactly the same percentage (often 0.0), so both arms
    have zero variance and scipy's t-test warns about catastrophic cancellation
    while returning a meaningless value.

    The correct reading: with no within-arm variance, a difference in means is
    separation with no evidence about its reliability, and no difference is no
    effect. Returning 1.0 for the identical case keeps the null calibrated
    rather than counting a numerical artifact as a hit.
    """
    import numpy as _np
    from scipy import stats as _stats

    av, bv = _np.asarray(a, dtype=float), _np.asarray(b, dtype=float)
    if av.size < 2 or bv.size < 2:
        return float("nan")
    if av.var(ddof=1) == 0 and bv.var(ddof=1) == 0:
        # identical constants -> no effect; different constants -> undefined
        # variance, so decline to call it significant
        return 1.0
    with _np.errstate(all="ignore"):
        import warnings as _warnings

        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            _, p = _stats.ttest_ind(av, bv)
    return float(p)


def power_curve(
    df: pd.DataFrame,
    panel: Panel,
    marker: str,
    panel_rule: rulemod.PanelRule | None = None,
    cutoff_rule: thr.ThresholdRule | None = None,
    cells_per_well: tuple[int, ...] = (50, 100, 200, 400, 800),
    n_replicates: tuple[int, ...] = (3, 4, 6, 8),
    true_shift_pp: float = 10.0,
    plate_gain_sd: float = 0.18,
    alpha: float = 0.05,
    n_sims: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulated power to detect a shift in percent-senescent.

    The design simulated is the realistic one: cells nested in wells, wells
    nested in plates, with a per-plate multiplicative gain shared across cells
    on that plate. The test is a two-sample t-test on the per-replicate
    percent-senescent values, which is what a wet lab will actually run.

    `plate_gain_sd` is the parameter that matters. Set it to 0 and the power
    curve is optimistic in a way that will not survive contact with real
    plates; the nested gain is precisely what eats the apparent power that a
    naive per-cell calculation promises.
    """
    from .io import marker_values
    from scipy import stats  # noqa: F401  (used via _safe_ttest_p)

    rng = np.random.default_rng(seed)
    neg = list(panel.negative_controls)
    ctrl = marker_values(df, panel, marker, neg)
    if ctrl.size < 10:
        raise ValueError(
            f"need >=10 control cells for marker {marker!r} to resample from, "
            f"got {ctrl.size}"
        )
    m = panel[marker]
    cutoff_rule = cutoff_rule or thr.ThresholdRule("mad", 2.0)

    # Baseline: where does the cutoff land, and what is the control's own
    # positive rate under it. The simulated effect is a shift on top of this.
    base_cut = cutoff_rule.cutoff(ctrl, m.direction)
    base_rate = float(
        thr.call_positive(ctrl, base_cut, m.direction).mean()
    )
    target_rate = np.clip(base_rate + true_shift_pp / 100.0, 0.0, 1.0)

    rows = []
    for n_cells in cells_per_well:
        for n_reps in n_replicates:
            hits = 0
            for _ in range(n_sims):
                ctrl_pcts, test_pcts = [], []
                for _rep in range(n_reps):
                    # one plate gain per replicate pair, shared by both arms,
                    # which is the paired-plate design most labs run
                    gain = rng.normal(0.0, plate_gain_sd)
                    c_cells = rng.choice(ctrl, size=n_cells, replace=True) + gain
                    t_cells = rng.choice(ctrl, size=n_cells, replace=True) + gain
                    # induce the target fraction by pushing responders past the
                    # cutoff, drawn from the control's own upper tail
                    n_resp = rng.binomial(
                        n_cells, max(target_rate - base_rate, 0.0)
                    )
                    if n_resp:
                        idx = rng.choice(n_cells, size=n_resp, replace=False)
                        bump = np.abs(
                            rng.normal(0.0, ctrl.std(ddof=1) * 1.5, size=n_resp)
                        ) + abs(base_cut - np.median(ctrl))
                        t_cells[idx] = (
                            np.median(ctrl) - bump
                            if m.direction == "down"
                            else np.median(ctrl) + bump
                        )
                    cut = cutoff_rule.cutoff(c_cells, m.direction)
                    ctrl_pcts.append(
                        metrics.pct_senescent(
                            thr.call_positive(c_cells, cut, m.direction)
                        )
                    )
                    test_pcts.append(
                        metrics.pct_senescent(
                            thr.call_positive(t_cells, cut, m.direction)
                        )
                    )
                p = _safe_ttest_p(test_pcts, ctrl_pcts)
                if np.isfinite(p) and p < alpha:
                    hits += 1
            rows.append(
                {
                    "marker": marker,
                    "cells_per_well": n_cells,
                    "n_replicates": n_reps,
                    "true_shift_pp": true_shift_pp,
                    "plate_gain_sd": plate_gain_sd,
                    "cutoff_rule": str(cutoff_rule),
                    "baseline_pct": 100.0 * base_rate,
                    "power": hits / n_sims,
                    "n_sims": n_sims,
                }
            )
    return pd.DataFrame(rows)


def min_detectable_effect(
    df: pd.DataFrame,
    panel: Panel,
    marker: str,
    shifts_pp: tuple[float, ...] = (2.5, 5.0, 7.5, 10.0, 15.0, 20.0),
    target_power: float = 0.8,
    **kwargs,
) -> pd.DataFrame:
    """Smallest shift detectable at `target_power` for each design.

    Often the more useful framing for a PI conversation: not "what power do I
    have" but "what is the smallest difference this design can see".
    """
    frames = [
        power_curve(df, panel, marker, true_shift_pp=s, **kwargs)
        for s in shifts_pp
    ]
    allp = pd.concat(frames, ignore_index=True)
    rows = []
    for (nc, nr), grp in allp.groupby(["cells_per_well", "n_replicates"]):
        ok = grp[grp["power"] >= target_power].sort_values("true_shift_pp")
        rows.append(
            {
                "marker": marker,
                "cells_per_well": nc,
                "n_replicates": nr,
                "target_power": target_power,
                "min_detectable_pp": (
                    float(ok.iloc[0]["true_shift_pp"]) if len(ok) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["cells_per_well", "n_replicates"]
    ).reset_index(drop=True)
