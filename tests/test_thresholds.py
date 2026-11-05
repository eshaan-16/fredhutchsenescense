import numpy as np
import pytest

from senescence.metrics import auc, control_separation, oriented_score
from senescence.thresholds import ThresholdRule, call_positive, default_grid


def test_sd_cutoff_matches_hand_calculation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu, sd = x.mean(), x.std(ddof=1)
    assert ThresholdRule("sd", 2.0).cutoff(x) == pytest.approx(mu + 2 * sd)


def test_down_marker_mirrors_the_cutoff():
    """A 'down' marker's cutoff must land below the control mean, not above."""
    x = np.random.default_rng(0).normal(10, 1, 500)
    up = ThresholdRule("sd", 2.0).cutoff(x, "up")
    down = ThresholdRule("sd", 2.0).cutoff(x, "down")
    assert down < x.mean() < up
    # and symmetric about the mean
    assert (up - x.mean()) == pytest.approx(x.mean() - down)


def test_percentile_direction_flips_the_quantile():
    x = np.arange(100.0)
    assert ThresholdRule("percentile", 95).cutoff(x, "up") == pytest.approx(
        np.percentile(x, 95)
    )
    assert ThresholdRule("percentile", 95).cutoff(x, "down") == pytest.approx(
        np.percentile(x, 5)
    )


def test_mad_resists_outliers_that_wreck_sd():
    """The motivating case: a few bright artifacts in a negative control."""
    clean = np.random.default_rng(1).normal(10, 1, 1000)
    dirty = np.concatenate([clean, [500.0, 600.0, 700.0]])
    sd_shift = abs(
        ThresholdRule("sd", 2).cutoff(dirty) - ThresholdRule("sd", 2).cutoff(clean)
    )
    mad_shift = abs(
        ThresholdRule("mad", 2).cutoff(dirty)
        - ThresholdRule("mad", 2).cutoff(clean)
    )
    assert mad_shift < sd_shift / 10


def test_percentile_cutoff_yields_expected_control_positive_rate():
    x = np.random.default_rng(2).normal(0, 1, 20000)
    cut = ThresholdRule("percentile", 95).cutoff(x)
    rate = call_positive(x, cut).mean()
    assert rate == pytest.approx(0.05, abs=0.01)


def test_nonfinite_values_are_called_negative_not_nan():
    v = np.array([1.0, np.nan, 10.0, np.inf])
    calls = call_positive(v, 5.0, "up")
    assert calls.dtype == bool
    assert calls.tolist() == [False, False, True, False]


def test_too_few_controls_raises():
    with pytest.raises(ValueError, match="at least 2"):
        ThresholdRule("sd", 2.0).cutoff(np.array([1.0]))


def test_bad_percentile_raises():
    with pytest.raises(ValueError, match="percentile k"):
        ThresholdRule("percentile", 150).cutoff(np.arange(10.0))


def test_grid_spans_the_documented_ranges():
    assert default_grid("sd")[0].k == pytest.approx(1.0)
    assert default_grid("sd")[-1].k == pytest.approx(3.0)
    assert default_grid("percentile")[0].k == pytest.approx(90.0)
    assert default_grid("percentile")[-1].k == pytest.approx(99.0)


def test_youden_j_is_zero_for_a_coin_flip_and_one_for_perfect():
    rng = np.random.default_rng(3)
    coin_p = rng.random(1000) < 0.5
    coin_n = rng.random(1000) < 0.5
    assert control_separation(coin_p, coin_n).youden_j == pytest.approx(0, abs=0.06)
    perfect = control_separation(np.ones(50, bool), np.zeros(50, bool))
    assert perfect.youden_j == pytest.approx(1.0)


def test_auc_handles_ties_and_matches_known_cases():
    assert auc(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == pytest.approx(0.5)
    assert auc(np.array([10.0, 11]), np.array([1.0, 2])) == pytest.approx(1.0)
    # all tied -> 0.5, which a naive threshold sweep gets wrong
    assert auc(np.ones(10), np.ones(10)) == pytest.approx(0.5)


def test_oriented_score_fixes_a_down_marker():
    rng = np.random.default_rng(4)
    # senescent cells have LOWER LMNB1
    pos, neg = rng.normal(5, 1, 500), rng.normal(10, 1, 500)
    assert auc(pos, neg) < 0.1  # looks anti-correlated raw
    assert auc(oriented_score(pos, "down"), oriented_score(neg, "down")) > 0.9
