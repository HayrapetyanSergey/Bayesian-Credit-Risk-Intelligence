
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta, betabinom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ai_agent"))

from knowledge_base import (
    get_dataset_summary,
    get_risk_group_posterior,
    predict_defaults,
    compare_risk_groups,
    get_model_diagnostics
)


def test_dataset_size():
    result = get_dataset_summary()

    assert result["total_customers"] == 30000
    assert result["observed_defaults"] == 6636
    assert result["non_defaults"] == 23364
    assert result["missing_values"] == 0


def test_default_rate():
    result = get_dataset_summary()

    expected = 6636 / 30000

    assert np.isclose(
        result["observed_default_rate"],
        expected,
        atol=1e-6
    )


def test_posterior_parameters():
    result = get_risk_group_posterior(
        risk_group="2+ months delay"
    )["results"][0]

    assert result["posterior_alpha"] == 2178
    assert result["posterior_beta"] == 954


def test_posterior_mean():
    result = get_risk_group_posterior(
        risk_group="2+ months delay"
    )["results"][0]

    expected = 2178 / (2178 + 954)

    assert np.isclose(
        result["posterior_pd"],
        expected,
        atol=1e-6
    )


def test_credible_interval():
    result = get_risk_group_posterior(
        risk_group="2+ months delay"
    )["results"][0]

    lower, upper = beta.ppf(
        [0.025, 0.975],
        2178,
        954
    )

    assert np.isclose(
        result["credible_interval_95"]["lower"],
        lower,
        atol=1e-6
    )

    assert np.isclose(
        result["credible_interval_95"]["upper"],
        upper,
        atol=1e-6
    )


def test_posterior_risk_ordering():
    result = get_risk_group_posterior()["results"]

    probabilities = [
        item["posterior_pd"]
        for item in result
    ]

    assert probabilities[0] < probabilities[1]
    assert probabilities[1] < probabilities[2]


def test_predictive_mean():
    result = predict_defaults(
        risk_group="2+ months delay",
        future_customers=500
    )

    expected = 500 * 2178 / (2178 + 954)

    assert np.isclose(
        result["expected_defaults"],
        expected,
        atol=1e-3
    )


def test_predictive_interval():
    result = predict_defaults(
        risk_group="2+ months delay",
        future_customers=500
    )

    expected_lower = betabinom.ppf(
        0.025,
        500,
        2178,
        954
    )

    expected_upper = betabinom.ppf(
        0.975,
        500,
        2178,
        954
    )

    assert result["predictive_interval_95"]["lower"] == expected_lower
    assert result["predictive_interval_95"]["upper"] == expected_upper


def test_invalid_prior():
    with pytest.raises(ValueError):
        get_risk_group_posterior(
            risk_group="2+ months delay",
            prior_alpha=-1
        )


def test_invalid_prediction_size():
    with pytest.raises(ValueError):
        predict_defaults(
            risk_group="2+ months delay",
            future_customers=-10
        )


def test_risk_comparison():
    result = compare_risk_groups()

    assert result["groups_ordered_by_risk"] == [
        "No reported delay",
        "1-month delay",
        "2+ months delay"
    ]


def test_mcmc_diagnostics():
    result = get_model_diagnostics()

    report = result["reports"]["mcmc_report.json"]

    assert report["chains"] == 4
    assert report["maximum_rhat"] < 1.01
    assert report["minimum_bulk_ess"] > 400
    assert report["diagnostics_passed"] is True
