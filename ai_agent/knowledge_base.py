
from pathlib import Path
import json
import pandas as pd
from scipy.stats import beta, betabinom

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "credit_with_groups.csv"

TABLES_PATH = ROOT / "results" / "tables"

RISK_GROUPS = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]


def load_credit_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required = {"default", "risk_group"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing dataset columns: {sorted(missing)}"
        )

    return df


def get_dataset_summary():
    df = load_credit_data()

    total = len(df)
    defaults = int(df["default"].sum())
    non_defaults = total - defaults

    result = {
        "total_customers": total,
        "observed_defaults": defaults,
        "non_defaults": non_defaults,
        "observed_default_rate": round(defaults / total, 6),
        "features": len(df.columns),
        "missing_values": int(df.isna().sum().sum())
    }

    if "age" in df.columns:
        result["mean_age"] = round(
            float(df["age"].mean()),
            2
        )

    if "limit_bal" in df.columns:
        result["mean_credit_limit_ntd"] = round(
            float(df["limit_bal"].mean()),
            2
        )

    return result


def get_risk_group_posterior(
    risk_group=None,
    prior_alpha=1,
    prior_beta=1
):
    df = load_credit_data()

    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError(
            "Prior parameters must be positive."
        )

    groups = (
        RISK_GROUPS
        if risk_group is None
        else [risk_group]
    )

    results = []

    for group in groups:
        subset = df[df["risk_group"] == group]

        if subset.empty:
            raise ValueError(
                f"Unknown or empty risk group: {group}"
            )

        n = len(subset)
        y = int(subset["default"].sum())

        a_post = prior_alpha + y
        b_post = prior_beta + n - y

        posterior_mean = a_post / (a_post + b_post)

        lower, upper = beta.ppf(
            [0.025, 0.975],
            a_post,
            b_post
        )

        results.append({
            "risk_group": group,
            "customers": n,
            "observed_defaults": y,
            "observed_default_rate": round(y / n, 6),
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "posterior_alpha": a_post,
            "posterior_beta": b_post,
            "posterior_pd": round(posterior_mean, 6),
            "credible_interval_95": {
                "lower": round(float(lower), 6),
                "upper": round(float(upper), 6)
            }
        })

    return {
        "model": "Beta-Binomial",
        "results": results
    }


def predict_defaults(
    risk_group,
    future_customers=1000,
    prior_alpha=1,
    prior_beta=1
):
    if not isinstance(future_customers, int):
        raise ValueError(
            "future_customers must be an integer."
        )

    if not 1 <= future_customers <= 100000:
        raise ValueError(
            "future_customers must be between 1 and 100000."
        )

    posterior = get_risk_group_posterior(
        risk_group=risk_group,
        prior_alpha=prior_alpha,
        prior_beta=prior_beta
    )["results"][0]

    a = posterior["posterior_alpha"]
    b = posterior["posterior_beta"]

    distribution = betabinom(
        future_customers,
        a,
        b
    )

    expected_defaults = float(distribution.mean())

    lower = int(distribution.ppf(0.025))
    upper = int(distribution.ppf(0.975))

    threshold = int(0.3 * future_customers)

    probability_over_30 = float(
        distribution.sf(threshold)
    )

    return {
        "risk_group": risk_group,
        "future_customers": future_customers,
        "posterior_pd": posterior["posterior_pd"],
        "expected_defaults": round(expected_defaults, 3),
        "predictive_interval_95": {
            "lower": lower,
            "upper": upper
        },
        "probability_more_than_30_percent_defaults": round(
            probability_over_30,
            6
        ),
        "predictive_distribution": "Beta-Binomial",
        "assumption": (
            "Future customers are exchangeable with "
            "historical customers in the selected group."
        )
    }


def compare_risk_groups():
    posterior = get_risk_group_posterior()["results"]

    ordered = sorted(
        posterior,
        key=lambda item: item["posterior_pd"]
    )

    comparisons = []

    for i in range(len(ordered) - 1):
        lower = ordered[i]
        higher = ordered[i + 1]

        comparisons.append({
            "lower_risk_group": lower["risk_group"],
            "higher_risk_group": higher["risk_group"],
            "posterior_pd_difference_pp": round(
                (
                    higher["posterior_pd"]
                    - lower["posterior_pd"]
                ) * 100,
                4
            )
        })

    return {
        "groups_ordered_by_risk": [
            item["risk_group"]
            for item in ordered
        ],
        "comparisons": comparisons
    }


def get_model_diagnostics():
    files = sorted(
        TABLES_PATH.glob("*mcmc*json")
    )

    reports = {}

    for path in files:
        with open(path, encoding="utf-8") as file:
            reports[path.name] = json.load(file)

    return {
        "reports": reports,
        "source": "Saved MCMC diagnostic reports",
        "note": (
            "Diagnostics describe sampler behavior, "
            "not proof of predictive model validity."
        )
    }


def get_available_tools():
    return {
        "get_dataset_summary": get_dataset_summary,
        "get_risk_group_posterior": get_risk_group_posterior,
        "predict_defaults": predict_defaults,
        "compare_risk_groups": compare_risk_groups,
        "get_model_diagnostics": get_model_diagnostics
    }
