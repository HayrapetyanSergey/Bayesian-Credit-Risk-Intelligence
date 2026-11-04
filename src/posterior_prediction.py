
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import betabinom

ROOT = Path(__file__).resolve().parents[1]
TABLES_PATH = ROOT / "results" / "tables"
FIGURES_PATH = ROOT / "results" / "figures"

TABLES_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SIMULATIONS = 100000
FUTURE_CUSTOMERS = [100, 500, 1000]
CREDIBLE_LEVEL = 0.95

RISK_GROUPS = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]

COLORS = {
    "No reported delay": "#2563eb",
    "1-month delay": "#f59e0b",
    "2+ months delay": "#dc2626"
}

rng = np.random.default_rng(SEED)

posterior = pd.read_csv(
    TABLES_PATH / "bayesian_posterior_summary.csv"
).set_index("risk_group")

if not set(RISK_GROUPS).issubset(posterior.index):
    raise ValueError("One or more posterior groups are missing")

records = []
simulations = {}

for group in RISK_GROUPS:
    alpha = float(posterior.loc[group, "posterior_alpha"])
    beta = float(posterior.loc[group, "posterior_beta"])

    for n_future in FUTURE_CUSTOMERS:
        predictive = betabinom(
            n_future,
            alpha,
            beta
        )

        expected_defaults = float(predictive.mean())
        predictive_sd = float(predictive.std())

        lower, upper = predictive.ppf([
            (1 - CREDIBLE_LEVEL) / 2,
            1 - (1 - CREDIBLE_LEVEL) / 2
        ])

        posterior_pd = rng.beta(
            alpha,
            beta,
            size=N_SIMULATIONS
        )

        predicted_defaults = rng.binomial(
            n_future,
            posterior_pd
        )

        simulations[(group, n_future)] = predicted_defaults

        records.append({
            "risk_group": group,
            "future_customers": n_future,
            "posterior_mean_pd": alpha / (alpha + beta),
            "expected_defaults": expected_defaults,
            "predictive_sd": predictive_sd,
            "predictive_lower_95": int(lower),
            "predictive_upper_95": int(upper),
            "expected_default_rate_pct": (
                100 * expected_defaults / n_future
            ),
            "simulated_mean": float(
                np.mean(predicted_defaults)
            ),
            "simulated_sd": float(
                np.std(predicted_defaults)
            ),
            "prob_more_than_30pct_default": float(
                np.mean(predicted_defaults > 0.30 * n_future)
            )
        })

results = pd.DataFrame(records)

results.to_csv(
    TABLES_PATH / "posterior_predictive_summary.csv",
    index=False
)

fig, axes = plt.subplots(
    3,
    1,
    figsize=(12, 12)
)

for ax, group in zip(axes, RISK_GROUPS):
    samples = simulations[(group, 1000)]

    counts = np.bincount(
        samples,
        minlength=1001
    )

    probabilities = counts / len(samples)

    indices = np.flatnonzero(probabilities)

    ax.bar(
        indices,
        probabilities[indices],
        width=1.0,
        color=COLORS[group],
        alpha=0.8
    )

    row = results[
        (results["risk_group"] == group) &
        (results["future_customers"] == 1000)
    ].iloc[0]

    ax.axvline(
        row["expected_defaults"],
        color="#111827",
        linestyle="--",
        linewidth=2,
        label=f"Expected: {row['expected_defaults']:.1f}"
    )

    ax.axvspan(
        row["predictive_lower_95"],
        row["predictive_upper_95"],
        alpha=0.12,
        color=COLORS[group],
        label="95% Predictive Interval"
    )

    ax.set_title(
        group,
        fontsize=13,
        fontweight="bold"
    )

    ax.set_xlabel("Number of Future Defaults")
    ax.set_ylabel("Predictive Probability")
    ax.legend()

fig.suptitle(
    "Posterior Predictive Distributions for 1,000 Customers per Group",
    fontsize=16,
    fontweight="bold",
    y=1.005
)

fig.tight_layout()

fig.savefig(
    FIGURES_PATH / "10_posterior_predictive_distributions.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 7))

scenario = results[
    results["future_customers"] == 1000
].set_index("risk_group").loc[RISK_GROUPS]

positions = np.arange(len(RISK_GROUPS))

means = scenario["expected_defaults"].to_numpy()
lower = scenario["predictive_lower_95"].to_numpy()
upper = scenario["predictive_upper_95"].to_numpy()

ax.barh(
    positions,
    means,
    color=[COLORS[g] for g in RISK_GROUPS],
    alpha=0.85,
    height=0.55
)

ax.errorbar(
    means,
    positions,
    xerr=[means - lower, upper - means],
    fmt="none",
    ecolor="#111827",
    capsize=7,
    linewidth=2
)

ax.set_yticks(positions)
ax.set_yticklabels(RISK_GROUPS)
ax.invert_yaxis()

ax.set_xlabel("Expected Number of Defaults")
ax.set_title(
    "Expected Defaults and 95% Predictive Intervals",
    fontsize=16,
    fontweight="bold",
    pad=18
)

fig.tight_layout()

fig.savefig(
    FIGURES_PATH / "11_predictive_risk_comparison.png",
    dpi=220
)

plt.close(fig)

report = {
    "model": "Beta-Binomial Posterior Predictive",
    "random_seed": SEED,
    "simulations_per_scenario": N_SIMULATIONS,
    "future_customer_scenarios": FUTURE_CUSTOMERS,
    "predictive_interval_level": CREDIBLE_LEVEL,
    "prediction_type": "Group-level future default counts",
    "assumption": (
        "Future customers are exchangeable with the observed "
        "customers within each repayment-history group"
    ),
    "results": results.to_dict(orient="records")
}

with open(
    TABLES_PATH / "posterior_predictive_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nPOSTERIOR PREDICTIVE ANALYSIS COMPLETED")
print("=" * 85)

print("\nPREDICTIVE SUMMARY")
print(
    results[
        [
            "risk_group",
            "future_customers",
            "expected_defaults",
            "predictive_lower_95",
            "predictive_upper_95",
            "prob_more_than_30pct_default"
        ]
    ].to_string(index=False)
)

print("\nSIMULATION VALIDATION")

results["simulation_mean_difference"] = abs(
    results["simulated_mean"] - results["expected_defaults"]
)

print(
    results[
        [
            "risk_group",
            "future_customers",
            "expected_defaults",
            "simulated_mean",
            "simulation_mean_difference"
        ]
    ].to_string(index=False)
)

print("\nGenerated figures:")
print("10_posterior_predictive_distributions.png")
print("11_predictive_risk_comparison.png")

print("\nPosterior predictive analysis completed successfully.")