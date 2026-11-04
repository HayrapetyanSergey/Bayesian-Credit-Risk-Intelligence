
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "credit_with_groups.csv"
FIGURES_PATH = ROOT / "results" / "figures"
TABLES_PATH = ROOT / "results" / "tables"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0
CREDIBLE_LEVEL = 0.95
N_SAMPLES = 100000

rng = np.random.default_rng(RANDOM_SEED)

risk_order = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]

df = pd.read_csv(DATA_PATH)

required = {"risk_group", "default"}

if not required.issubset(df.columns):
    raise ValueError("Required columns are missing")

if not df["default"].isin([0, 1]).all():
    raise ValueError("Invalid target values")

summary = (
    df.groupby("risk_group")
    .agg(
        customers=("default", "size"),
        defaults=("default", "sum")
    )
    .reindex(risk_order)
)

if summary.isna().any().any():
    raise ValueError("One or more risk groups are missing")

grid = np.linspace(0.000001, 0.999999, 2001)

posterior_samples = {}
posterior_parameters = {}
results = []
grid_differences = {}

for group, row in summary.iterrows():
    n = int(row["customers"])
    y = int(row["defaults"])

    alpha_post = PRIOR_ALPHA + y
    beta_post = PRIOR_BETA + n - y

    posterior_parameters[group] = {
        "alpha": alpha_post,
        "beta": beta_post
    }

    analytical_mean = beta.mean(
        alpha_post,
        beta_post
    )

    ci_lower, ci_upper = beta.ppf(
        [(1 - CREDIBLE_LEVEL) / 2,
         1 - (1 - CREDIBLE_LEVEL) / 2],
        alpha_post,
        beta_post
    )

    log_grid_density = beta.logpdf(
        grid,
        alpha_post,
        beta_post
    )

    grid_weights = np.exp(
        log_grid_density - np.max(log_grid_density)
    )

    grid_weights /= grid_weights.sum()

    grid_mean = np.sum(grid * grid_weights)

    grid_cdf = np.cumsum(grid_weights)

    grid_lower = np.interp(
        0.025,
        grid_cdf,
        grid
    )

    grid_upper = np.interp(
        0.975,
        grid_cdf,
        grid
    )

    samples = rng.beta(
        alpha_post,
        beta_post,
        N_SAMPLES
    )

    posterior_samples[group] = samples

    grid_differences[group] = abs(
        analytical_mean - grid_mean
    )

    results.append({
        "risk_group": group,
        "customers": n,
        "defaults": y,
        "observed_default_rate": y / n,
        "prior_alpha": PRIOR_ALPHA,
        "prior_beta": PRIOR_BETA,
        "posterior_alpha": alpha_post,
        "posterior_beta": beta_post,
        "posterior_mean": analytical_mean,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "grid_mean": grid_mean,
        "grid_ci_95_lower": grid_lower,
        "grid_ci_95_upper": grid_upper,
        "method_difference": abs(
            analytical_mean - grid_mean
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLES_PATH / "bayesian_posterior_summary.csv",
    index=False
)

if max(grid_differences.values()) > 0.001:
    raise ValueError(
        "Grid approximation differs substantially from analytical results"
    )

comparisons = []

for i in range(len(risk_order)):
    for j in range(i + 1, len(risk_order)):
        group_a = risk_order[i]
        group_b = risk_order[j]

        a = posterior_samples[group_a]
        b = posterior_samples[group_b]

        differences = b - a

        comparisons.append({
            "group_a": group_a,
            "group_b": group_b,
            "p_b_greater_than_a": float(
                np.mean(differences > 0)
            ),
            "mean_difference": float(
                np.mean(differences)
            ),
            "difference_ci_95_lower": float(
                np.quantile(differences, 0.025)
            ),
            "difference_ci_95_upper": float(
                np.quantile(differences, 0.975)
            )
        })

comparison_df = pd.DataFrame(comparisons)

comparison_df.to_csv(
    TABLES_PATH / "posterior_group_comparisons.csv",
    index=False
)

plt.style.use("seaborn-v0_8-whitegrid")

colors = ["#2563eb", "#f59e0b", "#dc2626"]

fig, ax = plt.subplots(figsize=(12, 7))

x = np.linspace(0.05, 0.85, 2000)

for group, color in zip(risk_order, colors):
    params = posterior_parameters[group]

    density = beta.pdf(
        x,
        params["alpha"],
        params["beta"]
    )

    ax.plot(
        x * 100,
        density,
        label=group,
        color=color,
        linewidth=2.5
    )

    ax.fill_between(
        x * 100,
        density,
        alpha=0.08,
        color=color
    )

ax.set_title(
    "Bayesian Posterior Distributions of Default Probability",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.set_xlabel("Default Probability (%)")
ax.set_ylabel("Posterior Density")
ax.legend(frameon=True)

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "06_bayesian_posterior_distributions.png",
    dpi=220
)
plt.close()

fig, ax = plt.subplots(figsize=(11, 6))

means = results_df["posterior_mean"].to_numpy() * 100
lower = results_df["ci_95_lower"].to_numpy() * 100
upper = results_df["ci_95_upper"].to_numpy() * 100

positions = np.arange(len(risk_order))

ax.errorbar(
    means,
    positions,
    xerr=[means - lower, upper - means],
    fmt="o",
    markersize=9,
    capsize=7,
    color="#2563eb",
    ecolor="#64748b",
    linewidth=2
)

ax.set_yticks(positions)
ax.set_yticklabels(risk_order)
ax.set_xlabel("Posterior Default Probability (%)")
ax.set_title(
    "Posterior Means and 95% Credible Intervals",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.invert_yaxis()

plt.tight_layout()
plt.savefig(
    FIGURES_PATH / "07_bayesian_credible_intervals.png",
    dpi=220
)
plt.close()

report = {
    "model": "Beta-Binomial",
    "prior": {
        "distribution": "Beta",
        "alpha": PRIOR_ALPHA,
        "beta": PRIOR_BETA
    },
    "estimation_methods": [
        "Analytical conjugate posterior",
        "Grid approximation"
    ],
    "credible_level": CREDIBLE_LEVEL,
    "random_seed": RANDOM_SEED,
    "posterior_sample_count": N_SAMPLES,
    "maximum_grid_mean_difference": float(
        max(grid_differences.values())
    ),
    "posterior_results": results_df.to_dict(
        orient="records"
    ),
    "group_comparisons": comparison_df.to_dict(
        orient="records"
    )
}

with open(
    TABLES_PATH / "bayesian_model_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        report,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nBAYESIAN MODEL COMPLETED")
print("=" * 75)

print("\nPOSTERIOR ESTIMATES")
print(
    results_df[
        [
            "risk_group",
            "posterior_mean",
            "ci_95_lower",
            "ci_95_upper",
            "grid_mean",
            "method_difference"
        ]
    ].to_string(index=False)
)

print("\nPOSTERIOR GROUP COMPARISONS")
print(
    comparison_df.to_string(index=False)
)

print("\nMaximum analytical-grid difference:")
print(max(grid_differences.values()))

print("\nBayesian model results saved successfully.")
