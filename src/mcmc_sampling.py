
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "credit_with_groups.csv"
TABLES_PATH = ROOT / "results" / "tables"
FIGURES_PATH = ROOT / "results" / "figures"
SAMPLES_PATH = ROOT / "results" / "posterior_samples"

for path in [TABLES_PATH, FIGURES_PATH, SAMPLES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
N_CHAINS = 4
N_TUNE = 1500
N_DRAWS = 3000
PRIOR_ALPHA = 1.0
PRIOR_BETA = 1.0

RISK_GROUPS = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]

df = pd.read_csv(DATA_PATH)

summary = (
    df.groupby("risk_group")["default"]
    .agg(["size", "sum"])
    .reindex(RISK_GROUPS)
)

if summary.isna().any().any():
    raise ValueError("Risk groups are missing")

def log_target(z, alpha, beta):
    return (
        -alpha * np.logaddexp(0.0, -z)
        -beta * np.logaddexp(0.0, z)
    )

def metropolis_sampler(alpha, beta, draws, tune, seed):
    rng = np.random.default_rng(seed)

    posterior_mean = alpha / (alpha + beta)
    z = np.log(posterior_mean / (1 - posterior_mean))
    z += rng.normal(0, 0.02)

    n = alpha + beta
    step_size = 2.3 / np.sqrt(
        n * posterior_mean * (1 - posterior_mean)
    )

    current_logp = log_target(z, alpha, beta)

    samples = np.empty(draws)
    accepted = 0
    total = tune + draws

    for iteration in range(total):
        proposal = z + rng.normal(0, step_size)
        proposal_logp = log_target(
            proposal, alpha, beta
        )

        log_acceptance = proposal_logp - current_logp

        if np.log(rng.random()) < min(0.0, log_acceptance):
            z = proposal
            current_logp = proposal_logp

            if iteration >= tune:
                accepted += 1

        if iteration >= tune:
            samples[iteration - tune] = expit(z)

    return samples, accepted / draws

all_samples = np.empty(
    (N_CHAINS, N_DRAWS, len(RISK_GROUPS))
)

acceptance_records = []

for group_index, group in enumerate(RISK_GROUPS):
    n = int(summary.loc[group, "size"])
    y = int(summary.loc[group, "sum"])

    alpha = PRIOR_ALPHA + y
    beta = PRIOR_BETA + n - y

    for chain in range(N_CHAINS):
        seed = SEED + group_index * 100 + chain

        samples, acceptance_rate = metropolis_sampler(
            alpha=alpha,
            beta=beta,
            draws=N_DRAWS,
            tune=N_TUNE,
            seed=seed
        )

        all_samples[chain, :, group_index] = samples

        acceptance_records.append({
            "risk_group": group,
            "chain": chain + 1,
            "acceptance_rate": acceptance_rate
        })

idata = az.from_dict(
    posterior={"pd": all_samples},
    coords={"risk_group": RISK_GROUPS},
    dims={"pd": ["risk_group"]}
)

idata.to_netcdf(
    SAMPLES_PATH / "mcmc_credit_risk.nc"
)

diagnostics = az.summary(
    idata,
    var_names=["pd"],
    hdi_prob=0.95,
    kind="all",
    round_to=6
)

diagnostics.to_csv(
    TABLES_PATH / "mcmc_diagnostics.csv"
)

acceptance_df = pd.DataFrame(acceptance_records)

acceptance_df.to_csv(
    TABLES_PATH / "mcmc_acceptance_rates.csv",
    index=False
)

analytical = pd.read_csv(
    TABLES_PATH / "bayesian_posterior_summary.csv"
).set_index("risk_group")

comparison_records = []

for group_index, group in enumerate(RISK_GROUPS):
    samples = all_samples[:, :, group_index].reshape(-1)

    analytical_mean = float(
        analytical.loc[group, "posterior_mean"]
    )

    mcmc_mean = float(np.mean(samples))

    comparison_records.append({
        "risk_group": group,
        "analytical_mean": analytical_mean,
        "mcmc_mean": mcmc_mean,
        "absolute_difference": abs(
            analytical_mean - mcmc_mean
        ),
        "mcmc_ci_lower": float(
            np.quantile(samples, 0.025)
        ),
        "mcmc_ci_upper": float(
            np.quantile(samples, 0.975)
        )
    })

comparison_df = pd.DataFrame(comparison_records)

comparison_df.to_csv(
    TABLES_PATH / "mcmc_vs_analytical.csv",
    index=False
)

axes = az.plot_trace(
    idata,
    var_names=["pd"],
    figsize=(13, 10),
    compact=False
)

fig = np.asarray(axes).ravel()[0].figure
fig.suptitle(
    "Metropolis-Hastings MCMC Trace and Posterior",
    fontsize=17,
    fontweight="bold",
    y=1.02
)

fig.tight_layout()

fig.savefig(
    FIGURES_PATH / "08_mcmc_trace_plots.png",
    dpi=220,
    bbox_inches="tight"
)

plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))

colors = ["#2563eb", "#f59e0b", "#dc2626"]

for group_index, group in enumerate(RISK_GROUPS):
    samples = all_samples[:, :, group_index].reshape(-1)

    ax.hist(
        samples * 100,
        bins=75,
        density=True,
        alpha=0.35,
        label=group,
        color=colors[group_index]
    )

ax.set_title(
    "MCMC Posterior Distributions",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel("Default Probability (%)")
ax.set_ylabel("Density")
ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURES_PATH / "09_mcmc_posterior_distributions.png",
    dpi=220
)

plt.close(fig)

rhat_values = az.rhat(
    idata,
    var_names=["pd"]
)["pd"].values

ess_values = az.ess(
    idata,
    var_names=["pd"],
    method="bulk"
)["pd"].values

max_rhat = float(np.max(rhat_values))
min_ess = float(np.min(ess_values))

report = {
    "algorithm": "Random-Walk Metropolis-Hastings",
    "parameterization": "Logit-transformed probability",
    "chains": N_CHAINS,
    "tuning_iterations_per_chain": N_TUNE,
    "posterior_draws_per_chain": N_DRAWS,
    "total_posterior_draws_per_group": N_CHAINS * N_DRAWS,
    "maximum_rhat": max_rhat,
    "minimum_bulk_ess": min_ess,
    "mean_acceptance_rate": float(
        acceptance_df["acceptance_rate"].mean()
    ),
    "convergence_thresholds": {
        "rhat_max": 1.01,
        "bulk_ess_min": 400
    },
    "diagnostics_passed": bool(
        max_rhat < 1.01 and min_ess > 400
    )
}

with open(
    TABLES_PATH / "mcmc_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4)

print("\nMCMC SAMPLING COMPLETED")
print("=" * 75)

print("\nMCMC DIAGNOSTICS")
print(diagnostics.to_string())

print("\nACCEPTANCE RATES")
print(acceptance_df.to_string(index=False))

print("\nMCMC VS ANALYTICAL POSTERIOR")
print(comparison_df.to_string(index=False))

print("\nCONVERGENCE SUMMARY")
print(f"Maximum R-hat: {max_rhat:.6f}")
print(f"Minimum Bulk ESS: {min_ess:.2f}")
print(
    f"Mean acceptance rate: "
    f"{report['mean_acceptance_rate']:.2%}"
)

if report["diagnostics_passed"]:
    print("\nConvergence diagnostics: PASSED")
else:
    print("\nConvergence diagnostics: REVIEW REQUIRED")

print("\nMCMC results saved successfully.")
