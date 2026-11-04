from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import beta, betabinom
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "credit_with_groups.csv"
TABLES_PATH = ROOT / "results" / "tables"
FIGURES_PATH = ROOT / "results" / "figures"

TABLES_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_PATH.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_SIZE = 0.2
CREDIBLE_LEVEL = 0.95

RISK_GROUPS = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]

PRIORS = [
    ("Uniform Beta(1,1)", 1.0, 1.0),
    ("Low-risk Beta(2,8)", 2.0, 8.0),
    ("Neutral Beta(5,5)", 5.0, 5.0)
]

df = pd.read_csv(DATA_PATH)

if not {"risk_group", "default"}.issubset(df.columns):
    raise ValueError("Required columns are missing")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=SEED,
    stratify=df["risk_group"]
)

train_summary = (
    train_df.groupby("risk_group")["default"]
    .agg(["size", "sum"])
    .reindex(RISK_GROUPS)
)

test_summary = (
    test_df.groupby("risk_group")["default"]
    .agg(["size", "sum"])
    .reindex(RISK_GROUPS)
)

if train_summary.isna().any().any() or test_summary.isna().any().any():
    raise ValueError("Missing risk groups in train or test split")

sensitivity_records = []

for prior_name, alpha_prior, beta_prior in PRIORS:
    for group in RISK_GROUPS:
        n = int(train_summary.loc[group, "size"])
        y = int(train_summary.loc[group, "sum"])

        alpha_post = alpha_prior + y
        beta_post = beta_prior + n - y

        posterior_mean = beta.mean(alpha_post, beta_post)
        lower, upper = beta.ppf(
            [0.025, 0.975],
            alpha_post,
            beta_post
        )

        sensitivity_records.append({
            "prior": prior_name,
            "risk_group": group,
            "train_customers": n,
            "train_defaults": y,
            "posterior_mean": posterior_mean,
            "ci_95_lower": lower,
            "ci_95_upper": upper
        })

sensitivity_df = pd.DataFrame(sensitivity_records)

sensitivity_df.to_csv(
    TABLES_PATH / "prior_sensitivity_summary.csv",
    index=False
)

validation_records = []

for group in RISK_GROUPS:
    n_train = int(train_summary.loc[group, "size"])
    y_train = int(train_summary.loc[group, "sum"])

    alpha_post = 1.0 + y_train
    beta_post = 1.0 + n_train - y_train

    n_test = int(test_summary.loc[group, "size"])
    y_test = int(test_summary.loc[group, "sum"])

    predictive = betabinom(n_test, alpha_post, beta_post)

    expected_defaults = float(predictive.mean())
    predictive_sd = float(predictive.std())

    lower, upper = predictive.ppf([0.025, 0.975])

    posterior_mean_pd = alpha_post / (alpha_post + beta_post)
    observed_test_rate = y_test / n_test

    covered = bool(lower <= y_test <= upper)

    group_test = test_df[test_df["risk_group"] == group].copy()
    group_test["predicted_pd"] = posterior_mean_pd

    brier = float(
        np.mean((group_test["default"] - group_test["predicted_pd"]) ** 2)
    )

    eps = 1e-12
    p = np.clip(posterior_mean_pd, eps, 1 - eps)

    log_loss = float(
        -np.mean(
            group_test["default"] * np.log(p) +
            (1 - group_test["default"]) * np.log(1 - p)
        )
    )

    validation_records.append({
        "risk_group": group,
        "train_customers": n_train,
        "train_defaults": y_train,
        "test_customers": n_test,
        "test_defaults": y_test,
        "posterior_mean_pd": posterior_mean_pd,
        "observed_test_rate": observed_test_rate,
        "expected_test_defaults": expected_defaults,
        "predictive_sd": predictive_sd,
        "predictive_lower_95": int(lower),
        "predictive_upper_95": int(upper),
        "coverage": covered,
        "brier_score": brier,
        "log_loss": log_loss
    })

validation_df = pd.DataFrame(validation_records)

validation_df.to_csv(
    TABLES_PATH / "heldout_validation_summary.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(12, 7))

colors = {
    "Uniform Beta(1,1)": "#2563eb",
    "Low-risk Beta(2,8)": "#f59e0b",
    "Neutral Beta(5,5)": "#dc2626"
}

x = np.arange(len(RISK_GROUPS))
width = 0.24

for i, (prior_name, _, _) in enumerate(PRIORS):
    subset = (
        sensitivity_df[sensitivity_df["prior"] == prior_name]
        .set_index("risk_group")
        .loc[RISK_GROUPS]
    )

    ax.bar(
        x + (i - 1) * width,
        subset["posterior_mean"].to_numpy() * 100,
        width=width,
        label=prior_name,
        color=colors[prior_name],
        alpha=0.85
    )

ax.set_xticks(x)
ax.set_xticklabels(RISK_GROUPS)
ax.set_ylabel("Posterior Default Probability (%)")
ax.set_title(
    "Prior Sensitivity Analysis of Posterior Means",
    fontsize=16,
    fontweight="bold",
    pad=18
)
ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES_PATH / "12_prior_sensitivity_analysis.png",
    dpi=220
)
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 6))

positions = np.arange(len(RISK_GROUPS))

expected = validation_df["expected_test_defaults"].to_numpy()
observed = validation_df["test_defaults"].to_numpy()
lower = validation_df["predictive_lower_95"].to_numpy()
upper = validation_df["predictive_upper_95"].to_numpy()

ax.barh(
    positions,
    expected,
    color="#2563eb",
    alpha=0.75,
    height=0.5,
    label="Expected test defaults"
)

ax.errorbar(
    expected,
    positions,
    xerr=[expected - lower, upper - expected],
    fmt="none",
    ecolor="#111827",
    capsize=7,
    linewidth=2,
    label="95% predictive interval"
)

ax.scatter(
    observed,
    positions,
    color="#dc2626",
    s=85,
    zorder=3,
    label="Observed test defaults"
)

ax.set_yticks(positions)
ax.set_yticklabels(RISK_GROUPS)
ax.invert_yaxis()
ax.set_xlabel("Number of Defaults in Test Set")
ax.set_title(
    "Held-out Predictive Validation by Risk Group",
    fontsize=16,
    fontweight="bold",
    pad=18
)
ax.legend()

fig.tight_layout()
fig.savefig(
    FIGURES_PATH / "13_heldout_validation.png",
    dpi=220
)
plt.close(fig)

report = {
    "assessment_methods": [
        "Prior sensitivity analysis",
        "Held-out predictive validation"
    ],
    "test_size": TEST_SIZE,
    "split_random_seed": SEED,
    "sensitivity_priors": [
        {"name": name, "alpha": a, "beta": b}
        for name, a, b in PRIORS
    ],
    "heldout_coverage_all_groups": bool(validation_df["coverage"].all()),
    "mean_brier_score": float(validation_df["brier_score"].mean()),
    "mean_log_loss": float(validation_df["log_loss"].mean()),
    "prior_sensitivity_results": sensitivity_df.to_dict(orient="records"),
    "heldout_validation_results": validation_df.to_dict(orient="records")
}

with open(
    TABLES_PATH / "model_assessment_report.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(report, file, indent=4, ensure_ascii=False)

print("\nMODEL ASSESSMENT COMPLETED")
print("=" * 80)

print("\nPRIOR SENSITIVITY SUMMARY")
print(
    sensitivity_df[
        ["prior", "risk_group", "posterior_mean", "ci_95_lower", "ci_95_upper"]
    ].to_string(index=False)
)

print("\nHELD-OUT VALIDATION SUMMARY")
print(
    validation_df[
        [
            "risk_group",
            "test_customers",
            "test_defaults",
            "posterior_mean_pd",
            "observed_test_rate",
            "expected_test_defaults",
            "predictive_lower_95",
            "predictive_upper_95",
            "coverage",
            "brier_score",
            "log_loss"
        ]
    ].to_string(index=False)
)

print("\nASSESSMENT OVERVIEW")
print(f"All groups covered by 95% predictive intervals: {validation_df['coverage'].all()}")
print(f"Mean Brier score: {validation_df['brier_score'].mean():.6f}")
print(f"Mean log loss: {validation_df['log_loss'].mean():.6f}")

print("\nGenerated figures:")
print("12_prior_sensitivity_analysis.png")
print("13_heldout_validation.png")

print("\nModel assessment completed successfully.")