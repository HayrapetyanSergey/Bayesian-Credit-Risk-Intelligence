
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "credit_cleaned.csv"
FIGURES_PATH = ROOT / "results" / "figures"
TABLES_PATH = ROOT / "results" / "tables"

FIGURES_PATH.mkdir(parents=True, exist_ok=True)
TABLES_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titlesize": 16,
    "axes.labelsize": 12,
    "font.family": "DejaVu Sans"
})

df["age_group"] = pd.cut(
    df["age"],
    bins=[0, 25, 35, 45, 55, np.inf],
    labels=["18-25", "26-35", "36-45", "46-55", "56+"]
)

df["credit_limit_group"] = pd.cut(
    df["limit_bal"],
    bins=[0, 50000, 100000, 200000, 500000, np.inf],
    labels=["<=50K", "50K-100K", "100K-200K", "200K-500K", ">500K"]
)

df["risk_group"] = np.select(
    [
        df["pay_1"] <= 0,
        df["pay_1"] == 1,
        df["pay_1"] >= 2
    ],
    [
        "No reported delay",
        "1-month delay",
        "2+ months delay"
    ],
    default="Unknown"
)

risk_order = [
    "No reported delay",
    "1-month delay",
    "2+ months delay"
]

def group_summary(column):
    summary = (
        df.groupby(column, observed=True, dropna=False)
        .agg(
            customers=("default", "size"),
            defaults=("default", "sum"),
            default_rate=("default", "mean")
        )
        .reset_index()
    )

    summary["non_defaults"] = (
        summary["customers"] - summary["defaults"]
    )

    summary["default_rate_pct"] = (
        summary["default_rate"] * 100
    ).round(2)

    summary.to_csv(
        TABLES_PATH / f"{column}_summary.csv",
        index=False
    )

    return summary

age_summary = group_summary("age_group")
credit_summary = group_summary("credit_limit_group")
risk_summary = group_summary("risk_group")
education_summary = group_summary("education")
marriage_summary = group_summary("marriage")

risk_summary["risk_group"] = pd.Categorical(
    risk_summary["risk_group"],
    categories=risk_order,
    ordered=True
)

risk_summary = risk_summary.sort_values("risk_group")

target_counts = df["default"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    ["Non-default", "Default"],
    target_counts.values,
    color=["#2563eb", "#ef4444"],
    width=0.55
)

for bar, count in zip(bars, target_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 200,
        f"{count:,}",
        ha="center",
        fontweight="bold"
    )

ax.set_title("Credit Default Distribution", fontweight="bold", pad=18)
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, target_counts.max() * 1.15)

sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_PATH / "01_default_distribution.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=df,
    x="limit_bal",
    hue="default",
    bins=45,
    stat="density",
    common_norm=False,
    element="step",
    fill=False,
    palette={0: "#2563eb", 1: "#ef4444"},
    ax=ax
)

ax.set_title("Credit Limit Distribution by Default Status", fontweight="bold")
ax.set_xlabel("Credit Limit")
ax.set_ylabel("Density")

sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_PATH / "02_credit_limit_distribution.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=age_summary,
    x="age_group",
    y="default_rate_pct",
    color="#2563eb",
    ax=ax
)

ax.set_title("Default Rate by Age Group", fontweight="bold")
ax.set_xlabel("Age Group")
ax.set_ylabel("Default Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=4)

ax.set_ylim(0, age_summary["default_rate_pct"].max() * 1.2)

sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_PATH / "03_default_rate_by_age.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=risk_summary,
    x="risk_group",
    y="default_rate_pct",
    color="#ef4444",
    ax=ax
)

ax.set_title("Default Rate by Repayment Delay Group", fontweight="bold")
ax.set_xlabel("Repayment Status")
ax.set_ylabel("Default Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=4)

ax.set_ylim(0, risk_summary["default_rate_pct"].max() * 1.2)

sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_PATH / "04_default_rate_by_risk_group.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=credit_summary,
    x="credit_limit_group",
    y="default_rate_pct",
    color="#0d9488",
    ax=ax
)

ax.set_title("Default Rate by Credit Limit Group", fontweight="bold")
ax.set_xlabel("Credit Limit Group")
ax.set_ylabel("Default Rate (%)")

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f%%", padding=4)

ax.set_ylim(0, credit_summary["default_rate_pct"].max() * 1.2)

sns.despine()
plt.tight_layout()
plt.savefig(FIGURES_PATH / "05_default_rate_by_credit_limit.png")
plt.close()

df.to_csv(
    ROOT / "data" / "processed" / "credit_with_groups.csv",
    index=False
)

eda_report = {
    "observations": int(len(df)),
    "defaults": int(df["default"].sum()),
    "non_defaults": int((df["default"] == 0).sum()),
    "observed_default_rate": float(df["default"].mean()),
    "average_age": float(df["age"].mean()),
    "median_age": float(df["age"].median()),
    "average_credit_limit": float(df["limit_bal"].mean()),
    "median_credit_limit": float(df["limit_bal"].median()),
    "min_credit_limit": int(df["limit_bal"].min()),
    "max_credit_limit": int(df["limit_bal"].max()),
    "risk_groups": risk_summary[
        ["risk_group", "customers", "defaults", "default_rate"]
    ].to_dict(orient="records")
}

with open(
    TABLES_PATH / "eda_summary.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(eda_report, file, indent=4, ensure_ascii=False)

print("\nEXPLORATORY DATA ANALYSIS COMPLETED")
print("=" * 65)

print(f"\nObservations: {len(df):,}")
print(f"Overall default rate: {df['default'].mean():.2%}")
print(f"Average age: {df['age'].mean():.2f}")
print(f"Average credit limit: {df['limit_bal'].mean():,.2f}")

print("\nDEFAULT RATE BY AGE GROUP")
print(age_summary.to_string(index=False))

print("\nDEFAULT RATE BY CREDIT LIMIT GROUP")
print(credit_summary.to_string(index=False))

print("\nDEFAULT RATE BY REPAYMENT DELAY GROUP")
print(risk_summary.to_string(index=False))

print("\nDEFAULT RATE BY EDUCATION")
print(education_summary.to_string(index=False))

print("\nDEFAULT RATE BY MARRIAGE")
print(marriage_summary.to_string(index=False))

print("\nGenerated figures:")
for figure in sorted(FIGURES_PATH.glob("*.png")):
    print(figure.name)

print("\nEDA completed successfully.")