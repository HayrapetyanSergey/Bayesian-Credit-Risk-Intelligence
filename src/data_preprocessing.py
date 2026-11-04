
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "credit_default.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "credit_cleaned.csv"
REPORT_PATH = ROOT / "results" / "tables" / "data_quality_report.json"

PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_PATH, skiprows=1)

df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df = df.rename(columns={
    "default_payment_next_month": "default",
    "pay_0": "pay_1"
})

if df.shape[1] != 25:
    raise ValueError(f"Expected 25 columns, found {df.shape[1]}")

if "default" not in df.columns:
    raise ValueError("Target variable 'default' was not found")

numeric_columns = df.columns.difference(["id"])

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

original_missing = int(df.isna().sum().sum())
duplicate_ids = int(df["id"].duplicated().sum())

if original_missing > 0:
    raise ValueError("Missing or nonnumeric values found")

if duplicate_ids > 0:
    raise ValueError("Duplicate IDs found")

if not df["default"].isin([0, 1]).all():
    raise ValueError("Target must contain only 0 and 1")

original_education = df["education"].value_counts().sort_index().to_dict()
original_marriage = df["marriage"].value_counts().sort_index().to_dict()

df["education"] = df["education"].replace({
    0: 4,
    5: 4,
    6: 4
})

df["marriage"] = df["marriage"].replace({
    0: 3
})

df["education"] = df["education"].astype(int)
df["marriage"] = df["marriage"].astype(int)
df["default"] = df["default"].astype(int)
df["id"] = df["id"].astype(int)

df = df.sort_values("id").reset_index(drop=True)

default_counts = df["default"].value_counts().sort_index()

report = {
    "dataset": "Default of Credit Card Clients",
    "observations": int(len(df)),
    "columns": int(df.shape[1]),
    "original_missing_values": original_missing,
    "duplicate_ids": duplicate_ids,
    "target_distribution": {
        str(k): int(v) for k, v in default_counts.items()
    },
    "default_rate": float(df["default"].mean()),
    "original_education_codes": {
        str(k): int(v) for k, v in original_education.items()
    },
    "original_marriage_codes": {
        str(k): int(v) for k, v in original_marriage.items()
    },
    "processed_education_codes": {
        str(k): int(v)
        for k, v in df["education"].value_counts().sort_index().items()
    },
    "processed_marriage_codes": {
        str(k): int(v)
        for k, v in df["marriage"].value_counts().sort_index().items()
    },
    "preprocessing": [
        "Removed the extra CSV header row",
        "Standardized column names",
        "Renamed PAY_0 to pay_1",
        "Combined undocumented education codes into Other",
        "Combined marriage code 0 into Other",
        "Preserved all observations"
    ]
}

df.to_csv(PROCESSED_PATH, index=False)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    json.dump(report, file, indent=4, ensure_ascii=False)

print("\nDATA PREPROCESSING COMPLETED")
print("=" * 55)
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isna().sum().sum()}")
print(f"Duplicate IDs: {df['id'].duplicated().sum()}")
print(f"Default rate: {df['default'].mean():.4%}")

print("\nTarget distribution:")
print(df["default"].value_counts().sort_index())

print("\nEducation distribution:")
print(df["education"].value_counts().sort_index())

print("\nMarriage distribution:")
print(df["marriage"].value_counts().sort_index())

print("\nData types:")
print(df.dtypes.value_counts())

print(f"\nProcessed dataset saved to: {PROCESSED_PATH}")
print(f"Quality report saved to: {REPORT_PATH}")
