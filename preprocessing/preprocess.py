import json
import os

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RAW_FILE = (
    "data/raw/2024/january/"
    "On_Time_Reporting_Carrier_On_Time_Performance_"
    "(1987_present)_2024_1.csv"
)

OUTPUT_DIR = "data/processed"


USE_COLUMNS = [
    "FlightDate",
    "Month",
    "DayOfWeek",
    "CRSDepTime",
    "DepDelayMinutes",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
    "Cancelled",
    "Diverted",
]


RECORD_ID_COLUMN = "RecordId"

FEATURE_COLUMNS = [
    "DepDelayMinutes",
    "CRSElapsedTime",
    "Distance",
    "DepTimeSin",
    "DepTimeCos",
    "DayOfWeekSin",
    "DayOfWeekCos",
]

TARGET_COLUMN = "ArrDel15"

CONTINUOUS_FEATURES = [
    "DepDelayMinutes",
    "CRSElapsedTime",
    "Distance",
]


# ============================================================
# Helper: convert HHMM departure time to minutes
# ============================================================

def hhmm_to_minutes(value):
    """
    Convert BTS CRSDepTime values such as:

        600  -> 360 minutes
        1330 -> 810 minutes
        2355 -> 1435 minutes
        2400 -> 0 minutes
    """

    if pd.isna(value):
        return np.nan

    value = int(value)

    if value == 2400:
        return 0

    hours = value // 100
    minutes = value % 100

    return hours * 60 + minutes


# ============================================================
# Helper: print target distribution
# ============================================================

def print_target_distribution(name, data):
    print(f"\n{name} target distribution:")

    counts = (
        data[TARGET_COLUMN]
        .astype(int)
        .value_counts()
        .sort_index()
    )

    percentages = (
        data[TARGET_COLUMN]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for label in counts.index:
        print(
            f"  {label}: "
            f"{counts[label]:,} "
            f"({percentages[label]:.2f}%)"
        )


# ============================================================
# Load raw data
# ============================================================

print("=" * 60)
print("FLIGHT DELAY PREPROCESSING")
print("=" * 60)

print("\nReading raw BTS data...")

df = pd.read_csv(
    RAW_FILE,
    usecols=USE_COLUMNS,
    low_memory=False,
)

print(f"Original rows:    {len(df):,}")
print(f"Original columns: {len(df.columns)}")


# ============================================================
# Convert FlightDate
# ============================================================

df["FlightDate"] = pd.to_datetime(
    df["FlightDate"],
    errors="coerce",
)


# ============================================================
# Remove cancelled and diverted flights
# ============================================================

print("\nRemoving cancelled and diverted flights...")

df = df[
    (df["Cancelled"] == 0) &
    (df["Diverted"] == 0)
].copy()

print(f"Rows after filter: {len(df):,}")


# ============================================================
# Missing-value inspection
# ============================================================

print("\nMissing values after cancelled/diverted filtering:")

columns_to_check = [
    "FlightDate",
    "Month",
    "DayOfWeek",
    "CRSDepTime",
    "DepDelayMinutes",
    "CRSElapsedTime",
    "Distance",
    "ArrDel15",
]

print(df[columns_to_check].isna().sum())


df = df.dropna(
    subset=columns_to_check
).copy()

print(
    f"\nRows after removing remaining missing values: "
    f"{len(df):,}"
)


# ============================================================
# Add stable record ID
# ============================================================

# RecordId is NOT a model feature.
# It exists only to make sampling and KNN tie-breaking deterministic.
df = df.reset_index(drop=True)
df[RECORD_ID_COLUMN] = np.arange(
    len(df),
    dtype=np.int64,
)

print(
    f"\nAssigned stable RecordId values: "
    f"0 to {len(df) - 1:,}"
)


# ============================================================
# Month inspection
# ============================================================

print("\nMonth values:")
print(
    sorted(
        df["Month"]
        .dropna()
        .unique()
        .tolist()
    )
)

# January only, so Month is constant and is not a model feature.


# ============================================================
# Convert departure time
# ============================================================

print("\nConverting CRSDepTime to minutes...")

df["DepTimeMinutes"] = (
    df["CRSDepTime"]
    .apply(hhmm_to_minutes)
)


# ============================================================
# Cyclical departure-time encoding
# ============================================================

df["DepTimeSin"] = np.sin(
    2 * np.pi *
    df["DepTimeMinutes"] / 1440
)

df["DepTimeCos"] = np.cos(
    2 * np.pi *
    df["DepTimeMinutes"] / 1440
)


# ============================================================
# Cyclical DayOfWeek encoding
# ============================================================

# BTS: Monday = 1, ..., Sunday = 7

df["DayOfWeekSin"] = np.sin(
    2 * np.pi *
    (df["DayOfWeek"] - 1) / 7
)

df["DayOfWeekCos"] = np.cos(
    2 * np.pi *
    (df["DayOfWeek"] - 1) / 7
)


# ============================================================
# Temporal Train / Validation / Test split
# ============================================================

print("\n" + "=" * 60)
print("TEMPORAL SPLIT")
print("=" * 60)

train_start = pd.Timestamp("2024-01-01")
validation_start = pd.Timestamp("2024-01-20")
test_start = pd.Timestamp("2024-01-25")
test_end = pd.Timestamp("2024-01-31")


train_df = df[
    (df["FlightDate"] >= train_start) &
    (df["FlightDate"] < validation_start)
].copy()

validation_df = df[
    (df["FlightDate"] >= validation_start) &
    (df["FlightDate"] < test_start)
].copy()

test_df = df[
    (df["FlightDate"] >= test_start) &
    (df["FlightDate"] <= test_end)
].copy()


print(
    f"Train:      "
    f"{train_df['FlightDate'].min().date()} "
    f"to "
    f"{train_df['FlightDate'].max().date()}"
)

print(
    f"Validation: "
    f"{validation_df['FlightDate'].min().date()} "
    f"to "
    f"{validation_df['FlightDate'].max().date()}"
)

print(
    f"Test:       "
    f"{test_df['FlightDate'].min().date()} "
    f"to "
    f"{test_df['FlightDate'].max().date()}"
)


print("\nRow counts:")
print(f"Train:      {len(train_df):,}")
print(f"Validation: {len(validation_df):,}")
print(f"Test:       {len(test_df):,}")
print(
    f"Total:      "
    f"{len(train_df) + len(validation_df) + len(test_df):,}"
)


# ============================================================
# Target distributions before standardization
# ============================================================

print_target_distribution(
    "TRAIN",
    train_df,
)

print_target_distribution(
    "VALIDATION",
    validation_df,
)

print_target_distribution(
    "TEST",
    test_df,
)


# ============================================================
# Compute scaler using TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 60)
print("STANDARDIZATION")
print("=" * 60)

scaler_mean = (
    train_df[CONTINUOUS_FEATURES]
    .mean()
)

scaler_std = (
    train_df[CONTINUOUS_FEATURES]
    .std(ddof=0)
)

print("\nTraining means:")
print(scaler_mean)

print("\nTraining standard deviations:")
print(scaler_std)


if (scaler_std == 0).any():
    raise ValueError(
        "One or more continuous features "
        "have zero standard deviation."
    )


# ============================================================
# Standardize using TRAIN statistics only
# ============================================================

train_df[CONTINUOUS_FEATURES] = (
    train_df[CONTINUOUS_FEATURES] -
    scaler_mean
) / scaler_std

validation_df[CONTINUOUS_FEATURES] = (
    validation_df[CONTINUOUS_FEATURES] -
    scaler_mean
) / scaler_std

test_df[CONTINUOUS_FEATURES] = (
    test_df[CONTINUOUS_FEATURES] -
    scaler_mean
) / scaler_std


# ============================================================
# Select final model columns
# ============================================================

# RecordId is saved but is NOT included in FEATURE_COLUMNS.
FINAL_COLUMNS = (
    [RECORD_ID_COLUMN] +
    FEATURE_COLUMNS +
    [TARGET_COLUMN]
)

train_final = train_df[FINAL_COLUMNS].copy()
validation_final = validation_df[FINAL_COLUMNS].copy()
test_final = test_df[FINAL_COLUMNS].copy()


train_final[RECORD_ID_COLUMN] = (
    train_final[RECORD_ID_COLUMN]
    .astype("int64")
)

validation_final[RECORD_ID_COLUMN] = (
    validation_final[RECORD_ID_COLUMN]
    .astype("int64")
)

test_final[RECORD_ID_COLUMN] = (
    test_final[RECORD_ID_COLUMN]
    .astype("int64")
)


train_final[TARGET_COLUMN] = (
    train_final[TARGET_COLUMN]
    .astype("int64")
)

validation_final[TARGET_COLUMN] = (
    validation_final[TARGET_COLUMN]
    .astype("int64")
)

test_final[TARGET_COLUMN] = (
    test_final[TARGET_COLUMN]
    .astype("int64")
)


# ============================================================
# Final validation
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATA CHECK")
print("=" * 60)

print("\nFinal columns:")
for column in FINAL_COLUMNS:
    print(f"  {column}")

print("\nModel feature columns:")
for column in FEATURE_COLUMNS:
    print(f"  {column}")

print("\nFinal shapes:")
print(f"Train:      {train_final.shape}")
print(f"Validation: {validation_final.shape}")
print(f"Test:       {test_final.shape}")

print("\nMissing values:")
print(
    f"Train:      "
    f"{train_final.isna().sum().sum()}"
)
print(
    f"Validation: "
    f"{validation_final.isna().sum().sum()}"
)
print(
    f"Test:       "
    f"{test_final.isna().sum().sum()}"
)


if not train_final[RECORD_ID_COLUMN].is_unique:
    raise ValueError("Training RecordId values are not unique.")

if not validation_final[RECORD_ID_COLUMN].is_unique:
    raise ValueError("Validation RecordId values are not unique.")

if not test_final[RECORD_ID_COLUMN].is_unique:
    raise ValueError("Test RecordId values are not unique.")


# ============================================================
# Display samples
# ============================================================

print("\nTraining sample:")
print(
    train_final
    .head()
    .to_string(index=False)
)

print("\nValidation sample:")
print(
    validation_final
    .head()
    .to_string(index=False)
)

print("\nTest sample:")
print(
    test_final
    .head()
    .to_string(index=False)
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Save Parquet datasets
# ============================================================

train_path = f"{OUTPUT_DIR}/train.parquet"
validation_path = f"{OUTPUT_DIR}/validation.parquet"
test_path = f"{OUTPUT_DIR}/test.parquet"

print("\n" + "=" * 60)
print("SAVING DATA")
print("=" * 60)

train_final.to_parquet(
    train_path,
    index=False,
)

validation_final.to_parquet(
    validation_path,
    index=False,
)

test_final.to_parquet(
    test_path,
    index=False,
)

print(f"\nSaved: {train_path}")
print(f"Saved: {validation_path}")
print(f"Saved: {test_path}")


# ============================================================
# Save preprocessing metadata
# ============================================================

metadata = {
    "dataset": {
        "source": "BTS On-Time Performance",
        "year": 2024,
        "month": 1,
    },

    "record_id": {
        "column": RECORD_ID_COLUMN,
        "used_as_feature": False,
        "purpose": (
            "Deterministic sampling and deterministic "
            "tie-breaking for exact KNN comparisons"
        ),
    },

    "prediction_target": TARGET_COLUMN,

    "features": FEATURE_COLUMNS,

    "continuous_features": CONTINUOUS_FEATURES,

    "splits": {
        "train": {
            "start": "2024-01-01",
            "end": "2024-01-19",
            "rows": len(train_final),
        },
        "validation": {
            "start": "2024-01-20",
            "end": "2024-01-24",
            "rows": len(validation_final),
        },
        "test": {
            "start": "2024-01-25",
            "end": "2024-01-31",
            "rows": len(test_final),
        },
    },

    "standardization": {
        "fit_on": "training_data_only",
        "mean": {
            key: float(value)
            for key, value in scaler_mean.items()
        },
        "std": {
            key: float(value)
            for key, value in scaler_std.items()
        },
    },

    "cyclical_encoding": {
        "CRSDepTime": {
            "output": [
                "DepTimeSin",
                "DepTimeCos",
            ],
            "period": 1440,
        },
        "DayOfWeek": {
            "output": [
                "DayOfWeekSin",
                "DayOfWeekCos",
            ],
            "period": 7,
        },
    },

    "filters": {
        "cancelled_removed": True,
        "diverted_removed": True,
    },
}

metadata_path = (
    f"{OUTPUT_DIR}/preprocessing_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metadata,
        file,
        indent=4,
    )

print(f"Saved: {metadata_path}")


# ============================================================
# Finished
# ============================================================

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(
    f"\nTraining rows:   "
    f"{len(train_final):,}"
)

print(
    f"Validation rows: "
    f"{len(validation_final):,}"
)

print(
    f"Testing rows:    "
    f"{len(test_final):,}"
)
