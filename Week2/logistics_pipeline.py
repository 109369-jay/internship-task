import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# =============================================================================
# STEP 1: SIMULATE RAW LOGISTICS DATASET WITH COMMON DATA QUALITY ISSUES
# =============================================================================
np.random.seed(42)
n_rows = 1000

# Base raw data generation
data = {
    "shipment_id": [f"TRK-{1000 + i}" for i in range(n_rows)],
    "origin_hub": np.random.choice(
        [" Hub_A ", "Hub_B", "hub_a", "Hub_C", None],
        size=n_rows,
        p=[0.25, 0.35, 0.15, 0.20, 0.05],
    ),
    "transport_mode": np.random.choice(
        ["Road", "Air", "Rail", "Sea", np.nan],
        size=n_rows,
        p=[0.50, 0.20, 0.15, 0.10, 0.05],
    ),
    "weight_kg": np.random.normal(loc=150, scale=40, size=n_rows),
    "shipping_cost_usd": np.random.exponential(scale=300, size=n_rows),
    "dispatch_time": pd.date_range(
        start="2026-01-01", periods=n_rows, freq="30min"
    ).astype(str),
    "delivery_status": np.random.choice(
        ["Delivered", "Delayed", "Cancelled", "In-Transit"],
        size=n_rows,
        p=[0.70, 0.15, 0.05, 0.10],
    ),
}

df_raw = pd.DataFrame(data)

# Inject intentional real-world anomalies:
# 1. Duplicate records (sync retries)
df_raw = pd.concat([df_raw, df_raw.iloc[0:15]], ignore_index=True)

# 2. Corrupt/negative sensor values and extreme sensor spikes
df_raw.loc[10:15, "weight_kg"] = -50.0  # Physically impossible values
df_raw.loc[30:33, "weight_kg"] = 5500.0  # Scale error/extreme outlier
df_raw.loc[45:50, "shipping_cost_usd"] = np.nan  # Unrecorded billing

print("=" * 60)
print(f"Raw Dataset Shape (with duplicates & flaws): {df_raw.shape}")
print("=" * 60)


# =============================================================================
# STEP 2: DATA CLEANING & HYGIENE
# =============================================================================
df_clean = df_raw.copy()

# A. Deduplication based on primary key
df_clean.drop_duplicates(subset=["shipment_id"], keep="first", inplace=True)

# B. Text standardization (trim trailing/leading whitespace and standardize case)
df_clean["origin_hub"] = df_clean["origin_hub"].str.strip().str.upper()

# C. Temporal parsing
df_clean["dispatch_time"] = pd.to_datetime(
    df_clean["dispatch_time"], errors="coerce"
)

# D. Correcting domain-specific physical impossibilities
# Negative or zero weights are converted to NaN for principled statistical imputation
df_clean.loc[df_clean["weight_kg"] <= 0, "weight_kg"] = np.nan


# =============================================================================
# STEP 3: MISSING VALUE IMPUTATION
# =============================================================================
# A. Categorical Imputation: Replace NaNs with the statistical mode
mode_hub = df_clean["origin_hub"].mode()[0]
df_clean["origin_hub"] = df_clean["origin_hub"].fillna(mode_hub)

mode_transport = df_clean["transport_mode"].mode()[0]
df_clean["transport_mode"] = df_clean["transport_mode"].fillna(mode_transport)

# B. Grouped Numerical Imputation:
# Impute weight using median weight per transport mode (air freight != road/sea freight)
df_clean["weight_kg"] = df_clean.groupby("transport_mode")[
    "weight_kg"
].transform(lambda grp: grp.fillna(grp.median()))

# C. Skewed Metric Imputation: Median replacement for right-skewed shipping costs
cost_median = df_clean["shipping_cost_usd"].median()
df_clean["shipping_cost_usd"] = df_clean["shipping_cost_usd"].fillna(
    cost_median
)


# =============================================================================
# STEP 4: OUTLIER DETECTION AND TREATMENT (IQR WINSORIZATION)
# =============================================================================
# Calculate IQR thresholds for weight_kg
Q1 = df_clean["weight_kg"].quantile(0.25)
Q3 = df_clean["weight_kg"].quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

# Cap extreme values to theoretical thresholds rather than dropping valid rows
df_clean["weight_kg_capped"] = np.clip(
    df_clean["weight_kg"], lower_fence, upper_fence
)


# =============================================================================
# STEP 5: FEATURE ENGINEERING & NORMALIZATION
# =============================================================================
# Feature Engineering: Cost per unit weight
df_clean["cost_per_kg"] = (
    df_clean["shipping_cost_usd"] / df_clean["weight_kg_capped"]
)

# A. Min-Max Normalization (0 to 1 scaling, suitable for clustering / neural nets)
minmax_scaler = MinMaxScaler()
df_clean["weight_norm_minmax"] = minmax_scaler.fit_transform(
    df_clean[["weight_kg_capped"]]
)

# B. Robust Scaling (median/IQR scaling, robust to heavy-tailed cost outliers)
robust_scaler = RobustScaler()
df_clean["cost_robust_scaled"] = robust_scaler.fit_transform(
    df_clean[["shipping_cost_usd"]]
)


# =============================================================================
# FINAL VERIFICATION & OUTPUT
# =============================================================================
print("Cleaned and Processed Dataset Shape:", df_clean.shape)
print("\nMissing Values Check across Columns:")
print(
    df_clean[
        [
            "shipment_id",
            "origin_hub",
            "transport_mode",
            "weight_kg_capped",
            "shipping_cost_usd",
        ]
    ].isnull().sum()
)

print("\nFinal Clean Data Sample:")
display_cols = [
    "shipment_id",
    "origin_hub",
    "transport_mode",
    "weight_kg_capped",
    "weight_norm_minmax",
    "shipping_cost_usd",
    "cost_robust_scaled",
    "cost_per_kg",
]
print(df_clean[display_cols].head(10))