import pandas as pd
from pathlib import Path


# ==========================================
# PATHS
# ==========================================

INPUT_FILE = Path(
    "ml/dataset/anomaly/reliability_weighted_scores.csv"
)

OUTPUT_DIR = Path("ml/dataset/risk")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "risk_assessment.csv"


# ==========================================
# CONFIGURATION
# ==========================================

# Movement intensity threshold based on the
# 99th percentile of normal-reference behavior.
#
# Normal reference:
# Avenue 01-16 + UMN
#
NORMAL_THRESHOLD = 1.50

# Movement intensity above this level is
# considered very high.
HIGH_RISK_THRESHOLD = 4.00

# Minimum number of consecutive abnormal
# windows required before declaring CRITICAL.
MIN_CRITICAL_PERSISTENCE = 2

# Minimum tracking reliability required
# for a CRITICAL decision.
CRITICAL_RELIABILITY_THRESHOLD = 0.85

# Below this reliability, the system cannot
# confidently assess risk.
UNRELIABLE_TRACKING_THRESHOLD = 0.80


# ==========================================
# MOVEMENT FEATURES
# ==========================================

MOVEMENT_FEATURES = [
    "avg_speed_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


# ==========================================
# LOAD DATA
# ==========================================

print("\n==========================================")
print("             RISK ENGINE")
print("==========================================")

df = pd.read_csv(INPUT_FILE)


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "source",
    "window_id",
    "tracking_reliability",
    *MOVEMENT_FEATURES,
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise RuntimeError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


# ==========================================
# NORMAL REFERENCE
#
# Avenue 01-16 + UMN
# ==========================================

reference_mask = (
    df["source"].str.match(
        r"^(0[1-9]|1[0-6])_features_behavioral$"
    )
    |
    (df["source"] == "umn_crowd_features_behavioral")
)

reference = df.loc[
    reference_mask
].copy()


if reference.empty:

    raise RuntimeError(
        "No normal reference windows found."
    )


print("\n==========================================")
print("          NORMAL REFERENCE")
print("==========================================")

print(
    f"Reference windows: {len(reference)}"
)


# ==========================================
# NORMAL REFERENCE MEDIANS
# ==========================================

reference_medians = (
    reference[MOVEMENT_FEATURES]
    .median()
)


print("\n===== NORMAL REFERENCE MEDIANS =====")

for feature in MOVEMENT_FEATURES:

    print(
        f"{feature:<30}"
        f"{reference_medians[feature]:.3f}"
    )


# ==========================================
# MOVEMENT INTENSITY INDEX
# ==========================================
#
# For every window:
#
# feature ratio =
#
# actual value / normal reference median
#
# The four ratios are averaged.
#
# Example:
#
# speed ratio        = actual speed / normal speed
# displacement ratio = actual displacement / normal displacement
# speed-change ratio = actual speed change / normal
# velocity-change ratio = actual velocity change / normal
#
# Movement Intensity Index =
# average of the four ratios
#
# ==========================================

movement_ratios = (
    df[MOVEMENT_FEATURES]
    .div(reference_medians)
)

movement_intensity = (
    movement_ratios.mean(axis=1)
)


# Add all derived columns at once.
df = pd.concat(
    [
        df,
        pd.DataFrame(
            {
                "movement_intensity_index":
                    movement_intensity
            },
            index=df.index,
        ),
    ],
    axis=1,
)


# ==========================================
# DATA-DERIVED NORMAL THRESHOLD
# ==========================================
#
# Calculate the 99th percentile of the
# movement intensity distribution inside
# the normal reference.
#
# This is used as a validation/reference
# value.
#
# Operational threshold = 1.50
#
# ==========================================

reference_ratios = (
    reference[MOVEMENT_FEATURES]
    .div(reference_medians)
)

reference_intensity = (
    reference_ratios.mean(axis=1)
)

normal_99th_percentile = (
    reference_intensity
    .quantile(0.99)
)


print("\n==========================================")
print("       MOVEMENT INTENSITY THRESHOLDS")
print("==========================================")

print(
    f"Normal reference 99th percentile: "
    f"{normal_99th_percentile:.3f}"
)

print(
    f"NORMAL:       < {NORMAL_THRESHOLD:.2f}"
)

print(
    f"HIGH_RISK:    "
    f"{NORMAL_THRESHOLD:.2f} - "
    f"< {HIGH_RISK_THRESHOLD:.2f}"
)

print(
    f"CRITICAL:     "
    f">= {HIGH_RISK_THRESHOLD:.2f}"
)


# ==========================================
# SORT WINDOWS
# ==========================================

df["window_id_numeric"] = pd.to_numeric(
    df["window_id"],
    errors="coerce"
)

df = (
    df.sort_values(
        [
            "source",
            "window_id_numeric"
        ]
    )
    .reset_index(drop=True)
)


# ==========================================
# MOVEMENT BAND
# ==========================================
#
# This is an intermediate movement-severity
# classification.
#
# It is NOT the final risk state.
#
# ==========================================

movement_band = pd.Series(
    "NORMAL",
    index=df.index,
)

movement_band.loc[
    df["movement_intensity_index"]
    >= NORMAL_THRESHOLD
] = "HIGH_RISK"

movement_band.loc[
    df["movement_intensity_index"]
    >= HIGH_RISK_THRESHOLD
] = "CRITICAL"


df["movement_band"] = movement_band


# ==========================================
# ABNORMAL MOVEMENT
# ==========================================

df["abnormal_movement"] = (
    df["movement_intensity_index"]
    >= NORMAL_THRESHOLD
)


# ==========================================
# MOVEMENT PERSISTENCE
# ==========================================
#
# Windows are processed independently
# for every video/source.
#
# Consecutive abnormal windows:
#
# NORMAL
# NORMAL
# HIGH
# HIGH
# HIGH
#
# becomes:
#
# 0
# 0
# 1
# 2
# 3
#
# ==========================================

df["movement_run_group"] = (
    df.groupby("source")[
        "abnormal_movement"
    ]
    .transform(
        lambda x:
        x.ne(x.shift()).cumsum()
    )
)

df["consecutive_abnormal_windows"] = (
    df.groupby(
        [
            "source",
            "movement_run_group",
        ]
    )
    .cumcount()
    + 1
)

df.loc[
    ~df["abnormal_movement"],
    "consecutive_abnormal_windows"
] = 0


# ==========================================
# TRACKING RELIABILITY
# ==========================================

df["tracking_quality"] = "RELIABLE"

df.loc[
    df["tracking_reliability"]
    < CRITICAL_RELIABILITY_THRESHOLD,
    "tracking_quality"
] = "REDUCED_CONFIDENCE"

df.loc[
    df["tracking_reliability"]
    < UNRELIABLE_TRACKING_THRESHOLD,
    "tracking_quality"
] = "UNRELIABLE"


# ==========================================
# VERY HIGH MOVEMENT
# ==========================================
#
# Extremely high movement alone is not enough
# to declare CRITICAL.
#
# It must persist.
#
# ==========================================

very_high_movement = (
    (df["movement_intensity_index"]
     >= HIGH_RISK_THRESHOLD)
    &
    (
        df["consecutive_abnormal_windows"]
        >= MIN_CRITICAL_PERSISTENCE
    )
)


# ==========================================
# FINAL RISK STATE
# ==========================================

df["risk_state"] = "NORMAL"

df["risk_reason"] = (
    "Movement consistent with normal reference"
)


# ==========================================
# HIGH RISK
#
# Movement is outside normal behavior but
# does not meet the CRITICAL conditions.
# ==========================================

high_risk_condition = (
    (df["movement_intensity_index"]
     >= NORMAL_THRESHOLD)
    &
    (df["movement_intensity_index"]
     < HIGH_RISK_THRESHOLD)
)


df.loc[
    high_risk_condition,
    "risk_state"
] = "HIGH_RISK"

df.loc[
    high_risk_condition,
    "risk_reason"
] = (
    "Movement above normal reference"
)


# ==========================================
# CRITICAL
#
# Conditions:
#
# 1. Very high movement
# 2. Persistent for >= 2 windows
# 3. Reliable tracking
#
# ==========================================

critical_condition = (
    very_high_movement
    &
    (
        df["tracking_reliability"]
        >= CRITICAL_RELIABILITY_THRESHOLD
    )
)


df.loc[
    critical_condition,
    "risk_state"
] = "CRITICAL"

df.loc[
    critical_condition,
    "risk_reason"
] = (
    "Persistent extreme movement "
    "with reliable tracking"
)


# ==========================================
# EXTREME MOVEMENT WITH REDUCED CONFIDENCE
#
# The movement is extreme and persistent,
# but tracking reliability is below the
# CRITICAL confidence threshold.
#
# Therefore:
#
# Do NOT call it CRITICAL.
#
# But also do NOT call it NORMAL.
#
# ==========================================

uncertain_critical_condition = (
    very_high_movement
    &
    (
        df["tracking_reliability"]
        < CRITICAL_RELIABILITY_THRESHOLD
    )
)


df.loc[
    uncertain_critical_condition,
    "risk_state"
] = "HIGH_RISK"

df.loc[
    uncertain_critical_condition,
    "risk_reason"
] = (
    "Extreme movement detected but "
    "tracking reliability is below "
    "CRITICAL threshold"
)


# ==========================================
# TRACKING UNRELIABLE
#
# If tracking quality is extremely poor,
# we should not make a confident risk
# assessment.
#
# ==========================================

unreliable_tracking = (
    df["tracking_reliability"]
    < UNRELIABLE_TRACKING_THRESHOLD
)


df.loc[
    unreliable_tracking,
    "risk_state"
] = "TRACKING_UNRELIABLE"

df.loc[
    unreliable_tracking,
    "risk_reason"
] = (
    "Tracking reliability is too low "
    "for confident risk assessment"
)


# ==========================================
# CLEAN TEMPORARY COLUMNS
# ==========================================

df = df.drop(
    columns=[
        "window_id_numeric",
        "movement_run_group",
    ]
)


# ==========================================
# FINAL COLUMN ORDER
# ==========================================

preferred_columns = [
    "source",
    "window_id",
    "window_start",
    "window_end",

    # Main movement features
    "avg_speed_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",

    # Main derived score
    "movement_intensity_index",

    # Movement interpretation
    "movement_band",
    "abnormal_movement",
    "consecutive_abnormal_windows",

    # Tracking quality
    "tracking_reliability",
    "tracking_quality",

    # Final decision
    "risk_state",
    "risk_reason",
]


existing_preferred_columns = [
    column
    for column in preferred_columns
    if column in df.columns
]

remaining_columns = [
    column
    for column in df.columns
    if column not in existing_preferred_columns
]

df = df[
    existing_preferred_columns
    + remaining_columns
]


# ==========================================
# SUMMARY
# ==========================================

print("\n==========================================")
print("          RISK STATE SUMMARY")
print("==========================================")

print(
    df["risk_state"]
    .value_counts()
    .to_string()
)


# ==========================================
# MOVEMENT BAND SUMMARY
# ==========================================

print("\n==========================================")
print("       MOVEMENT BAND SUMMARY")
print("==========================================")

print(
    df["movement_band"]
    .value_counts()
    .to_string()
)


# ==========================================
# DATASET RISK SUMMARY
# ==========================================

print("\n==========================================")
print("        DATASET RISK SUMMARY")
print("==========================================")

dataset_summary = (
    df.groupby(
        [
            "source",
            "risk_state",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    dataset_summary.to_string()
)


# ==========================================
# NON-NORMAL WINDOWS
# ==========================================

print("\n==========================================")
print("        NON-NORMAL WINDOWS")
print("==========================================")

display_columns = [
    "source",
    "window_id",
    "movement_intensity_index",
    "movement_band",
    "consecutive_abnormal_windows",
    "tracking_reliability",
    "tracking_quality",
    "risk_state",
    "risk_reason",
]


non_normal = df[
    df["risk_state"] != "NORMAL"
][display_columns]


if non_normal.empty:

    print(
        "No non-normal windows detected."
    )

else:

    print(
        non_normal
        .round(3)
        .to_string(index=False)
    )


# ==========================================
# SAVE
# ==========================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==========================================")
print("          RISK ENGINE COMPLETE")
print("==========================================")

print(
    f"Saved to: {OUTPUT_FILE}"
)

print("==========================================")