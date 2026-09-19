import pandas as pd
import numpy as np
from pathlib import Path

from ml.features.tracking_cleaning import detect_tracking_anomalies
from ml.analysis.create_time_windows import create_time_windows
from ml.analysis.behavioral_features import calculate_behavioral_features


# ============================================================
# PATHS
# ============================================================

REFERENCE_FILE = Path(
    "ml/dataset/reference/normal_reference_stats.csv"
)


# ============================================================
# RISK CONFIGURATION
# ============================================================

NORMAL_THRESHOLD = 1.50
HIGH_RISK_THRESHOLD = 4.00

MIN_CRITICAL_PERSISTENCE = 2

CRITICAL_RELIABILITY_THRESHOLD = 0.85
UNRELIABLE_TRACKING_THRESHOLD = 0.80


# ============================================================
# MOVEMENT FEATURES
# ============================================================

MOVEMENT_FEATURES = [
    "avg_speed_mean",
    "avg_displacement_mean",
    "avg_speed_change_mean",
    "avg_velocity_change_mean",
]


# ============================================================
# LOAD NORMAL REFERENCE
# ============================================================

def load_normal_reference():

    if not REFERENCE_FILE.exists():

        raise RuntimeError(
            f"Normal reference file not found: "
            f"{REFERENCE_FILE}"
        )

    reference = pd.read_csv(
        REFERENCE_FILE
    )

    required_columns = [
        "feature",
        "median",
    ]

    missing = [
        column
        for column in required_columns
        if column not in reference.columns
    ]

    if missing:

        raise RuntimeError(
            "Normal reference is missing columns: "
            + ", ".join(missing)
        )

    medians = {}

    for feature in MOVEMENT_FEATURES:

        row = reference[
            reference["feature"] == feature
        ]

        if row.empty:

            raise RuntimeError(
                f"Reference median missing for: {feature}"
            )

        median = float(
            row.iloc[0]["median"]
        )

        if median <= 0:

            raise RuntimeError(
                f"Invalid reference median for "
                f"{feature}: {median}"
            )

        medians[feature] = median

    return medians


# ============================================================
# TRACKING RELIABILITY
# ============================================================

def calculate_tracking_reliability(
    tracking_df
):

    df = tracking_df.copy()

    numeric_columns = [
        "frame",
        "timestamp",
        "x",
        "y",
        "width",
        "height",
        "movement_x",
        "movement_y",
        "displacement",
        "velocity_x",
        "velocity_y",
        "speed",
        "density",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "person_id",
            "frame",
            "timestamp",
            "x",
            "y",
        ]
    ).copy()

    if df.empty:

        return {
            "tracking_reliability": 0.0,
            "tracking_anomaly_percentage": 100.0,
            "large_jump_percentage": 100.0,
            "tracking_anomaly_rows": 0,
            "large_jump_rows": 0,
        }

    df = detect_tracking_anomalies(
        df
    )

    total_rows = len(df)

    anomaly_rows = int(
        df["tracking_anomaly"].sum()
    )

    anomaly_percentage = (
        anomaly_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    large_jump_rows = int(
        (
            df["trajectory_displacement"]
            > 100
        ).sum()
    )

    large_jump_percentage = (
        large_jump_rows / total_rows * 100
        if total_rows > 0
        else 0
    )

    reliability = 1.0

    reliability -= min(
        anomaly_percentage / 20,
        0.50
    )

    reliability -= min(
        large_jump_percentage / 10,
        0.30
    )

    reliability = max(
        0.0,
        min(1.0, reliability)
    )

    return {
        "tracking_reliability": reliability,
        "tracking_anomaly_percentage": anomaly_percentage,
        "large_jump_percentage": large_jump_percentage,
        "tracking_anomaly_rows": anomaly_rows,
        "large_jump_rows": large_jump_rows,
    }


# ============================================================
# MOVEMENT INTENSITY
# ============================================================

def calculate_movement_intensity(
    behavioral_df,
    reference_medians
):

    data = behavioral_df.copy()

    for feature in MOVEMENT_FEATURES:

        if feature not in data.columns:

            raise RuntimeError(
                f"Missing movement feature: {feature}"
            )

    ratios = (
        data[MOVEMENT_FEATURES]
        .div(
            pd.Series(reference_medians)
        )
    )

    data["movement_intensity_index"] = (
        ratios.mean(axis=1)
    )

    return data


# ============================================================
# APPLY RISK LOGIC
# ============================================================

def apply_risk_logic(
    df,
    tracking_reliability
):

    data = df.copy()

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    if "window_id" in data.columns:

        data = data.sort_values(
            "window_id"
        ).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # Movement band
    # --------------------------------------------------------

    data["movement_band"] = "NORMAL"

    data.loc[
        data["movement_intensity_index"]
        >= NORMAL_THRESHOLD,
        "movement_band"
    ] = "HIGH_RISK"

    data.loc[
        data["movement_intensity_index"]
        >= HIGH_RISK_THRESHOLD,
        "movement_band"
    ] = "CRITICAL"

    # --------------------------------------------------------
    # Abnormal movement
    # --------------------------------------------------------

    data["abnormal_movement"] = (
        data["movement_intensity_index"]
        >= NORMAL_THRESHOLD
    )

    # --------------------------------------------------------
    # Consecutive abnormal windows
    # --------------------------------------------------------

    consecutive = []

    count = 0

    for abnormal in data[
        "abnormal_movement"
    ]:

        if abnormal:

            count += 1

        else:

            count = 0

        consecutive.append(count)

    data[
        "consecutive_abnormal_windows"
    ] = consecutive

    # --------------------------------------------------------
    # Tracking quality
    # --------------------------------------------------------

    if tracking_reliability >= CRITICAL_RELIABILITY_THRESHOLD:

        tracking_quality = "RELIABLE"

    elif tracking_reliability >= UNRELIABLE_TRACKING_THRESHOLD:

        tracking_quality = "REDUCED_CONFIDENCE"

    else:

        tracking_quality = "UNRELIABLE"

    data["tracking_reliability"] = (
        tracking_reliability
    )

    data["tracking_quality"] = (
        tracking_quality
    )

    # --------------------------------------------------------
    # Default risk
    # --------------------------------------------------------

    data["risk_state"] = "NORMAL"

    data["risk_reason"] = (
        "Movement consistent with normal reference"
    )

    # --------------------------------------------------------
    # HIGH RISK
    # --------------------------------------------------------

    high_risk_condition = (
        (data["movement_intensity_index"]
         >= NORMAL_THRESHOLD)
        &
        (data["movement_intensity_index"]
         < HIGH_RISK_THRESHOLD)
    )

    data.loc[
        high_risk_condition,
        "risk_state"
    ] = "HIGH_RISK"

    data.loc[
        high_risk_condition,
        "risk_reason"
    ] = (
        "Movement above normal reference"
    )

    # --------------------------------------------------------
    # VERY HIGH MOVEMENT
    # --------------------------------------------------------

    very_high_movement = (
        (data["movement_intensity_index"]
         >= HIGH_RISK_THRESHOLD)
        &
        (
            data[
                "consecutive_abnormal_windows"
            ]
            >= MIN_CRITICAL_PERSISTENCE
        )
    )

    # --------------------------------------------------------
    # CRITICAL
    # --------------------------------------------------------

    critical_condition = (
        very_high_movement
        &
        (
            tracking_reliability
            >= CRITICAL_RELIABILITY_THRESHOLD
        )
    )

    data.loc[
        critical_condition,
        "risk_state"
    ] = "CRITICAL"

    data.loc[
        critical_condition,
        "risk_reason"
    ] = (
        "Persistent extreme movement "
        "with reliable tracking"
    )

    # --------------------------------------------------------
    # EXTREME MOVEMENT WITH REDUCED CONFIDENCE
    # --------------------------------------------------------

    uncertain_critical = (
        very_high_movement
        &
        (
            tracking_reliability
            < CRITICAL_RELIABILITY_THRESHOLD
        )
        &
        (
            tracking_reliability
            >= UNRELIABLE_TRACKING_THRESHOLD
        )
    )

    data.loc[
        uncertain_critical,
        "risk_state"
    ] = "HIGH_RISK"

    data.loc[
        uncertain_critical,
        "risk_reason"
    ] = (
        "Extreme movement detected but "
        "tracking reliability is below "
        "CRITICAL threshold"
    )

    # --------------------------------------------------------
    # UNRELIABLE TRACKING
    # --------------------------------------------------------

    unreliable = (
        tracking_reliability
        < UNRELIABLE_TRACKING_THRESHOLD
    )

    if unreliable:

        data["risk_state"] = (
            "TRACKING_UNRELIABLE"
        )

        data["risk_reason"] = (
            "Tracking reliability is too low "
            "for confident risk assessment"
        )

    return data


# ============================================================
# COMPLETE LIVE ANALYSIS
# ============================================================

def analyze_tracking_csv(
    tracking_csv_path
):

    print("\n")
    print("=" * 60)
    print("          LIVE RISK ANALYSIS")
    print("=" * 60)

    tracking_csv_path = Path(
        tracking_csv_path
    )

    if not tracking_csv_path.exists():

        raise FileNotFoundError(
            f"Tracking CSV not found: "
            f"{tracking_csv_path}"
        )

    # ========================================================
    # 1. LOAD TRACKING DATA
    # ========================================================

    tracking_df = pd.read_csv(
        tracking_csv_path
    )

    print(
        "\nTracking CSV:",
        tracking_csv_path
    )

    print(
        "Tracking rows:",
        len(tracking_df)
    )

    # ========================================================
    # 2. TRACKING RELIABILITY
    # ========================================================

    reliability_info = (
        calculate_tracking_reliability(
            tracking_df
        )
    )

    tracking_reliability = (
        reliability_info[
            "tracking_reliability"
        ]
    )

    print(
        "\nTracking reliability:",
        round(
            tracking_reliability,
            3
        )
    )

    # ========================================================
    # 3. FRAME-LEVEL FEATURES
    # ========================================================

    numeric_columns = [
        "frame",
        "timestamp",
        "x",
        "y",
        "width",
        "height",
        "movement_x",
        "movement_y",
        "displacement",
        "velocity_x",
        "velocity_y",
        "speed",
        "density",
    ]

    feature_df = tracking_df.copy()

    for column in numeric_columns:

        if column in feature_df.columns:

            feature_df[column] = pd.to_numeric(
                feature_df[column],
                errors="coerce"
            )

    feature_df = feature_df.dropna(
        subset=[
            "person_id",
            "frame",
            "timestamp",
            "speed",
            "displacement",
            "velocity_x",
            "velocity_y",
            "direction",
            "density",
        ]
    ).copy()

    from ml.features.basic_features import (
        calculate_basic_features
    )

    from ml.features.temporal_features import (
        calculate_temporal_features
    )

    from ml.features.directional_features import (
        calculate_directional_features
    )

    cleaned = detect_tracking_anomalies(
        feature_df
    )

    movement_df = cleaned[
        ~cleaned["tracking_anomaly"]
    ].copy()

    basic = calculate_basic_features(
        cleaned,
        movement_df
    )

    temporal = calculate_temporal_features(
        cleaned
    )

    directional = calculate_directional_features(
        movement_df
    )

    features = basic.merge(
        temporal,
        on="frame",
        how="left"
    )

    features = features.merge(
        directional,
        on="frame",
        how="left"
    )

    timestamps = (
        cleaned.groupby("frame")["timestamp"]
        .first()
        .reset_index()
    )

    features = features.merge(
        timestamps,
        on="frame",
        how="left"
    )

    features = features.fillna(0)

    # ========================================================
    # 4. FIVE-SECOND WINDOWS
    # ========================================================

    windowed = create_time_windows(
        features,
        window_seconds=5
    )

    print(
        "5-second windows:",
        len(windowed)
    )

    if windowed.empty:

        raise RuntimeError(
            "No time windows were generated."
        )

    # ========================================================
    # 5. BEHAVIORAL FEATURES
    # ========================================================

    behavioral = (
        calculate_behavioral_features(
            windowed
        )
    )

    # ========================================================
    # 6. LOAD NORMAL REFERENCE
    # ========================================================

    reference_medians = (
        load_normal_reference()
    )

    # ========================================================
    # 7. MOVEMENT INTENSITY
    # ========================================================

    risk_df = calculate_movement_intensity(
        behavioral,
        reference_medians
    )

    # ========================================================
    # 8. APPLY RISK LOGIC
    # ========================================================

    risk_df = apply_risk_logic(
        risk_df,
        tracking_reliability
    )

    # ========================================================
    # 9. OVERALL VIDEO RISK
    # ========================================================

    # Risk priority for the complete video.
    #
    # CRITICAL > HIGH_RISK > NORMAL
    #
    # TRACKING_UNRELIABLE is handled separately
    # because it represents lack of confidence,
    # not a higher danger level.

    risk_priority = {
        "NORMAL": 0,
        "HIGH_RISK": 1,
        "CRITICAL": 2,
    }

    valid_risk_states = [
        state
        for state in risk_df["risk_state"]
        if state in risk_priority
    ]

    if valid_risk_states:

        overall_risk_state = max(
            valid_risk_states,
            key=lambda state:
            risk_priority[state]
        )

        # Find the most important window.
        overall_row = (
            risk_df[
                risk_df["risk_state"]
                == overall_risk_state
            ]
            .sort_values(
                "movement_intensity_index",
                ascending=False
            )
            .iloc[0]
        )

    else:

        overall_risk_state = (
            "TRACKING_UNRELIABLE"
        )

        overall_row = risk_df.iloc[-1]

    # ========================================================
    # 10. FINAL RESULT
    # ========================================================

    result = {

        "risk_state":
            str(overall_risk_state),

        "risk_reason":
            str(overall_row["risk_reason"]),

        "movement_intensity_index":
            round(
                float(
                    overall_row[
                        "movement_intensity_index"
                    ]
                ),
                3
            ),

        "tracking_reliability":
            round(
                float(
                    tracking_reliability
                ),
                3
            ),

        "tracking_quality":
            str(
                overall_row[
                    "tracking_quality"
                ]
            ),

        "window_id":
            int(
                overall_row[
                    "window_id"
                ]
            ),

        "total_windows":
            len(risk_df),
    }

    # ========================================================
    # 11. WINDOW RESULTS
    # ========================================================

    windows = []

    for _, row in risk_df.iterrows():

        windows.append(
            {
                "window_id":
                    int(
                        row["window_id"]
                    ),

                "window_start":
                    float(
                        row["window_start"]
                    ),

                "window_end":
                    float(
                        row["window_end"]
                    ),

                "movement_intensity_index":
                    round(
                        float(
                            row[
                                "movement_intensity_index"
                            ]
                        ),
                        3
                    ),

                "risk_state":
                    str(
                        row["risk_state"]
                    ),

                "risk_reason":
                    str(
                        row["risk_reason"]
                    ),
            }
        )

    result["windows"] = windows

    # ========================================================
    # 12. DISPLAY FINAL RESULT
    # ========================================================

    print("\n==========================================")
    print("              FINAL RESULT")
    print("==========================================")

    print(
        "Overall video risk:",
        result["risk_state"]
    )

    print(
        "Risk reason:",
        result["risk_reason"]
    )

    print(
        "Peak movement intensity:",
        result[
            "movement_intensity_index"
        ]
    )

    print(
        "Tracking reliability:",
        result[
            "tracking_reliability"
        ]
    )

    print(
        "Most severe window:",
        result[
            "window_id"
        ]
    )

    print(
        "Total windows:",
        result[
            "total_windows"
        ]
    )

    print(
        "=========================================="
    )

    return result