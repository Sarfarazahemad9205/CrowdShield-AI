import pandas as pd
from pathlib import Path


# ==========================================
# 1. File path
# ==========================================

INPUT_FILE = Path("ml/dataset/crowd_features.csv")


# ==========================================
# 2. Load dataset
# ==========================================

df = pd.read_csv(INPUT_FILE)


print("==========================================")
print("       CROWD FEATURE ANALYSIS")
print("==========================================")


# ==========================================
# 3. Basic information
# ==========================================

print("\n===== BASIC INFORMATION =====")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))

print("Columns:")
print(df.columns.tolist())


# ==========================================
# 4. Missing values
# ==========================================

print("\n===== MISSING VALUES =====")

missing = df.isnull().sum()

print(missing)


# ==========================================
# 5. Duplicate rows
# ==========================================

print("\n===== DUPLICATE ROWS =====")

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)


# ==========================================
# 6. Duplicate frames
# ==========================================

print("\n===== DUPLICATE FRAMES =====")

duplicate_frames = df["frame"].duplicated().sum()

print("Duplicate frame entries:", duplicate_frames)


# ==========================================
# 7. Frame information
# ==========================================

print("\n===== FRAME INFORMATION =====")

print("Minimum frame:", df["frame"].min())
print("Maximum frame:", df["frame"].max())
print("Unique frames:", df["frame"].nunique())


# ==========================================
# 8. People count analysis
# ==========================================

print("\n===== PEOPLE COUNT =====")

print("Minimum:", df["people_count"].min())
print("Maximum:", df["people_count"].max())
print("Average:", df["people_count"].mean())
print("Median:", df["people_count"].median())


# ==========================================
# 9. Speed analysis
# ==========================================

print("\n===== SPEED =====")

print("Average speed:", df["avg_speed"].mean())
print("Median speed:", df["avg_speed"].median())
print("Maximum average speed:", df["avg_speed"].max())


# ==========================================
# 10. Extreme speed values
# ==========================================

print("\n===== EXTREME SPEED VALUES =====")

speed_limit = df["avg_speed"].quantile(0.99)

print("99th percentile speed:", speed_limit)

extreme_speed = df[
    df["avg_speed"] > speed_limit
]

print("Number of extreme-speed frames:", len(extreme_speed))

print(
    extreme_speed[
        [
            "frame",
            "timestamp",
            "people_count",
            "avg_speed",
            "max_speed"
        ]
    ].head(10).to_string(index=False)
)


# ==========================================
# 11. Displacement analysis
# ==========================================

print("\n===== DISPLACEMENT =====")

print(
    "Average displacement:",
    df["avg_displacement"].mean()
)

print(
    "Maximum displacement:",
    df["avg_displacement"].max()
)

print(
    "Median displacement:",
    df["avg_displacement"].median()
)


# ==========================================
# 12. Density analysis
# ==========================================

print("\n===== DENSITY =====")

print(
    "Minimum density:",
    df["avg_density"].min()
)

print(
    "Maximum density:",
    df["avg_density"].max()
)

print(
    "Average density:",
    df["avg_density"].mean()
)

print(
    "Median density:",
    df["avg_density"].median()
)


# ==========================================
# 13. Velocity analysis
# ==========================================

print("\n===== VELOCITY =====")

print(
    "Average X velocity:",
    df["avg_velocity_x"].mean()
)

print(
    "Average Y velocity:",
    df["avg_velocity_y"].mean()
)

print(
    "Maximum absolute X velocity:",
    df["avg_velocity_x"].abs().max()
)

print(
    "Maximum absolute Y velocity:",
    df["avg_velocity_y"].abs().max()
)


# ==========================================
# 14. Feature statistics
# ==========================================

print("\n===== FEATURE STATISTICS =====")

feature_columns = [
    "people_count",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_displacement",
    "max_displacement",
    "avg_velocity_x",
    "avg_velocity_y",
    "avg_width",
    "avg_height",
    "avg_density"
]

print(
    df[feature_columns].describe().to_string()
)


# ==========================================
# 15. Correlation analysis
# ==========================================

print("\n===== FEATURE CORRELATION =====")

correlation = df[feature_columns].corr()

print(
    correlation.round(2).to_string()
)


# ==========================================
# 16. Check for suspicious values
# ==========================================

print("\n===== SUSPICIOUS VALUES =====")


negative_speed = (df["avg_speed"] < 0).sum()
negative_displacement = (df["avg_displacement"] < 0).sum()
negative_density = (df["avg_density"] < 0).sum()


print("Negative average speed:", negative_speed)
print("Negative average displacement:", negative_displacement)
print("Negative average density:", negative_density)


# ==========================================
# 17. Check very large movement values
# ==========================================

print("\n===== LARGE MOVEMENT VALUES =====")

large_movement = df[
    (df["avg_speed"] > 1000) |
    (df["max_speed"] > 2000) |
    (df["max_displacement"] > 100)
]

print(
    "Frames with unusually large movement:",
    len(large_movement)
)

print(
    large_movement[
        [
            "frame",
            "timestamp",
            "people_count",
            "avg_speed",
            "max_speed",
            "avg_displacement",
            "max_displacement"
        ]
    ].head(20).to_string(index=False)
)

 # ==========================================
# 18. Temporal feature analysis
# ==========================================

print("\n===== TEMPORAL FEATURES =====")

temporal_columns = [
    "avg_acceleration",
    "max_acceleration",
    "acceleration_std",
    "avg_speed_change",
    "max_speed_change",
    "avg_velocity_change",
    "max_velocity_change",
    "avg_displacement_change"
]

print(
    df[temporal_columns].describe().to_string()
)


# ==========================================
# 19. Directional feature analysis
# ==========================================

print("\n===== DIRECTIONAL FEATURES =====")

directional_columns = [
    "direction_alignment",
    "directional_disorder"
]

print(
    df[directional_columns].describe().to_string()
)


# ==========================================
# 20. Directional range validation
# ==========================================

print("\n===== DIRECTIONAL RANGE CHECK =====")

alignment_invalid = (
    (df["direction_alignment"] < 0) |
    (df["direction_alignment"] > 1)
).sum()

disorder_invalid = (
    (df["directional_disorder"] < 0) |
    (df["directional_disorder"] > 1)
).sum()

print(
    "Direction alignment outside [0,1]:",
    alignment_invalid
)

print(
    "Directional disorder outside [0,1]:",
    disorder_invalid
)


# ==========================================
# 21. Temporal feature negative values
# ==========================================

print("\n===== TEMPORAL FEATURE VALIDATION =====")

for column in temporal_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    print(
        f"{column}: {negative_count} negative values"
    )


# ==========================================
# 22. Temporal feature outliers
# ==========================================

print("\n===== TEMPORAL OUTLIERS =====")

for column in temporal_columns:

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + 1.5 * iqr

    outliers = (
        df[column] > upper_limit
    ).sum()

    print(
        f"{column}: "
        f"{outliers} potential outliers "
        f"(upper limit = {upper_limit:.2f})"
    )


# ==========================================
# 23. Correlation of all features
# ==========================================

print("\n===== ALL FEATURE CORRELATIONS =====")

all_feature_columns = [
    "people_count",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_displacement",
    "max_displacement",
    "avg_velocity_x",
    "avg_velocity_y",
    "avg_acceleration",
    "max_acceleration",
    "acceleration_std",
    "avg_speed_change",
    "max_speed_change",
    "avg_velocity_change",
    "max_velocity_change",
    "avg_displacement_change",
    "direction_alignment",
    "directional_disorder",
    "avg_width",
    "avg_height",
    "avg_density"
]

correlation = df[all_feature_columns].corr()

print(
    correlation.round(2).to_string()
)


# ==========================================
# 24. Highly correlated feature pairs
# ==========================================

print("\n===== HIGH CORRELATION PAIRS =====")

for i in range(len(correlation.columns)):

    for j in range(i + 1, len(correlation.columns)):

        value = correlation.iloc[i, j]

        if abs(value) >= 0.90:

            print(
                f"{correlation.columns[i]} <-> "
                f"{correlation.columns[j]}: "
                f"{value:.3f}"
            )