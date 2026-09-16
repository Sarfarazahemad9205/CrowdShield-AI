import pandas as pd
from pathlib import Path


# ============================================================
# FIND ALL TRACKING CSV FILES
# ============================================================

PROCESSED_DIR = Path("processed")

tracking_files = sorted(
    PROCESSED_DIR.glob("*_processed_data.csv")
)

print("=" * 60)
print("STAMPEDEGUARD TRACKING ANOMALY ANALYSIS")
print("=" * 60)

print("Tracking files found:", len(tracking_files))

for file in tracking_files:
    print(" -", file.name)


# ============================================================
# ANALYZE EACH VIDEO
# ============================================================

for file in tracking_files:

    print("\n")
    print("=" * 60)
    print(f"ANALYZING: {file.name}")
    print("=" * 60)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    df = pd.read_csv(file)

    df = df.sort_values(
        ["person_id", "frame"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # CALCULATE POSITION CHANGE
    # --------------------------------------------------------

    df["position_change"] = (
        df.groupby("person_id")["x"].diff().abs()
        +
        df.groupby("person_id")["y"].diff().abs()
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    print("\n===== BASIC INFORMATION =====")

    print("Total tracking rows:", len(df))

    print("Unique people:",
          df["person_id"].nunique())

    print("Unique frames:",
          df["frame"].nunique())

    # --------------------------------------------------------
    # EXTREME POSITION CHANGES
    # --------------------------------------------------------

    print("\n===== EXTREME POSITION CHANGES =====")

    extreme = df[
        df["position_change"] > 100
    ].sort_values(
        "position_change",
        ascending=False
    )

    print(
        "Rows with position change > 100 pixels:",
        len(extreme)
    )

    # Percentage of rows
    anomaly_percentage = (
        len(extreme) / len(df) * 100
        if len(df) > 0
        else 0
    )

    print(
        "Percentage of rows > 100 pixels:",
        round(anomaly_percentage, 3),
        "%"
    )

    # --------------------------------------------------------
    # TOP 30 EXTREME MOVEMENTS
    # --------------------------------------------------------

    print("\n===== TOP 30 EXTREME MOVEMENTS =====")

    columns = [
        "frame",
        "person_id",
        "x",
        "y",
        "movement_x",
        "movement_y",
        "displacement",
        "speed",
        "position_change",
        "people_count"
    ]

    # Only use columns that exist
    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print(
        extreme[available_columns]
        .head(30)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # LARGEST TRACKING JUMPS
    # --------------------------------------------------------

    print("\n===== LARGEST TRACKING JUMPS =====")

    largest_jumps = (
        df[available_columns]
        .sort_values(
            "position_change",
            ascending=False
        )
        .head(20)
    )

    print(
        largest_jumps.to_string(index=False)
    )

    # --------------------------------------------------------
    # SPEED EXTREMES
    # --------------------------------------------------------

    print("\n===== SPEED EXTREMES =====")

    if "speed" in df.columns:

        print(
            "Maximum speed:",
            round(df["speed"].max(), 2)
        )

        print(
            "95th percentile speed:",
            round(df["speed"].quantile(0.95), 2)
        )

        print(
            "99th percentile speed:",
            round(df["speed"].quantile(0.99), 2)
        )

    # --------------------------------------------------------
    # DISPLACEMENT EXTREMES
    # --------------------------------------------------------

    print("\n===== DISPLACEMENT EXTREMES =====")

    if "displacement" in df.columns:

        print(
            "Maximum displacement:",
            round(df["displacement"].max(), 2)
        )

        print(
            "95th percentile displacement:",
            round(
                df["displacement"].quantile(0.95),
                2
            )
        )

        print(
            "99th percentile displacement:",
            round(
                df["displacement"].quantile(0.99),
                2
            )
        )

    # --------------------------------------------------------
    # ANOMALIES BY PERSON
    # --------------------------------------------------------

    print("\n===== ANOMALIES BY PERSON =====")

    if len(extreme) > 0:

        anomaly_by_person = (
            extreme
            .groupby("person_id")
            .size()
            .sort_values(ascending=False)
            .head(20)
        )

        print(anomaly_by_person)

    else:

        print("No position jumps greater than 100 pixels.")

    # --------------------------------------------------------
    # TRAJECTORY CONTINUITY CHECK
    # --------------------------------------------------------

    print("\n===== TRAJECTORY CONTINUITY CHECK =====")

    # Take the 10 largest jumps automatically
    suspicious_rows = (
        extreme
        .head(10)
    )

    if len(suspicious_rows) == 0:

        print("No suspicious jumps found.")

    else:

        for _, row in suspicious_rows.iterrows():

            person_id = row["person_id"]
            frame = int(row["frame"])

            print("\n------------------------------------------")

            print(
                f"Person {person_id} around frame {frame}"
            )

            print("------------------------------------------")

            trajectory = df[
                (df["person_id"] == person_id)
                &
                (
                    df["frame"].between(
                        frame - 3,
                        frame + 3
                    )
                )
            ]

            trajectory_columns = [
                "frame",
                "person_id",
                "x",
                "y",
                "movement_x",
                "movement_y",
                "displacement",
                "speed",
                "direction"
            ]

            trajectory_columns = [
                column
                for column in trajectory_columns
                if column in trajectory.columns
            ]

            print(
                trajectory[trajectory_columns]
                .to_string(index=False)
            )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)
print("ANOMALY ANALYSIS COMPLETED")
print("=" * 60)