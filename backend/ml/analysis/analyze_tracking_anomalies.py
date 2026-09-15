import pandas as pd


# ============================================================
# FILE PATH
# ============================================================

INPUT_FILE = "processed/stampede_video_processed_data.csv"


# ============================================================
# LOAD TRACKING DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

df = df.sort_values(
    ["person_id", "frame"]
).reset_index(drop=True)


# ============================================================
# CALCULATE POSITION CHANGE
# ============================================================

df["position_change"] = (
    df.groupby("person_id")["x"].diff().abs()
    +
    df.groupby("person_id")["y"].diff().abs()
)


# ============================================================
# SELECT EXTREME FRAMES
# ============================================================

target_frames = [
    2,
    630,
    659,
    1377,
    2073,
    2160,
    2226,
    2234,
]


print("==========================================")
print("STAMPedeGUARD TRACKING ANOMALY ANALYSIS")
print("==========================================")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n===== BASIC INFORMATION =====")

print("Total tracking rows:", len(df))
print("Unique people:", df["person_id"].nunique())
print("Unique frames:", df["frame"].nunique())


# ============================================================
# EXTREME POSITION CHANGES
# ============================================================

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


print(
    extreme[
        [
            "frame",
            "person_id",
            "x",
            "y",
            "movement_x",
            "movement_y",
            "displacement",
            "speed",
            "position_change",
        ]
    ]
    .head(30)
    .to_string(index=False)
)


# ============================================================
# INSPECT IMPORTANT FRAMES
# ============================================================

print("\n===== TARGET FRAME ANALYSIS =====")


for frame in target_frames:

    print("\n------------------------------------------")
    print("FRAME:", frame)
    print("------------------------------------------")

    frame_data = df[
        df["frame"] == frame
    ].copy()

    print(
        frame_data[
            [
                "frame",
                "person_id",
                "x",
                "y",
                "movement_x",
                "movement_y",
                "displacement",
                "speed",
                "direction",
            ]
        ]
        .sort_values(
            "displacement",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# FIND LARGEST TRACKING JUMPS
# ============================================================

print("\n===== LARGEST TRACKING JUMPS =====")

largest_jumps = (
    df[
        [
            "frame",
            "person_id",
            "x",
            "y",
            "movement_x",
            "movement_y",
            "displacement",
            "speed",
            "position_change",
        ]
    ]
    .sort_values(
        "position_change",
        ascending=False
    )
    .head(20)
)


print(
    largest_jumps.to_string(index=False)
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==========================================") 
# ============================================================
# TRAJECTORY CONTINUITY CHECK
# ============================================================

print("\n===== TRAJECTORY CONTINUITY CHECK =====")

suspicious_cases = [
    (2333, 2226),
    (2742, 2400),
    (2177, 2160),
    (2675, 2342),
    (2619, 2327),
    (1503, 1377),
    (750, 659),
    (678, 630),
    (4, 2),
]


for person_id, frame in suspicious_cases:

    print("\n------------------------------------------")
    print(
        f"Person {person_id} around frame {frame}"
    )
    print("------------------------------------------")

    trajectory = df[
        (df["person_id"] == person_id) &
        (df["frame"].between(frame - 3, frame + 3))
    ][
        [
            "frame",
            "person_id",
            "x",
            "y",
            "movement_x",
            "movement_y",
            "displacement",
            "speed",
            "direction",
        ]
    ]

    print(
        trajectory.to_string(index=False)
    )