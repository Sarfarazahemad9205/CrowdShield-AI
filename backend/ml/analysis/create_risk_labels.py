from pathlib import Path
import pandas as pd


INPUT_DIR = Path("ml/dataset/behavioral")
OUTPUT_DIR = Path("ml/dataset/labeled")


def assign_risk_label(row):
    """
    Assign a research-oriented risk label based on
    abnormal crowd behavior.

    Labels:
        0 -> NORMAL
        1 -> HIGH_RISK
        2 -> CRITICAL_RISK
    """

    score = row["abnormal_behavior_score"]

    # Very strong abnormal behavior
    if score >= 2.0:
        return 2

    # Moderate abnormal behavior
    elif score >= 0.75:
        return 1

    # Relatively stable behavior
    else:
        return 0


def label_dataset(df):
    """
    Add numerical and human-readable risk labels.
    """

    data = df.copy()

    data["risk_label"] = data.apply(
        assign_risk_label,
        axis=1
    )

    data["risk_state"] = data["risk_label"].map({
        0: "NORMAL",
        1: "HIGH_RISK",
        2: "CRITICAL_RISK"
    })

    return data


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Only use independent datasets.
    files = [
        INPUT_DIR / "umn_crowd_features_behavioral.csv",
        INPUT_DIR / "railway_crowd_features_behavioral.csv",
        INPUT_DIR / "Bengaluru_Crowd_stampede_features_behavioral.csv",
    ]

    print("=" * 60)
    print("RISK LABEL CREATION")
    print("=" * 60)

    for file in files:

        if not file.exists():
            print(f"\nFile not found: {file}")
            continue

        print(f"\nProcessing: {file.name}")

        df = pd.read_csv(file)

        labeled = label_dataset(df)

        output_file = (
            OUTPUT_DIR /
            file.name.replace(
                "_behavioral.csv",
                "_labeled.csv"
            )
        )

        labeled.to_csv(
            output_file,
            index=False
        )

        print("Total windows:", len(labeled))

        print("\nRisk distribution:")
        print(
            labeled["risk_state"]
            .value_counts()
            .sort_index()
        )

        print("\nSaved to:")
        print(output_file)

    print("\n" + "=" * 60)
    print("RISK LABEL CREATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()