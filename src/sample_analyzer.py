import argparse
import sys
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from sample_calculator import calculate_sample

COLUMN = "category"


def evaluate_predictions(manual_df, automatic_df):
    y_true = manual_df[COLUMN]
    y_pred = automatic_df[COLUMN]

    labels = sorted(y_true.unique())

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted"
    )

    return cm, acc, precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Sample Analyzer")

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset or manual sample"
    )

    parser.add_argument(
        "--sample",
        choices=["yes", "no"],
        required=True,
        help="Indicates whether the dataset is already a sample"
    )

    parser.add_argument(
        "--level",
        type=int,
        help="Confidence level (90, 95, 99)"
    )

    parser.add_argument(
        "--interval",
        type=float,
        help="Confidence interval percentage (e.g. 5)"
    )

    parser.add_argument(
        "--output",
        default="manual_sample.csv",
        help="Output path for generated sample"
    )

    parser.add_argument(
        "--metrics-output",
        help="Path to CSV file where evaluation metrics will be saved"
    )

    parser.add_argument(
        "--compare",
        help="Path to automatically labeled sample"
    )

    args = parser.parse_args()

    if args.sample == "no":
        if args.level is None or args.interval is None:
            print("ERROR: --level and --interval are required")
            sys.exit(1)

        df = pd.read_csv(args.dataset)

        sample_df = calculate_sample(
            df,
            args.level,
            args.interval
        )

        sample_df.to_csv(args.output, index=False)

        print("\n[SAMPLE GENERATED]")
        print(f"Instances: {len(sample_df)}")
        print(f"Saved to: {args.output}")


    else:
        if args.compare is None:
            print("ERROR: --compare is required in evaluation mode")
            sys.exit(1)

        manual_df = pd.DataFrame

        try:
            if args.dataset.endswith('.csv'):
                manual_df = pd.read_csv(args.dataset)
            elif args.dataset.endswith('.json'):
                manual_df = pd.read_json(args.dataset)
            if args.compare.endswith('.csv'):
                automatic_df = pd.read_csv(args.compare)
            elif args.compare.endswith('.json'):
                automatic_df = pd.read_json(args.compare)
            else:
                print("Unsupported file format. Use CSV or JSON.")
                return
        except Exception as e:
            print(f"Error while opening file: {e}")
            return

        if COLUMN not in manual_df.columns:
            sys.exit(f"Missing column '{COLUMN}' in manual sample")

        if COLUMN not in automatic_df.columns:
            sys.exit(f"Missing column '{COLUMN}' in automatic sample")

        cm, acc, prec, rec, f1 = evaluate_predictions(
            manual_df,
            automatic_df
        )

        metrics_df = pd.DataFrame([{
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        }])

        if args.metrics_output:
            metrics_df.to_csv(args.metrics_output, index=False)

        if args.metrics_output:
            base_path = args.metrics_output.replace(".csv", "")
            pd.DataFrame(cm).to_csv(base_path + "_confusion_matrix.csv", index=False)

        print("\nCONFUSION MATRIX")
        print(cm)

        print("\nMETRICS")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")


if __name__ == "__main__":
    main()
