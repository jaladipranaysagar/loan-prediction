"""Train and evaluate loan eligibility models, then write test predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TARGET = "Loan_Status"
ID_COLUMN = "Loan_ID"


def build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    features = frame.drop(columns=[ID_COLUMN, TARGET], errors="ignore")
    categorical = features.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]

    numeric_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_steps, numeric), ("categorical", categorical_steps, categorical)]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=DATA_DIR / "loan_train.csv")
    parser.add_argument("--test", type=Path, default=DATA_DIR / "loan_test.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if TARGET not in train:
        raise ValueError(f"Training data must contain {TARGET!r}.")
    if ID_COLUMN not in test:
        raise ValueError(f"Test data must contain {ID_COLUMN!r}.")

    y = train[TARGET].map({"N": 0, "Y": 1})
    if y.isna().any():
        raise ValueError("Loan_Status must contain only 'Y' and 'N'.")
    X = train.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=args.random_state,
        stratify=y,
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=args.random_state,
        ),
    }
    results: list[tuple[str, float, Pipeline]] = []
    print("Validation results (held-out 20%, stratified):")
    for name, estimator in candidates.items():
        pipeline = Pipeline(
            [("preprocess", build_preprocessor(train)), ("model", estimator)]
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_valid)
        probability = pipeline.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, probability)
        accuracy = accuracy_score(y_valid, pred)
        results.append((name, auc, pipeline))
        print(f"\n{name}: ROC-AUC={auc:.3f} | accuracy={accuracy:.3f}")
        print(classification_report(y_valid, pred, target_names=["N", "Y"], zero_division=0))

    winner_name, _, _ = max(results, key=lambda item: item[1])
    winner = candidates[winner_name]
    final_model = Pipeline(
        [("preprocess", build_preprocessor(train)), ("model", winner)]
    )
    final_model.fit(X, y)

    # Match the sample submission convention: Loan_Status is 0/1.
    test_X = test.drop(columns=[ID_COLUMN], errors="ignore")
    predictions = final_model.predict(test_X).astype(int)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: predictions})
    submission.to_csv(args.output_dir / "submission.csv", index=False)
    joblib.dump(final_model, args.output_dir / "loan_model.joblib")
    print(f"\nSelected model: {winner_name} (highest validation ROC-AUC)")
    print(f"Saved submission: {args.output_dir / 'submission.csv'}")
    print(f"Saved trained model: {args.output_dir / 'loan_model.joblib'}")


if __name__ == "__main__":
    main()
