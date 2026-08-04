"""
Explainability Module
-----------------------
Applies SHAP to a trained model (default: XGBoost, since tree models work
with the fast TreeExplainer) to identify which features drove each
classification. Saves a summary plot to reports/figures/.

Note on SHAP array shapes: different SHAP versions return multi-class
TreeExplainer output in different shapes —
  - older versions: a list of (n_samples, n_features) arrays, one per class
  - newer versions (e.g. 0.5x): a single (n_samples, n_features, n_classes)
    array
`_per_class_shap_list()` below normalizes either shape into a consistent
list-of-per-class-arrays so summary_plot always receives what it expects.
Without this, summary_plot silently misreads the array axes and produces a
meaningless plot (e.g. treating the number of classes as if it were a
handful of features).

Usage:
    python src/explainability.py --dataset nslkdd --model xgboost
"""

import argparse
import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def per_class_shap_list(shap_values, n_features):
    """Normalizes any SHAP multiclass TreeExplainer output into a list of
    (n_samples, n_features) arrays, one per class. Handles:
      - list of 2D arrays (older SHAP versions)
      - single 3D array shaped (n_samples, n_features, n_classes) (newer)
      - single 3D array shaped (n_classes, n_samples, n_features) (rare)
      - single 2D array (binary classification — wrapped as a 1-item list)
    """
    if isinstance(shap_values, list):
        return shap_values

    arr = np.asarray(shap_values)

    if arr.ndim == 2:
        return [arr]

    if arr.ndim == 3:
        if arr.shape[1] == n_features:
            # (n_samples, n_features, n_classes) -> split on last axis
            return [arr[:, :, c] for c in range(arr.shape[2])]
        elif arr.shape[2] == n_features:
            # (n_classes, n_samples, n_features) -> split on first axis
            return [arr[c] for c in range(arr.shape[0])]

    raise ValueError(f"Unrecognized SHAP values shape: {arr.shape}")


def explain(dataset, model_name, sample_size=500):
    test = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_test.csv"))
    X_test = test.drop(columns=["attack_category"])
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)

    model_path = os.path.join(MODELS_DIR, f"{dataset}_{model_name}.pkl")
    model = joblib.load(model_path)

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_sample)
    shap_list = per_class_shap_list(raw_shap_values, n_features=X_sample.shape[1])

    print(f"Detected {len(shap_list)} class(es), "
          f"{X_sample.shape[1]} features per class.")

    # Global summary plot: average absolute SHAP value across all classes,
    # so every one of the real features (not classes) appears correctly.
    plt.figure()
    shap.summary_plot(shap_list, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{dataset}_{model_name}_shap_summary.png"))
    plt.close()

    print(f"SHAP summary plot saved to reports/figures/"
          f"{dataset}_{model_name}_shap_summary.png")

    return explainer, shap_list, X_sample


def explain_single_record(explainer, model, record: pd.DataFrame, class_names):
    """Returns the top contributing features for a single traffic record's
    prediction — used by the dashboard to show per-record explanations."""
    raw_shap_values = explainer.shap_values(record)
    shap_list = per_class_shap_list(raw_shap_values, n_features=record.shape[1])

    pred_class = model.predict(record)[0]
    class_shap = shap_list[pred_class][0]

    contributions = pd.Series(class_shap, index=record.columns).sort_values(
        key=abs, ascending=False
    )
    return class_names[pred_class], contributions.head(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nslkdd", "unsw"])
    parser.add_argument("--model", default="xgboost",
                         choices=["xgboost", "random_forest"],
                         help="SHAP TreeExplainer supports tree-based models directly.")
    args = parser.parse_args()
    explain(args.dataset, args.model)
