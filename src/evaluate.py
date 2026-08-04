"""
Evaluation Module
------------------
Loads all trained models for a dataset and generates a complete evaluation
report: accuracy/precision/recall/F1, per-class classification report,
confusion matrix, ROC curves (one-vs-rest, with AUC per class), and
precision-recall curves. Everything is saved into one organized folder:

    reports/<dataset>/
        model_comparison.csv              <- summary table, all models
        <model>_classification_report.txt <- per-class precision/recall/F1
        <model>_confusion_matrix.png
        <model>_roc_curve.png
        <model>_precision_recall_curve.png

Usage:
    python src/evaluate.py --dataset nslkdd
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, roc_curve, auc, precision_recall_curve,
    average_precision_score,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_test(dataset):
    test = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_test.csv"))
    X_test = test.drop(columns=["attack_category"])
    y_test = test["attack_category"]
    artifacts = joblib.load(os.path.join(PROCESSED_DIR, f"{dataset}_artifacts.pkl"))
    class_names = artifacts["target_encoder"].classes_
    return X_test, y_test, class_names


def predict(model_name, model, X_test):
    if model_name == "neural_net":
        return model.predict(X_test, verbose=0)  # returns probabilities directly
    return model.predict_proba(X_test)


def plot_confusion_matrix(cm, class_names, title, save_path):
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curves(y_test_bin, y_proba, class_names, title, save_path):
    """One-vs-rest ROC curve per class, plus micro-average."""
    n_classes = len(class_names)
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    for i, cname in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                  label=f"{cname} (AUC = {roc_auc[i]:.3f})")
    plt.plot(fpr["micro"], tpr["micro"], color="black", linestyle="--", lw=2,
              label=f"micro-average (AUC = {roc_auc['micro']:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle=":", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return roc_auc


def plot_precision_recall_curves(y_test_bin, y_proba, class_names, title, save_path):
    n_classes = len(class_names)
    plt.figure(figsize=(7, 6))
    colors = plt.cm.tab10.colors
    ap_scores = {}
    for i, cname in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        ap_scores[cname] = ap
        plt.plot(recall, precision, color=colors[i % len(colors)], lw=2,
                  label=f"{cname} (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return ap_scores


def main(dataset):
    out_dir = os.path.join(REPORTS_ROOT, dataset)
    os.makedirs(out_dir, exist_ok=True)

    X_test, y_test, class_names = load_test(dataset)
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    model_files = {
        "random_forest": os.path.join(MODELS_DIR, f"{dataset}_random_forest.pkl"),
        "xgboost": os.path.join(MODELS_DIR, f"{dataset}_xgboost.pkl"),
        "neural_net": os.path.join(MODELS_DIR, f"{dataset}_neural_net.keras"),
    }

    results = []
    for name, path in model_files.items():
        if not os.path.exists(path):
            print(f"Skipping {name} (not found at {path})")
            continue

        try:
            if name == "neural_net":
                from tensorflow import keras
                model = keras.models.load_model(path)
            else:
                model = joblib.load(path)
        except Exception as e:
            print(f"WARNING: could not load '{name}' ({type(e).__name__}: {e}). "
                  f"Skipping this model — results for other models are unaffected.")
            continue

        y_proba = predict(name, model, X_test)
        y_pred = np.argmax(y_proba, axis=1)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        results.append({
            "model": name, "accuracy": acc, "precision_macro": prec,
            "recall_macro": rec, "f1_macro": f1,
        })

        # Per-class classification report (text)
        report_txt = classification_report(
            y_test, y_pred, target_names=class_names, zero_division=0
        )
        with open(os.path.join(out_dir, f"{name}_classification_report.txt"), "w") as f:
            f.write(f"Model: {name} | Dataset: {dataset}\n\n")
            f.write(report_txt)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, class_names, f"{name} — {dataset} — Confusion Matrix",
            os.path.join(out_dir, f"{name}_confusion_matrix.png"),
        )

        # ROC curves (one-vs-rest, per class + micro-average)
        roc_auc = plot_roc_curves(
            y_test_bin, y_proba, class_names,
            f"{name} — {dataset} — ROC Curves (One-vs-Rest)",
            os.path.join(out_dir, f"{name}_roc_curve.png"),
        )

        # Precision-Recall curves
        ap_scores = plot_precision_recall_curves(
            y_test_bin, y_proba, class_names,
            f"{name} — {dataset} — Precision-Recall Curves",
            os.path.join(out_dir, f"{name}_precision_recall_curve.png"),
        )

        print(f"{name}: acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f} "
              f"macro_AUC={np.mean([v for k, v in roc_auc.items() if k != 'micro']):.4f}")

    df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    out_path = os.path.join(out_dir, "model_comparison.csv")
    df.to_csv(out_path, index=False)

    # Keep a copy at the old flat location too, since the dashboard reads
    # from here to auto-select the best model.
    df.to_csv(os.path.join(REPORTS_ROOT, f"{dataset}_model_comparison.csv"), index=False)

    print(f"\nAll results saved to {out_dir}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nslkdd", "unsw"])
    args = parser.parse_args()
    main(args.dataset)