"""
Exploratory Data Analysis Module
-----------------------------------
Generates every chart typically needed for an IDS research paper's data
analysis / results sections:

  Per-dataset EDA (reports/<dataset>/eda/):
    - class_distribution.png       attack category counts (train vs test)
    - protocol_distribution.png    protocol type countplot
    - service_distribution.png     top-15 service countplot
    - correlation_heatmap.png      numeric feature correlation heatmap
    - feature_distributions.png    histograms of key numeric features
    - boxplots_by_class.png        key features by attack category
    - feature_importance_<model>.png   built-in feature importance (RF/XGB)

  Cross-model / cross-dataset comparison (reports/comparison/):
    - model_comparison_<dataset>.png   grouped bar: acc/prec/rec/F1 per model
    - cross_dataset_generalization.png bar chart of the 2 transfer results

Usage:
    python src/eda.py --dataset nslkdd
    python src/eda.py --dataset unsw
    python src/eda.py --comparison-only     # only the cross-model/dataset charts
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_nslkdd, load_unsw, _map_label

sns.set_theme(style="whitegrid")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "reports")

# A representative subset of numeric features to visualize in detail —
# using all 38+ would produce unreadable/unusable plots for a paper.
KEY_NUMERIC_FEATURES = {
    "nslkdd": ["duration", "src_bytes", "dst_bytes", "count", "srv_count",
               "serror_rate", "same_srv_rate", "dst_host_srv_count"],
    "unsw": ["dur", "sbytes", "dbytes", "rate", "sttl", "dttl",
             "ct_srv_src", "ct_dst_ltm"],
}

CATEGORICAL_COLS = {
    "nslkdd": {"protocol": "protocol_type", "service": "service"},
    "unsw": {"protocol": "proto", "service": "service"},
}


def load_raw_labeled(dataset):
    """Loads raw (unencoded, unscaled) data with the mapped attack_category
    column — best for human-readable EDA plots."""
    if dataset == "nslkdd":
        train, test = load_nslkdd()
    else:
        train, test = load_unsw()
    train["attack_category"] = train["label"].apply(_map_label)
    test["attack_category"] = test["label"].apply(_map_label)
    return train, test


def plot_class_distribution(train, test, dataset, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    order = sorted(train["attack_category"].unique())
    sns.countplot(x="attack_category", data=train, order=order,
                  hue="attack_category", palette="viridis", legend=False, ax=axes[0])
    axes[0].set_title(f"{dataset} — Train Set Class Distribution")
    axes[0].set_xlabel("Attack Category")
    axes[0].tick_params(axis="x", rotation=30)
    for c in axes[0].containers:
        axes[0].bar_label(c)

    sns.countplot(x="attack_category", data=test, order=order,
                  hue="attack_category", palette="magma", legend=False, ax=axes[1])
    axes[1].set_title(f"{dataset} — Test Set Class Distribution")
    axes[1].set_xlabel("Attack Category")
    axes[1].tick_params(axis="x", rotation=30)
    for c in axes[1].containers:
        axes[1].bar_label(c)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150)
    plt.close()


def plot_categorical_distribution(train, col, title, out_path, top_n=None):
    plt.figure(figsize=(9, 5))
    counts = train[col].value_counts()
    if top_n:
        counts = counts.head(top_n)
    sns.barplot(x=counts.values, y=counts.index, hue=counts.index,
                palette="crest", legend=False)
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correlation_heatmap(train, numeric_cols, dataset, out_dir):
    corr = train[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"{dataset} — Correlation Heatmap (Key Numeric Features)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_feature_distributions(train, numeric_cols, dataset, out_dir):
    n = len(numeric_cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, col in enumerate(numeric_cols):
        # log1p helps with the heavy-tailed byte/count features common in
        # network traffic data — raw histograms are unreadable otherwise.
        sns.histplot(np.log1p(train[col].clip(lower=0)), bins=40,
                     color="steelblue", ax=axes[i])
        axes[i].set_title(f"log1p({col})")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.suptitle(f"{dataset} — Key Feature Distributions (log-scaled)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "feature_distributions.png"), dpi=150)
    plt.close()


def plot_boxplots_by_class(train, numeric_cols, dataset, out_dir):
    cols = numeric_cols[:4]  # keep readable — 4 features max
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        plot_df = train.copy()
        plot_df[col] = np.log1p(plot_df[col].clip(lower=0))
        sns.boxplot(x="attack_category", y=col, data=plot_df,
                    hue="attack_category", palette="Set2", legend=False, ax=axes[i])
        axes[i].set_title(f"log1p({col}) by Class")
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle(f"{dataset} — Key Features by Attack Category")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "boxplots_by_class.png"), dpi=150)
    plt.close()


def plot_feature_importance(dataset, model_name, out_dir):
    model_path = os.path.join(MODELS_DIR, f"{dataset}_{model_name}.pkl")
    if not os.path.exists(model_path):
        return
    model = joblib.load(model_path)
    if not hasattr(model, "feature_importances_"):
        return

    processed = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_train.csv"))
    feature_names = processed.drop(columns=["attack_category"]).columns

    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="flare", legend=False)
    plt.title(f"{dataset} — {model_name} — Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"feature_importance_{model_name}.png"), dpi=150)
    plt.close()


def run_dataset_eda(dataset):
    out_dir = os.path.join(REPORTS_ROOT, dataset, "eda")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{dataset}] Loading raw labeled data...")
    train, test = load_raw_labeled(dataset)

    print(f"[{dataset}] Class distribution...")
    plot_class_distribution(train, test, dataset, out_dir)

    cat_cols = CATEGORICAL_COLS[dataset]
    print(f"[{dataset}] Categorical distributions...")
    plot_categorical_distribution(
        train, cat_cols["protocol"], f"{dataset} — Protocol Type Distribution",
        os.path.join(out_dir, "protocol_distribution.png"),
    )
    plot_categorical_distribution(
        train, cat_cols["service"], f"{dataset} — Top 15 Services",
        os.path.join(out_dir, "service_distribution.png"), top_n=15,
    )

    numeric_cols = KEY_NUMERIC_FEATURES[dataset]
    print(f"[{dataset}] Correlation heatmap...")
    plot_correlation_heatmap(train, numeric_cols, dataset, out_dir)

    print(f"[{dataset}] Feature distributions...")
    plot_feature_distributions(train, numeric_cols, dataset, out_dir)

    print(f"[{dataset}] Boxplots by class...")
    plot_boxplots_by_class(train, numeric_cols, dataset, out_dir)

    print(f"[{dataset}] Feature importance plots...")
    for model_name in ["random_forest", "xgboost"]:
        plot_feature_importance(dataset, model_name, out_dir)

    print(f"[{dataset}] All EDA plots saved to {out_dir}/")


def plot_model_comparison(dataset):
    """Grouped bar chart comparing all 3 models on accuracy/precision/recall/F1."""
    report_path = os.path.join(REPORTS_ROOT, dataset, "model_comparison.csv")
    if not os.path.exists(report_path):
        print(f"Skipping model comparison chart for {dataset} — run evaluate.py first.")
        return
    df = pd.read_csv(report_path).set_index("model")
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    out_dir = os.path.join(REPORTS_ROOT, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    df[metrics].plot(kind="bar", figsize=(9, 6), colormap="viridis")
    plt.title(f"{dataset} — Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"model_comparison_{dataset}.png"), dpi=150)
    plt.close()
    print(f"Model comparison chart saved to {out_dir}/model_comparison_{dataset}.png")


def plot_cross_dataset_chart():
    path = os.path.join(REPORTS_ROOT, "cross_dataset_generalization.csv")
    if not os.path.exists(path):
        print("Skipping cross-dataset chart — run cross_dataset.py first.")
        return
    df = pd.read_csv(path)
    df["pair"] = df["train_dataset"] + " → " + df["test_dataset"]
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    out_dir = os.path.join(REPORTS_ROOT, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    df.set_index("pair")[metrics].plot(kind="bar", figsize=(9, 6), colormap="plasma")
    plt.title("Cross-Dataset Generalization Performance")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_dataset_generalization.png"), dpi=150)
    plt.close()
    print(f"Cross-dataset chart saved to {out_dir}/cross_dataset_generalization.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["nslkdd", "unsw"])
    parser.add_argument("--comparison-only", action="store_true",
                         help="Only generate the cross-model/cross-dataset comparison charts.")
    args = parser.parse_args()

    if args.comparison_only:
        for ds in ["nslkdd", "unsw"]:
            plot_model_comparison(ds)
        plot_cross_dataset_chart()
    elif args.dataset:
        run_dataset_eda(args.dataset)
        plot_model_comparison(args.dataset)
    else:
        parser.error("Provide --dataset nslkdd|unsw, or use --comparison-only")