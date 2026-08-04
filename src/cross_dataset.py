"""
Cross-Dataset Generalization Module
------------------------------------
Trains a model on one benchmark dataset and tests it on another to
measure robustness to unseen traffic distributions — this is the
project's core research contribution (see synopsis section 8).

Methodology (important for reproducibility in your report):
1. NSL-KDD and UNSW-NB15 use different column names for equivalent
   features (e.g. duration/dur, src_bytes/sbytes, protocol_type/proto).
   Columns are mapped to shared canonical names via COLUMN_RENAME_MAP.
2. Numeric features were scaled independently per dataset during
   preprocessing.py. Using those scaled values directly across datasets
   would silently mix two different scales. So this script first
   inverse-transforms each dataset back to RAW values (using the scaler
   saved in each dataset's artifacts.pkl), then fits ONE shared
   StandardScaler on the training dataset only and applies it to both.
3. Categorical features (protocol, service) are similarly decoded back
   to their original text labels (not dataset-specific integer codes),
   normalized onto a shared vocabulary (see SERVICE_SYNONYMS), and then
   one-hot encoded with a shared OneHotEncoder fit on the training
   dataset only (unseen categories at test time map to an all-zero row
   via handle_unknown="ignore" — the statistically correct way to
   handle categories the model never saw during training).

Usage:
    python src/cross_dataset.py --train nslkdd --test unsw
    python src/cross_dataset.py --train unsw --test nslkdd
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Maps each dataset's raw column name -> a shared canonical name.
COLUMN_RENAME_MAP = {
    "nslkdd": {
        "duration": "duration", "src_bytes": "src_bytes", "dst_bytes": "dst_bytes",
        "protocol_type": "protocol", "service": "service",
    },
    "unsw": {
        "dur": "duration", "sbytes": "src_bytes", "dbytes": "dst_bytes",
        "proto": "protocol", "service": "service",
    },
}

NUMERIC_COMMON_FEATURES = ["duration", "src_bytes", "dst_bytes"]
CATEGORICAL_COMMON_FEATURES = ["protocol", "service"]
COMMON_FEATURES = NUMERIC_COMMON_FEATURES + CATEGORICAL_COMMON_FEATURES

# NSL-KDD and UNSW-NB15 spell some service names differently even though
# they mean the same thing. Normalize known synonyms onto one shared token.
SERVICE_SYNONYMS = {
    "ftp-data": "ftp_data", "ftp_data": "ftp_data",
    "domain_u": "dns", "domain": "dns", "dns": "dns",
    "pop_3": "pop3", "pop3": "pop3",
}

# Only these services are meaningfully shared/comparable across both
# datasets. UNSW-NB15's protocol field alone has 130+ distinct raw values
# (many exotic/rare, e.g. "sctp", "unas") and NSL-KDD's service field has
# ~70 mostly dataset-specific values — one-hot encoding the full raw
# vocabulary (tried first) added 70-140+ mostly-empty columns that the
# model partly overfit to on the training set rather than learning
# transferable signal. Bucketing anything outside this shared whitelist
# into "other" keeps only the categories that genuinely exist in both
# datasets, which is what makes cross-dataset comparison meaningful.
KNOWN_SERVICES = {
    "http", "ftp", "ftp_data", "smtp", "ssh", "dns",
    "pop3", "telnet", "ssl", "irc", "snmp", "dhcp",
}
KNOWN_PROTOCOLS = {"tcp", "udp", "icmp"}


def _normalize_service(value) -> str:
    v = str(value).strip().lower().replace("-", "_")
    v = SERVICE_SYNONYMS.get(v, v)
    return v if v in KNOWN_SERVICES else "other"


def _normalize_protocol(value) -> str:
    v = str(value).strip().lower()
    return v if v in KNOWN_PROTOCOLS else "other"


def load_raw_common_features(dataset, split):
    """Loads processed data, undoes BOTH the numeric scaling and categorical
    label-encoding applied in preprocessing.py, so we get back real-world
    values that can be fairly compared/re-encoded across datasets."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_{split}.csv"))
    artifacts = joblib.load(os.path.join(PROCESSED_DIR, f"{dataset}_artifacts.pkl"))
    scaler = artifacts["scaler"]
    num_cols = artifacts["num_cols"]
    cat_encoders = artifacts["encoders"]

    # Undo numeric scaling
    df[num_cols] = scaler.inverse_transform(df[num_cols])

    # Undo categorical label-encoding -> back to original text labels
    for col, encoder in cat_encoders.items():
        if col in df.columns:
            df[col] = encoder.inverse_transform(df[col].astype(int))

    df = df.rename(columns=COLUMN_RENAME_MAP[dataset])

    # Normalize categorical vocabulary onto a shared representation
    df["protocol"] = df["protocol"].apply(_normalize_protocol)
    df["service"] = df["service"].apply(_normalize_service)

    return df[COMMON_FEATURES + ["attack_category"]]


def main(train_dataset, test_dataset):
    train_df = load_raw_common_features(train_dataset, "train")
    test_df = load_raw_common_features(test_dataset, "test")

    y_train = train_df["attack_category"]
    y_test = test_df["attack_category"]

    # --- Numeric features: fit ONE shared scaler on the training set only ---
    shared_scaler = StandardScaler()
    X_train_num = shared_scaler.fit_transform(train_df[NUMERIC_COMMON_FEATURES])
    X_test_num = shared_scaler.transform(test_df[NUMERIC_COMMON_FEATURES])

    # --- Categorical features: fit ONE shared one-hot encoder on train only ---
    # handle_unknown="ignore" -> categories never seen during training
    # (e.g. a UNSW protocol that doesn't exist in NSL-KDD) become an
    # all-zero row instead of crashing or being silently mismatched.
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_train_cat = encoder.fit_transform(train_df[CATEGORICAL_COMMON_FEATURES])
    X_test_cat = encoder.transform(test_df[CATEGORICAL_COMMON_FEATURES])

    X_train = np.hstack([X_train_num, X_train_cat])
    X_test = np.hstack([X_test_num, X_test_cat])

    print(f"Feature space: {NUMERIC_COMMON_FEATURES} + "
          f"{X_train_cat.shape[1]} one-hot category columns "
          f"= {X_train.shape[1]} total features")

    model = XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        objective="multi:softprob", num_class=y_train.nunique(),
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\nTrain on: {train_dataset} | Test on: {test_dataset}")
    print(f"Accuracy: {acc:.4f} | Precision(macro): {prec:.4f} | "
          f"Recall(macro): {rec:.4f} | F1(macro): {f1:.4f}\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model + encoders together (needed to reproduce this exact
    # feature space later, e.g. for SHAP or the dashboard)
    bundle_path = os.path.join(
        MODELS_DIR, f"crossgen_{train_dataset}_to_{test_dataset}.pkl",
    )
    joblib.dump(
        {"model": model, "scaler": shared_scaler, "encoder": encoder,
         "numeric_features": NUMERIC_COMMON_FEATURES,
         "categorical_features": CATEGORICAL_COMMON_FEATURES},
        bundle_path,
    )

    result_row = pd.DataFrame([{
        "train_dataset": train_dataset, "test_dataset": test_dataset,
        "accuracy": acc, "precision_macro": prec,
        "recall_macro": rec, "f1_macro": f1,
    }])
    out_path = os.path.join(REPORTS_DIR, "cross_dataset_generalization.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Replace any prior row for this exact train/test pair rather than
        # appending duplicates when re-running.
        existing = existing[
            ~((existing["train_dataset"] == train_dataset) &
              (existing["test_dataset"] == test_dataset))
        ]
        result_row = pd.concat([existing, result_row], ignore_index=True)
    result_row.to_csv(out_path, index=False)
    print(f"Result saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, choices=["nslkdd", "unsw"])
    parser.add_argument("--test", required=True, choices=["nslkdd", "unsw"])
    args = parser.parse_args()
    main(args.train, args.test)