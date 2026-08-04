"""
Data Preprocessing Module
-------------------------
Loads raw NSL-KDD or UNSW-NB15 CSVs, cleans them, encodes categorical
features, scales numeric features, maps labels into 5 attack categories
(Normal, DoS, DDoS, Brute Force, Malware), and saves processed train/test
splits to data/processed/.

Usage:
    python src/preprocessing.py --dataset nslkdd
    python src/preprocessing.py --dataset unsw
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

# NSL-KDD column names (41 features + label + difficulty)
NSLKDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Maps raw attack labels -> unified 5-class scheme used across the project.
# Extend/adjust this if your dataset version has additional label spellings.
ATTACK_CATEGORY_MAP = {
    # --- Normal ---
    "normal": "Normal",
    # --- DoS ---
    "neptune": "DoS", "smurf": "DoS", "back": "DoS", "teardrop": "DoS",
    "pod": "DoS", "land": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "dos": "DoS",
    # --- DDoS (UNSW-NB15 uses this explicitly; NSL-KDD folds it into DoS) ---
    "ddos": "DDoS",
    # --- Brute Force ---
    "guess_passwd": "Brute Force", "ftp_write": "Brute Force",
    "imap": "Brute Force", "multihop": "Brute Force", "phf": "Brute Force",
    "warezmaster": "Brute Force", "warezclient": "Brute Force",
    "spy": "Brute Force", "snmpgetattack": "Brute Force",
    "snmpguess": "Brute Force", "httptunnel": "Brute Force",
    "sendmail": "Brute Force", "named": "Brute Force",
    "xlock": "Brute Force", "xsnoop": "Brute Force",
    "worm": "Brute Force", "fuzzers": "Brute Force",
    # --- Malware / probing / exploits bucket ---
    "ipsweep": "Malware", "nmap": "Malware", "portsweep": "Malware",
    "satan": "Malware", "mscan": "Malware", "saint": "Malware",
    "buffer_overflow": "Malware", "loadmodule": "Malware",
    "perl": "Malware", "rootkit": "Malware", "ps": "Malware",
    "sqlattack": "Malware", "xterm": "Malware", "shellcode": "Malware",
    "backdoor": "Malware", "backdoors": "Malware", "analysis": "Malware",
    "exploits": "Malware", "generic": "Malware", "reconnaissance": "Malware",
}


def _map_label(raw_label: str) -> str:
    key = str(raw_label).strip().lower()
    return ATTACK_CATEGORY_MAP.get(key, "Malware")  # unseen labels default to Malware bucket


def load_nslkdd():
    # NSL-KDD files live inside data/raw/nsl-kdd/ after extraction
    subdir = os.path.join(RAW_DIR, "nsl-kdd")
    train_path = os.path.join(subdir, "KDDTrain+.txt")
    test_path = os.path.join(subdir, "KDDTest+.txt")
    train = pd.read_csv(train_path, names=NSLKDD_COLUMNS)
    test = pd.read_csv(test_path, names=NSLKDD_COLUMNS)
    train.drop(columns=["difficulty"], inplace=True)
    test.drop(columns=["difficulty"], inplace=True)
    return train, test


def load_unsw():
    # UNSW-NB15 files live inside data/raw/UNSW-NB15/
    subdir = os.path.join(RAW_DIR, "UNSW-NB15")
    train_path = os.path.join(subdir, "UNSW_NB15_training-set.csv")
    test_path = os.path.join(subdir, "UNSW_NB15_testing-set.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    # UNSW uses 'attack_cat' for category and 'label' for binary normal/attack
    for df in (train, test):
        df.drop(columns=[c for c in ["id", "label"] if c in df.columns], inplace=True)
        df.rename(columns={"attack_cat": "label"}, inplace=True)
        df["label"] = df["label"].fillna("normal")
    return train, test


def preprocess(dataset: str):
    if dataset == "nslkdd":
        train, test = load_nslkdd()
    elif dataset == "unsw":
        train, test = load_unsw()
    else:
        raise ValueError("dataset must be 'nslkdd' or 'unsw'")

    train["attack_category"] = train["label"].apply(_map_label)
    test["attack_category"] = test["label"].apply(_map_label)
    train.drop(columns=["label"], inplace=True)
    test.drop(columns=["label"], inplace=True)

    # Identify categorical vs numeric columns (excluding target)
    cat_cols = [c for c in train.columns
                if train[c].dtype == "object" and c != "attack_category"]
    num_cols = [c for c in train.columns
                if c not in cat_cols + ["attack_category"]]

    # Encode categoricals (fit on train, apply to test; unseen -> 'unknown' bucket)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        le.fit(list(train[col].unique()) + ["__unseen__"])
        test[col] = test[col].apply(lambda v: v if v in le.classes_ else "__unseen__")
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        encoders[col] = le

    # Encode target label
    target_le = LabelEncoder()
    train["attack_category"] = target_le.fit_transform(train["attack_category"])
    test["attack_category"] = test["attack_category"].apply(
        lambda v: v if v in target_le.classes_ else "Malware"
    )
    test["attack_category"] = target_le.transform(test["attack_category"])

    # Scale numeric features
    scaler = StandardScaler()
    train[num_cols] = scaler.fit_transform(train[num_cols])
    test[num_cols] = scaler.transform(test[num_cols])

    # Persist processed data + preprocessing artifacts
    train.to_csv(os.path.join(PROCESSED_DIR, f"{dataset}_train.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, f"{dataset}_test.csv"), index=False)
    feature_columns = [c for c in train.columns if c != "attack_category"]
    joblib.dump(
        {"encoders": encoders, "scaler": scaler, "target_encoder": target_le,
         "cat_cols": cat_cols, "num_cols": num_cols,
         "feature_columns": feature_columns},
        os.path.join(PROCESSED_DIR, f"{dataset}_artifacts.pkl"),
    )

    print(f"[{dataset}] train shape: {train.shape}, test shape: {test.shape}")
    print(f"[{dataset}] classes: {list(target_le.classes_)}")
    return train, test


def transform_raw_dataframe(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Applies the SAME encoding/scaling used during training (loaded from
    the saved {dataset}_artifacts.pkl) to a raw, unprocessed dataframe of
    new records — used for inference on brand-new traffic samples (e.g.
    dashboard manual entry or an uploaded CSV) so they go through an
    identical pipeline to what the model was trained on.

    `df` must contain the dataset's raw feature columns (unscaled numerics,
    original text categoricals) — i.e. the same schema as the raw
    KDDTrain+/UNSW CSVs, minus the label column.
    """
    artifacts = joblib.load(os.path.join(PROCESSED_DIR, f"{dataset}_artifacts.pkl"))
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
    cat_cols = artifacts["cat_cols"]
    num_cols = artifacts["num_cols"]
    feature_columns = artifacts["feature_columns"]

    df = df.copy()
    for col in cat_cols:
        le = encoders[col]
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda v: v if v in le.classes_ else "__unseen__")
        df[col] = le.transform(df[col])

    df[num_cols] = scaler.transform(df[num_cols])

    return df[feature_columns]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nslkdd", "unsw"])
    args = parser.parse_args()
    preprocess(args.dataset)