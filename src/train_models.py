"""
Model Training Module
----------------------
Trains and tunes Random Forest, XGBoost, and a Neural Network classifier
on a processed dataset, then saves each model to models/.

Usage:
    python src/train_models.py --dataset nslkdd
"""

import argparse
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_processed(dataset):
    train = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, f"{dataset}_test.csv"))
    X_train = train.drop(columns=["attack_category"])
    y_train = train["attack_category"]
    X_test = test.drop(columns=["attack_category"])
    y_test = test["attack_category"]
    return X_train, y_train, X_test, y_test


def train_random_forest(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid, cv=3, scoring="f1_macro", n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print("Best RF params:", grid.best_params_)
    return grid.best_estimator_


def train_xgboost(X_train, y_train):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(y_train.unique()),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_net(X_train, y_train, num_classes):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=15, batch_size=256,
              validation_split=0.1, verbose=2)
    return model


def main(dataset):
    X_train, y_train, X_test, y_test = load_processed(dataset)
    num_classes = y_train.nunique()

    print("Training Random Forest...")
    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, f"{dataset}_random_forest.pkl"))

    print("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)
    joblib.dump(xgb, os.path.join(MODELS_DIR, f"{dataset}_xgboost.pkl"))

    print("Training Neural Network...")
    nn = train_neural_net(X_train, y_train, num_classes)
    nn.save(os.path.join(MODELS_DIR, f"{dataset}_neural_net.keras"))

    print(f"All models trained and saved to {MODELS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["nslkdd", "unsw"])
    args = parser.parse_args()
    main(args.dataset)
