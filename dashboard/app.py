"""
Dashboard Module
------------------
Streamlit-based interface for classifying network traffic samples.

- Automatically selects and loads the best-performing model for the chosen
  dataset (by macro-F1 score from src/evaluate.py's saved comparison
  report) — no manual model picking.
- Two ways to provide traffic data for classification:
    1. Upload a CSV of raw traffic records
    2. Enter a single record's values manually (a real analyst use case:
       checking one suspicious connection by hand)
- Shows the predicted class, confidence, and a SHAP explanation of which
  features drove the prediction (tree models only — see note below).

Usage:
    streamlit run dashboard/app.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.preprocessing import transform_raw_dataframe
from src.explainability import per_class_shap_list

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

st.set_page_config(page_title="AI-Powered NIDS Dashboard", layout="wide")
st.title("🛡️ AI-Powered Network Intrusion Detection System")
st.caption("Classify network traffic and view SHAP-based explanations, "
           "using the best-performing model for the selected dataset.")

# --- Sidebar: dataset selection only — model is chosen automatically ---
st.sidebar.header("Configuration")
dataset = st.sidebar.selectbox("Dataset", ["nslkdd", "unsw"])


@st.cache_resource
def load_artifacts(dataset):
    return joblib.load(os.path.join(PROCESSED_DIR, f"{dataset}_artifacts.pkl"))


@st.cache_data
def get_best_model_name(dataset):
    """Reads src/evaluate.py's saved comparison report and returns the
    model with the highest macro-F1 score — this is what makes model
    selection automatic instead of a manual dropdown."""
    report_path = os.path.join(REPORTS_DIR, f"{dataset}_model_comparison.csv")
    if not os.path.exists(report_path):
        return None, None
    df = pd.read_csv(report_path).sort_values("f1_macro", ascending=False)
    best = df.iloc[0]
    return best["model"], df


@st.cache_resource
def load_model(dataset, model_name):
    if model_name == "neural_net":
        from tensorflow import keras
        path = os.path.join(MODELS_DIR, f"{dataset}_neural_net.keras")
        return keras.models.load_model(path)
    path = os.path.join(MODELS_DIR, f"{dataset}_{model_name}.pkl")
    return joblib.load(path)


def predict_proba(model, model_name, X):
    """Unifies prediction across sklearn/XGBoost models (.predict_proba)
    and the Keras neural net (.predict returns softmax probabilities
    directly, with no separate predict_proba method)."""
    if model_name == "neural_net":
        return model.predict(X, verbose=0)
    return model.predict_proba(X)


try:
    artifacts = load_artifacts(dataset)
    best_model_name, comparison_df = get_best_model_name(dataset)
    class_names = artifacts["target_encoder"].classes_
    cat_cols = artifacts["cat_cols"]
    num_cols = artifacts["num_cols"]
    feature_columns = artifacts["feature_columns"]
    encoders = artifacts["encoders"]
    scaler = artifacts["scaler"]
except FileNotFoundError:
    st.warning(
        f"Processed data not found for '{dataset}'. Run:\n\n"
        f"```\npython src/preprocessing.py --dataset {dataset}\n```"
    )
    st.stop()

if best_model_name is None:
    st.warning(
        "No evaluation report found yet, so the best model can't be "
        f"determined automatically. Run:\n\n"
        f"```\npython src/train_models.py --dataset {dataset}\n"
        f"python src/evaluate.py --dataset {dataset}\n```"
    )
    st.stop()

model = load_model(dataset, best_model_name)

best_row = comparison_df.iloc[0]
st.sidebar.success(
    f"**Auto-selected model:** {best_model_name}\n\n"
    f"F1 (macro): {best_row['f1_macro']:.3f}\n\n"
    f"Accuracy: {best_row['accuracy']:.3f}"
)
with st.sidebar.expander("Full model comparison"):
    st.dataframe(comparison_df.set_index("model"))

# --- Input mode ---
st.subheader("1. Provide a traffic record")
input_mode = st.radio(
    "Input method",
    ["Upload CSV", "Enter values manually"],
    horizontal=True,
)

raw_record = None  # single-row raw DataFrame, before encoding/scaling

if input_mode == "Upload CSV":
    st.caption(
        f"CSV must contain these {len(feature_columns)} raw columns "
        "(unscaled numbers, original category names — same schema as the "
        "raw dataset, no label column):"
    )
    st.code(", ".join(feature_columns), language=None)

    template = pd.DataFrame([{
        **{c: [v for v in encoders[c].classes_ if v != "__unseen__"][0]
           for c in cat_cols},
        **{c: float(scaler.mean_[num_cols.index(c)]) for c in num_cols},
    }])[feature_columns]
    st.download_button(
        "Download CSV template (with example values)",
        template.to_csv(index=False),
        file_name=f"{dataset}_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded is not None:
        uploaded_df = pd.read_csv(uploaded)
        missing = [c for c in feature_columns if c not in uploaded_df.columns]
        if missing:
            st.error(f"Uploaded CSV is missing required columns: {missing}")
            st.stop()
        row_idx = 0
        if len(uploaded_df) > 1:
            row_idx = st.number_input(
                "Row to classify", 0, len(uploaded_df) - 1, 0
            )
        raw_record = uploaded_df.iloc[[row_idx]][feature_columns]

else:  # Enter values manually
    st.caption(
        "Enter a single traffic record's raw values. Numeric fields default "
        "to the training-set average; categorical fields default to a "
        "common value — adjust whatever you actually want to test."
    )
    with st.form("manual_entry_form"):
        values = {}
        col_a, col_b = st.columns(2)
        for i, col in enumerate(feature_columns):
            target_col = col_a if i % 2 == 0 else col_b
            if col in cat_cols:
                options = [c for c in encoders[col].classes_ if c != "__unseen__"]
                values[col] = target_col.selectbox(col, options)
            else:
                default_val = float(scaler.mean_[num_cols.index(col)])
                values[col] = target_col.number_input(
                    col, value=round(default_val, 4)
                )
        submitted = st.form_submit_button("Classify this record")
    if submitted:
        raw_record = pd.DataFrame([values])[feature_columns]

if raw_record is None:
    st.info("Upload a CSV or submit the manual entry form above to classify a record.")
    st.stop()

st.write("Raw input record:")
st.dataframe(raw_record)

# --- Transform using the SAME pipeline the model was trained on ---
processed_record = transform_raw_dataframe(raw_record, dataset)

# --- Prediction ---
st.subheader("2. Prediction")
proba = predict_proba(model, best_model_name, processed_record)[0]
pred_idx = int(np.argmax(proba))
pred_label = class_names[pred_idx]

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Predicted Class", pred_label)
    st.caption(f"Model used: **{best_model_name}**")
with col2:
    proba_df = pd.DataFrame({"class": class_names, "probability": proba}).sort_values(
        "probability", ascending=False
    )
    st.bar_chart(proba_df.set_index("class"))

# --- Explainability ---
st.subheader("3. Why did the model predict this?")
if best_model_name == "neural_net":
    st.info(
        "SHAP explanation is only available for the tree-based models "
        "(Random Forest / XGBoost) via the fast TreeExplainer used in this "
        "project. The auto-selected best model is the neural network, so "
        "no explanation is shown here — this is expected, not an error."
    )
else:
    with st.spinner("Computing SHAP explanation..."):
        explainer = shap.TreeExplainer(model)
        raw_shap_values = explainer.shap_values(processed_record)
        shap_list = per_class_shap_list(
            raw_shap_values, n_features=processed_record.shape[1]
        )
        class_shap = shap_list[pred_idx][0]
        contributions = (
            pd.Series(class_shap, index=processed_record.columns)
            .sort_values(key=abs, ascending=False)
            .head(10)
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contributions.values]
    ax.barh(contributions.index[::-1], contributions.values[::-1], color=colors[::-1])
    ax.set_xlabel("SHAP value (impact on prediction)")
    ax.set_title(f"Top features driving '{pred_label}' classification")
    st.pyplot(fig)

    st.caption(
        "Red bars push the prediction toward the predicted class; "
        "blue bars push away from it."
    )