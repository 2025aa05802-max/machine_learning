"""
BITS ID : 2025AA05802
Dry Bean Classification - Streamlit Web Application
=====================================================
Interactive ML Classification Dashboard for the Dry Bean Dataset.
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# ─── Page Configuration ──────────────────────────────────────
st.set_page_config(
    page_title="Dry Bean Classification Dashboard",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# ─── Load Artefacts ───────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    """Load all saved models, scaler, label encoder, and results."""
    with open(os.path.join(MODEL_DIR, 'results.json'), 'r') as f:
        results = json.load(f)

    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))

    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'KNN': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }

    models = {}
    for name, fname in model_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    return results, scaler, label_encoder, models

results, scaler, label_encoder, models = load_artefacts()

# ─── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("BITS ID : 2025AA05802")
st.sidebar.title("Dry Bean Classifier")
st.sidebar.markdown("---")

# ── Dataset Upload ────────────────────────────────────────────
st.sidebar.header("Upload Dataset")
st.sidebar.caption("Upload a CSV file with test data (same features as the Dry Bean dataset). "
                    "The last column should be the class label named **Class**.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# ── Model Selection ───────────────────────────────────────────
st.sidebar.header("Choose Model")
model_names = list(models.keys())
selected_model = st.sidebar.selectbox("Choose a classification model", model_names)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Assignment 2** — ML (BITS MTech AI/ML)  \n"
    "Dataset: [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)"
)

# ─── Main Header ──────────────────────────────────────────────
st.title("Dry Bean Classification Dashboard")
st.markdown(
    "An interactive dashboard to explore **6 ML classification models** trained on the "
    "[Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) "
    "from the UCI Machine Learning Repository."
)

# ─── Tabs ─────────────────────────────────────────────────────
tab_overview, tab_predict, tab_compare = st.tabs([
    "Dataset & Model Overview",
    "Predict & Evaluate",
    "Model Comparison"
])

# ======================================================================
# TAB 1 — Dataset & Model Overview
# ======================================================================
with tab_overview:
    st.header("Dataset Information")
    info = results['dataset_info']
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Instances", f"{info['total_instances']:,}")
    col2.metric("Features", info['total_features'])
    col3.metric("Classes", info['num_classes'])
    col4.metric("Train / Test", f"{info['train_size']:,} / {info['test_size']:,}")

    st.markdown(f"**Source:** {info['source']}  ")
    st.markdown(f"**Classes:** {', '.join(results['class_names'])}")
    st.markdown(f"**Features:** {', '.join(results['feature_names'])}")

    st.markdown("---")
    st.header("Model Performance Overview")

    metrics_df = pd.DataFrame(results['metrics']).T
    metrics_df.index.name = 'Model'

    # Highlight best value per column
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: #d4edda; font-weight: bold' if v else '' for v in is_max]

    st.dataframe(
        metrics_df.style.apply(highlight_max, axis=0).format("{:.4f}"),
        use_container_width=True
    )

# ======================================================================
# TAB 2 — Predict & Evaluate
# ======================================================================
with tab_predict:

    # Resolve the dataframe to use
    if uploaded_file is not None:
        df_eval = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file loaded — {df_eval.shape[0]} rows, {df_eval.shape[1]} columns")
    else:
        default_test_path = os.path.join(DATA_DIR, 'dry_bean_test.csv')
        if os.path.exists(default_test_path):
            df_eval = pd.read_csv(default_test_path)
            st.info("No file uploaded. Using the built-in test dataset (2 723 samples).")
        else:
            st.warning("Please upload a CSV file from the sidebar.")
            st.stop()

    # Show a preview
    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df_eval.head(20), use_container_width=True)

    # Separate features & labels
    feature_names = results['feature_names']

    if 'Class' in df_eval.columns:
        has_labels = True
        X_eval = df_eval[feature_names]
        y_true_labels = df_eval['Class']
        # Encode
        le = label_encoder
        y_true = le.transform(y_true_labels)
    else:
        has_labels = False
        X_eval = df_eval[feature_names]

    # Scale
    X_eval_scaled = scaler.transform(X_eval)

    # Predict
    model = models[selected_model]
    y_pred = model.predict(X_eval_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    st.markdown("---")
    st.subheader(f"Predictions using **{selected_model}**")

    # If labels are available → show evaluation metrics
    if has_labels:
        y_pred_proba = model.predict_proba(X_eval_scaled)

        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        # ── Metrics Cards ────────────────────────────────────
        st.markdown("#### Evaluation Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{acc:.4f}")
        m2.metric("AUC", f"{auc:.4f}")
        m3.metric("Precision", f"{prec:.4f}")
        m4.metric("Recall", f"{rec:.4f}")
        m5.metric("F1 Score", f"{f1:.4f}")
        m6.metric("MCC", f"{mcc:.4f}")

        st.markdown("---")

        # ── Confusion Matrix & Classification Report ─────────
        col_cm, col_cr = st.columns(2)

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=results['class_names'],
                yticklabels=results['class_names'],
                ax=ax_cm
            )
            ax_cm.set_xlabel('Predicted Label')
            ax_cm.set_ylabel('True Label')
            ax_cm.set_title(f'Confusion Matrix — {selected_model}')
            plt.tight_layout()
            st.pyplot(fig_cm)

        with col_cr:
            st.markdown("#### Classification Report")
            report = classification_report(
                y_true, y_pred,
                target_names=results['class_names'],
                output_dict=True
            )
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

    # ── Predictions Table ────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Prediction Results")
    pred_df = df_eval.copy()
    pred_df['Predicted Class'] = y_pred_labels
    if has_labels:
        pred_df['Correct'] = pred_df['Class'] == pred_df['Predicted Class']
    st.dataframe(pred_df.head(100), use_container_width=True)

# ======================================================================
# TAB 3 — Model Comparison
# ======================================================================
with tab_compare:
    st.header("Model Comparison")

    metrics_df = pd.DataFrame(results['metrics']).T
    metrics_df.index.name = 'Model'

    # ── Comparison Table ─────────────────────────────────────
    st.subheader("Comparison Table")
    st.dataframe(
        metrics_df.style.apply(highlight_max, axis=0).format("{:.4f}"),
        use_container_width=True
    )

    # ── Bar Charts ───────────────────────────────────────────
    st.subheader("Metric-wise Comparison")

    metric_choice = st.selectbox(
        "Select metric to compare",
        ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    )

    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("viridis", n_colors=len(metrics_df))
    bars = ax_bar.bar(metrics_df.index, metrics_df[metric_choice], color=colors)
    ax_bar.set_ylabel(metric_choice)
    ax_bar.set_title(f'{metric_choice} Comparison Across Models')
    ax_bar.set_ylim(min(metrics_df[metric_choice]) - 0.02, max(metrics_df[metric_choice]) + 0.02)

    for bar, val in zip(bars, metrics_df[metric_choice]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()
    st.pyplot(fig_bar)

    # ── Radar Chart ──────────────────────────────────────────
    st.subheader("Radar Chart — All Metrics")

    categories = list(metrics_df.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig_radar, ax_radar = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    color_palette = sns.color_palette("tab10", n_colors=len(metrics_df))

    for idx, (model_name, row) in enumerate(metrics_df.iterrows()):
        values = row.tolist()
        values += values[:1]
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color_palette[idx])
        ax_radar.fill(angles, values, alpha=0.05, color=color_palette[idx])

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10)
    ax_radar.set_ylim(0.85, 1.0)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax_radar.set_title("Model Performance Radar Chart", y=1.08, fontsize=14)
    plt.tight_layout()
    st.pyplot(fig_radar)

    # ── Best model per metric ────────────────────────────────
    st.subheader("Best Model by Each Metric")
    best_data = []
    for metric in metrics_df.columns:
        best_model = metrics_df[metric].idxmax()
        best_val = metrics_df[metric].max()
        best_data.append({'Metric': metric, 'Best Model': best_model, 'Score': f"{best_val:.4f}"})
    st.table(pd.DataFrame(best_data))

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "ML Assignment 2 — BITS Pilani MTech AI/ML | "
    "Dry Bean Classification Dashboard"
    "</div>",
    unsafe_allow_html=True
)
