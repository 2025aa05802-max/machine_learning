"""
Train all 6 ML Classification Models on the Dry Bean Dataset
Models: Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost
Metrics: Accuracy, AUC, Precision, Recall, F1, MCC
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ============================================================
# 1. Load and Prepare Data
# ============================================================
print("=" * 60)
print("LOADING DRY BEAN DATASET")
print("=" * 60)

# Determine the correct path for the data file
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_path = os.path.join(project_dir, 'data', 'dry_bean.csv')

df = pd.read_csv(data_path)
print(f"Dataset Shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Instances: {df.shape[0]}")
print(f"\nClass Distribution:\n{df['Class'].value_counts()}")
print(f"\nMissing Values:\n{df.isnull().sum().sum()}")

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = list(le.classes_)
print(f"\nClass Names: {class_names}")

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler and label encoder for the Streamlit app
model_dir = os.path.join(project_dir, 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))

# Save test data for Streamlit app demo
test_df = pd.DataFrame(X_test, columns=X.columns)
test_df['Class'] = le.inverse_transform(y_test)
test_df.to_csv(os.path.join(project_dir, 'data', 'dry_bean_test.csv'), index=False)
print(f"Test data saved for Streamlit app demo.")

# ============================================================
# 2. Define Models
# ============================================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=2000, random_state=42, solver='lbfgs'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, max_depth=15
    ),
    'KNN': KNeighborsClassifier(
        n_neighbors=7
    ),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, random_state=42, use_label_encoder=False,
        eval_metric='mlogloss', verbosity=0
    )
}

# ============================================================
# 3. Train, Evaluate, and Save Each Model
# ============================================================
results = {}
all_metrics = []

print("\n" + "=" * 60)
print("TRAINING AND EVALUATING MODELS")
print("=" * 60)

for name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"Training: {name}")
    print(f"{'─' * 50}")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics = {
        'Accuracy': round(accuracy, 4),
        'AUC': round(auc, 4),
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1 Score': round(f1, 4),
        'MCC': round(mcc, 4)
    }

    results[name] = metrics
    all_metrics.append({'Model': name, **metrics})

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC:       {mcc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:\n{cm}")

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=class_names)
    print(f"\n  Classification Report:\n{report}")

    # Save model
    model_filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"  Model saved: {model_filename}")

# ============================================================
# 4. Comparison Table
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON TABLE")
print("=" * 60)

comparison_df = pd.DataFrame(all_metrics)
comparison_df = comparison_df.set_index('Model')
print(comparison_df.to_string())

# Save results to JSON for Streamlit app
results_path = os.path.join(model_dir, 'results.json')
with open(results_path, 'w') as f:
    json.dump({
        'metrics': results,
        'class_names': class_names,
        'feature_names': list(X.columns),
        'dataset_info': {
            'name': 'Dry Bean Dataset',
            'source': 'UCI Machine Learning Repository',
            'total_instances': int(df.shape[0]),
            'total_features': int(df.shape[1] - 1),
            'num_classes': len(class_names),
            'train_size': int(X_train.shape[0]),
            'test_size': int(X_test.shape[0])
        }
    }, f, indent=2)

print(f"\nResults saved to: {results_path}")

# ============================================================
# 5. Best Model Summary
# ============================================================
print("\n" + "=" * 60)
print("BEST MODEL BY EACH METRIC")
print("=" * 60)

for metric in ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']:
    best_model = comparison_df[metric].idxmax()
    best_value = comparison_df[metric].max()
    print(f"  {metric:12s}: {best_model} ({best_value:.4f})")

print("\n✅ All models trained and saved successfully!")
print(f"📁 Models saved in: {model_dir}")
print(f"📊 Results saved in: {results_path}")
