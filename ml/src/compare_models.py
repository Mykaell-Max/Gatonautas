#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')


def load_model_and_predict(model_path, X, threshold_file=None):
    """Load model and make predictions."""
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X)[:, 1]
    
    # Use optimal threshold if provided, otherwise use default 0.5
    if threshold_file and Path(threshold_file).exists():
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
        threshold = threshold_data['threshold']
    else:
        threshold = 0.5
    
    # Convert probabilities to binary predictions using optimal threshold
    predictions = (probabilities >= threshold).astype(int) + 1  # Convert 0,1 to 1,2
    
    return predictions, probabilities


def evaluate_model(y_true, y_pred, y_prob, model_name):
    """Evaluate a single model."""
    # Convert predictions to binary for ROC-AUC
    y_true_binary = (y_true == 2).astype(int)
    
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=2, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=2, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_binary, y_prob))
    }
    
    return metrics


def load_stacking_predictions(meta_learner_path, meta_info_path, X):
    """Load stacking model and make predictions."""
    meta_learner = joblib.load(meta_learner_path)
    
    with open(meta_info_path, 'r') as f:
        meta_info = json.load(f)
    
    # Get base models
    base_models = {
        'gb': {'model_path': 'artifacts/gb_model_pipeline.joblib'},
        'lgbm': {'model_path': 'artifacts/lgbm_model_pipeline.joblib'},
        'rf': {'model_path': 'artifacts/rf_model_pipeline.joblib'}
    }
    
    # Generate meta-features
    meta_features = np.zeros((X.shape[0], len(base_models)))
    
    for i, (model_name, model_info) in enumerate(base_models.items()):
        model = joblib.load(model_info['model_path'])
        predictions = model.predict_proba(X)[:, 1]
        meta_features[:, i] = predictions
    
    # Get meta-learner predictions
    meta_predictions = meta_learner.predict_proba(meta_features)[:, 1]
    meta_pred_binary = (meta_predictions >= 0.5).astype(int)
    meta_pred_binary = meta_pred_binary + 1  # Convert 0,1 to 1,2
    
    return meta_pred_binary, meta_predictions


def prepare_data(data_path, schema_path):
    """Prepare data for evaluation."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Load feature schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Clean data
    df = df.replace('', np.nan)
    
    # Convert numeric columns
    for col in schema['numeric_features']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Select features
    all_features = schema['numeric_features'] + schema['categorical_features']
    missing_features = [f for f in all_features if f not in df.columns]
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            df[feature] = np.nan
    
    X = df[all_features]
    y = df['label'] if 'label' in df.columns else None
    
    print(f"Data shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    if y is not None:
        print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def main(args):
    # Prepare data
    X, y_true = prepare_data(args.data, args.schema)
    
    if y_true is None:
        print("Error: No target column 'label' found in data")
        return
    
    results = []
    
    # Evaluate individual models
    models = [
        ("Gradient Boosting", "artifacts/gb_model_pipeline.joblib", "artifacts/gb_threshold.json"),
        ("LightGBM", "artifacts/lgbm_model_pipeline.joblib", "artifacts/lgbm_threshold.json"),
        ("Random Forest", "artifacts/rf_model_pipeline.joblib", "artifacts/rf_threshold.json")
    ]
    
    for model_name, model_path, threshold_path in models:
        if Path(model_path).exists():
            print(f"\nEvaluating {model_name}...")
            y_pred, y_prob = load_model_and_predict(model_path, X, threshold_path)
            metrics = evaluate_model(y_true, y_pred, y_prob, model_name)
            results.append(metrics)
        else:
            print(f"Warning: {model_name} model not found at {model_path}")
    
    # Evaluate stacking model
    if Path("artifacts/meta_learner.joblib").exists():
        print(f"\nEvaluating Stacking Model...")
        y_pred_stack, y_prob_stack = load_stacking_predictions(
            "artifacts/meta_learner.joblib",
            "artifacts/meta_learner_info.json",
            X
        )
        metrics = evaluate_model(y_true, y_pred_stack, y_prob_stack, "Stacking")
        results.append(metrics)
    else:
        print("Warning: Stacking model not found")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Print comparison
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    # Sort by F1 score
    results_df = results_df.sort_values('f1', ascending=False)
    
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Save results
    results_df.to_csv("model_comparison.csv", index=False)
    print(f"\nResults saved to model_comparison.csv")
    
    # Print detailed classification reports
    print("\n" + "="*80)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*80)
    
    for _, row in results_df.iterrows():
        model_name = row['model']
        print(f"\n{model_name}:")
        print("-" * 40)
        
        if model_name == "Stacking":
            y_pred, y_prob = load_stacking_predictions(
                "artifacts/meta_learner.joblib",
                "artifacts/meta_learner_info.json",
                X
            )
        else:
            model_info = next((name, path, threshold) for name, path, threshold in models if name == model_name)
            model_path = model_info[1]
            threshold_path = model_info[2]
            y_pred, y_prob = load_model_and_predict(model_path, X, threshold_path)
        
        print(classification_report(y_true, y_pred, target_names=['Non-Exoplanet', 'Exoplanet']))
        print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare all trained models")
    parser.add_argument("--data", default="data/processed/kepler_lc/test.csv", help="Path to test data")
    parser.add_argument("--schema", default="artifacts/gb_feature_schema.json", help="Path to feature schema")
    
    args = parser.parse_args()
    main(args)
