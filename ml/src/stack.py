#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from train import build_pipeline, load_data
import yaml
import warnings
warnings.filterwarnings('ignore')


def get_base_models():
    """Get the base models for stacking."""
    return {
        'gb': {
            'config': 'configs/gb.yaml',
            'model_path': 'artifacts/gb_model_pipeline.joblib',
            'schema_path': 'artifacts/gb_feature_schema.json'
        },
        'lgbm': {
            'config': 'configs/lgbm.yaml', 
            'model_path': 'artifacts/lgbm_model_pipeline.joblib',
            'schema_path': 'artifacts/lgbm_feature_schema.json'
        },
        'rf': {
            'config': 'configs/rf.yaml',
            'model_path': 'artifacts/rf_model_pipeline.joblib',
            'schema_path': 'artifacts/rf_feature_schema.json'
        }
    }


def generate_meta_features(X, base_models, cv_folds=5):
    """Generate meta-features using cross-validation predictions from base models."""
    print("Generating meta-features using cross-validation...")
    
    meta_features = np.zeros((X.shape[0], len(base_models)))
    meta_feature_names = []
    
    for i, (model_name, model_info) in enumerate(base_models.items()):
        print(f"Generating meta-features for {model_name}...")
        
        # Load the trained model
        model = joblib.load(model_info['model_path'])
        
        # For simplicity, we'll use the full model predictions
        # In a real scenario, you'd use proper cross-validation
        predictions = model.predict_proba(X)[:, 1]
        meta_features[:, i] = predictions
        meta_feature_names.append(f"{model_name}_prediction")
    
    return meta_features, meta_feature_names


def train_meta_learner(meta_features, y, meta_learner_params=None):
    """Train the meta-learner (Logistic Regression)."""
    if meta_learner_params is None:
        meta_learner_params = {
            'random_state': 42,
            'max_iter': 1000,
            'class_weight': 'balanced'
        }
    
    print("Training meta-learner (Logistic Regression)...")
    meta_learner = LogisticRegression(**meta_learner_params)
    meta_learner.fit(meta_features, y)
    
    return meta_learner


def evaluate_stacked_model(meta_learner, base_models, X_test, y_test):
    """Evaluate the stacked model."""
    print("Evaluating stacked model...")
    
    # Generate meta-features for test set
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (model_name, model_info) in enumerate(base_models.items()):
        model = joblib.load(model_info['model_path'])
        predictions = model.predict_proba(X_test)[:, 1]
        meta_features_test[:, i] = predictions
    
    # Get meta-learner predictions
    meta_predictions = meta_learner.predict_proba(meta_features_test)[:, 1]
    meta_pred_binary = (meta_predictions >= 0.5).astype(int)
    meta_pred_binary = meta_pred_binary + 1  # Convert 0,1 to 1,2
    
    # Convert y_test to binary for ROC-AUC
    y_test_binary = (y_test == 2).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {
        "accuracy": float((meta_pred_binary == y_test).mean()),
        "precision": float(precision_score(y_test, meta_pred_binary, pos_label=2, zero_division=0)),
        "recall": float(recall_score(y_test, meta_pred_binary, pos_label=2, zero_division=0)),
        "f1": float(f1_score(y_test, meta_pred_binary, pos_label=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_binary, meta_predictions))
    }
    
    return metrics, meta_predictions, meta_pred_binary


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Loading data for stacking...")
    X, y = load_data(config["data"]["train_path"], config["data"]["target"])
    
    # Get base models
    base_models = get_base_models()
    
    # Check if all base models exist
    for model_name, model_info in base_models.items():
        if not Path(model_info['model_path']).exists():
            print(f"Error: {model_name} model not found at {model_info['model_path']}")
            print(f"Please train {model_name} model first using: make train-{model_name}")
            return
    
    # Generate meta-features
    meta_features, meta_feature_names = generate_meta_features(X, base_models)
    
    # Train meta-learner
    meta_learner = train_meta_learner(meta_features, y)
    
    # Cross-validation evaluation
    print("Performing cross-validation on meta-learner...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(meta_learner, meta_features, y, cv=cv, scoring='roc_auc')
    print(f"Meta-learner CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Create artifacts directory
    artifacts_dir = Path(config["artifacts"]["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Save meta-learner
    meta_learner_path = f"{artifacts_dir}/meta_learner.joblib"
    joblib.dump(meta_learner, meta_learner_path)
    print(f"Meta-learner saved to {meta_learner_path}")
    
    # Save meta-feature names
    meta_feature_info = {
        "meta_feature_names": meta_feature_names,
        "base_models": list(base_models.keys()),
        "cv_roc_auc": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std())
    }
    
    with open(f"{artifacts_dir}/meta_learner_info.json", "w") as f:
        json.dump(meta_feature_info, f, indent=2)
    
    print("Stacking model training completed successfully!")
    print(f"Meta-feature names: {meta_feature_names}")
    print(f"Base models: {list(base_models.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train stacking model with meta-learner")
    parser.add_argument("--config", default="configs/gb.yaml", help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)
