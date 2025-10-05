#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, f1_score, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import yaml
import warnings
warnings.filterwarnings('ignore')

# Consistent random state from trains.py
RDS = 68


def build_pipeline(numeric_features, categorical_features, model_params, model_name="gradient_boosting", sampling_config=None):
    """Build the preprocessing and model pipeline with imbalanced data handling."""
    
    from sklearn.impute import SimpleImputer
    
    # Only Random Forest and Gradient Boosting don't need scaling
    # LightGBM can benefit from scaling in some cases
    if model_name.lower() in ["gradient_boosting", "random_forest"]:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
    else:
        num_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_features),
        ],
        remainder="drop",
    )
    
    # Choose model based on name
    if model_name.lower() == "lightgbm":
        model = lgb.LGBMClassifier(**model_params)
    elif model_name.lower() == "gradient_boosting":
        model = GradientBoostingClassifier(**model_params)
    elif model_name.lower() == "random_forest":
        model = RandomForestClassifier(**model_params)
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: 'gradient_boosting', 'lightgbm', 'random_forest'")
    
    # Use imbalanced pipeline if sampling config is provided (from trains.py approach)
    if sampling_config and sampling_config.get("use_sampling", False):
        smote_strategy = sampling_config.get("smote_strategy", 0.25)
        under_strategy = sampling_config.get("under_strategy", 0.75)
        sampling_random_state = sampling_config.get("random_state", RDS)
        
        pipeline = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(sampling_strategy=smote_strategy, random_state=sampling_random_state)),
            ("under", RandomUnderSampler(sampling_strategy=under_strategy, random_state=sampling_random_state)),
            ("classifier", model)
        ])
    else:
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    
    return pipeline


def load_data(data_path, target_col):
    """Load and prepare the data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    if 'row_index' in df.columns:
        df = df.drop(columns=['row_index'])
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Convert numeric columns to numeric, coercing errors to NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Missing values per column:\n{X.isnull().sum().sum()} total missing values")
    
    return X, y


def evaluate_model(pipeline, X_test, y_test, threshold=0.5):
    """Evaluate the model and return metrics."""
    
    # Get probability for class 2 (exoplanet)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Class 2 is at index 1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred = y_pred + 1  # Convert 0,1 to 1,2
    
    # Convert y_test to binary for ROC-AUC (1->0, 2->1)
    y_test_binary = (y_test == 2).astype(int)
    
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "precision": float(precision_score(y_test, y_pred, pos_label=2, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=2, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test_binary, y_pred_proba)),
        "threshold": float(threshold)
    }
    
    return metrics, y_pred_proba, y_pred


def find_optimal_threshold(y_true, y_pred_proba, metric="f1"):
    """Find optimal threshold based on specified metric."""
    # Convert y_true to binary for threshold optimization
    y_true_binary = (y_true == 2).astype(int)
    
    precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_proba)
    
    if metric == "f1":
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5
    
    return optimal_threshold


def analyze_feature_importance(pipeline, feature_names, model_name="classifier"):
    """Analyze and display feature importance (from trains.py)."""
    try:
        # Get the classifier from the pipeline
        if hasattr(pipeline, 'named_steps'):
            classifier = pipeline.named_steps[model_name]
        else:
            classifier = pipeline
        
        # Check if the model has feature_importances_
        if hasattr(classifier, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': classifier.feature_importances_
            }).sort_values(by='importance', ascending=False)
            
            print(f"\n=== Feature Importance Analysis ===")
            print(f"Top 10 Most Important Features:")
            print(feature_importance.head(10))
            
            return feature_importance
        else:
            print(f"Model {model_name} does not support feature importance analysis")
            return None
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")
        return None


def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the full dataset
    X, y = load_data(config["data"]["train_path"], config["data"]["target"])
    
    # Use consistent random state from trains.py
    random_state = config["evaluation"].get("random_state", RDS)
    
    # Use a larger test set for better evaluation (30% instead of 20%)
    test_size = 0.3 if config["evaluation"]["test_size"] < 0.3 else config["evaluation"]["test_size"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    pipeline = build_pipeline(
        config["features"]["numeric"],
        config["features"]["categorical"],
        config["model"]["params"],
        config["model"]["name"],
        config.get("sampling", None)
    )
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    print("Performing cross-validation...")
    # Use StratifiedKFold like in trains.py
    cv = StratifiedKFold(n_splits=config["evaluation"]["cv_folds"], shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, 
        cv=cv,
        scoring='roc_auc'
    )
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("Evaluating on test set...")
    metrics, y_pred_proba, y_pred = evaluate_model(pipeline, X_test, y_test)
    
    optimal_threshold = find_optimal_threshold(
        y_test, y_pred_proba, 
        config["evaluation"]["threshold_metric"]
    )
    
    metrics_optimal, _, _ = evaluate_model(pipeline, X_test, y_test, optimal_threshold)
    metrics_optimal["optimal_threshold"] = float(optimal_threshold)
    
    print("\n=== Test Set Results ===")
    for metric, value in metrics_optimal.items():
        print(f"{metric}: {value:.4f}")
    
    # Analyze feature importance (from trains.py)
    all_features = config["features"]["numeric"] + config["features"]["categorical"]
    feature_importance = analyze_feature_importance(pipeline, all_features, "classifier")
    
    artifacts_dir = Path(config["artifacts"]["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(pipeline, config["artifacts"]["model_path"])
    print(f"Model saved to {config['artifacts']['model_path']}")
    
    with open(config["artifacts"]["metrics_path"], 'w') as f:
        json.dump(metrics_optimal, f, indent=2)
    
    feature_schema = {
        "numeric_features": config["features"]["numeric"],
        "categorical_features": config["features"]["categorical"],
        "target": config["data"]["target"]
    }
    with open(config["artifacts"]["feature_schema_path"], 'w') as f:
        json.dump(feature_schema, f, indent=2)
    
    with open(config["artifacts"]["threshold_path"], 'w') as f:
        json.dump({"threshold": float(optimal_threshold)}, f, indent=2)
    
    # Save feature importance if available
    if feature_importance is not None:
        feature_importance.to_csv("artifacts/feature_importance.csv", index=False)
        print("Feature importance saved to artifacts/feature_importance.csv")
    
    # Save test set for evaluation
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv("artifacts/test_set.csv", index=False)
    print("Test set saved to artifacts/test_set.csv")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gradient Boosting model for exoplanet detection")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)
