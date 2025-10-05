#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_stacking_model(meta_learner_path, meta_learner_info_path):
    """Load the stacking model components."""
    meta_learner = joblib.load(meta_learner_path)
    
    with open(meta_learner_info_path, 'r') as f:
        meta_info = json.load(f)
    
    return meta_learner, meta_info


def get_base_models():
    """Get the base models for stacking."""
    return {
        'gb': {
            'model_path': 'artifacts/gb_model_pipeline.joblib',
        },
        'lgbm': {
            'model_path': 'artifacts/lgbm_model_pipeline.joblib',
        },
        'rf': {
            'model_path': 'artifacts/rf_model_pipeline.joblib',
        }
    }


def prepare_data(data_path, feature_schema_path):
    """Prepare data for prediction."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Load feature schema
    with open(feature_schema_path, 'r') as f:
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
        # Add missing features with NaN values
        for feature in missing_features:
            df[feature] = np.nan
    
    X = df[all_features]
    
    print(f"Data shape: {X.shape}")
    print(f"Missing values: {X.isnull().sum().sum()}")
    
    return X


def predict_stacking(X, meta_learner, meta_info, base_models):
    """Make predictions using the stacking model."""
    print("Generating base model predictions...")
    
    # Generate meta-features
    meta_features = np.zeros((X.shape[0], len(base_models)))
    
    for i, (model_name, model_info) in enumerate(base_models.items()):
        print(f"Loading {model_name} model...")
        model = joblib.load(model_info['model_path'])
        
        # Get predictions
        predictions = model.predict_proba(X)[:, 1]
        meta_features[:, i] = predictions
    
    # Get meta-learner predictions
    print("Making meta-learner predictions...")
    meta_predictions = meta_learner.predict_proba(meta_features)[:, 1]
    meta_pred_binary = (meta_predictions >= 0.5).astype(int)
    meta_pred_binary = meta_pred_binary + 1  # Convert 0,1 to 1,2
    
    return meta_predictions, meta_pred_binary


def main(args):
    # Load stacking model
    meta_learner, meta_info = load_stacking_model(
        args.meta_learner, 
        args.meta_info
    )
    
    # Get base models
    base_models = get_base_models()
    
    # Prepare data
    X = prepare_data(args.input, args.schema)
    
    # Make predictions
    probabilities, predictions = predict_stacking(X, meta_learner, meta_info, base_models)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'prediction': predictions,
        'probability': probabilities
    })
    
    # Add base model predictions
    for i, model_name in enumerate(meta_info['base_models']):
        model = joblib.load(base_models[model_name]['model_path'])
        base_predictions = model.predict_proba(X)[:, 1]
        results[f'{model_name}_probability'] = base_predictions
    
    # Save results
    results.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    
    # Print summary
    print("\n=== Prediction Summary ===")
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted exoplanets: {(predictions == 2).sum()}")
    print(f"Predicted non-exoplanets: {(predictions == 1).sum()}")
    print(f"Average probability: {probabilities.mean():.4f}")
    
    # Show base model agreement
    print("\n=== Base Model Agreement ===")
    for model_name in meta_info['base_models']:
        base_probs = results[f'{model_name}_probability']
        agreement = np.corrcoef(probabilities, base_probs)[0, 1]
        print(f"{model_name} correlation: {agreement:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using stacking model")
    parser.add_argument("--meta-learner", required=True, help="Path to meta-learner model")
    parser.add_argument("--meta-info", required=True, help="Path to meta-learner info")
    parser.add_argument("--schema", required=True, help="Path to feature schema")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", required=True, help="Path to output predictions")
    
    args = parser.parse_args()
    main(args)
