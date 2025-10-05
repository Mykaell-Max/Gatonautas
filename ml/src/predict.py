#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


def load_model_and_schema(model_path, schema_path):
    """Load the trained model and feature schema."""
    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    
    print(f"Loading feature schema from {schema_path}")
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    return pipeline, schema


def load_threshold(threshold_path):
    """Load the optimal threshold."""
    with open(threshold_path, 'r') as f:
        threshold_data = json.load(f)
    return threshold_data["threshold"]


def prepare_data(df, schema):
    """Prepare data for prediction by ensuring correct feature order."""
    if 'row_index' in df.columns:
        df = df.drop(columns=['row_index'])
    
    if schema["target"] in df.columns:
        df = df.drop(columns=[schema["target"]])
    
    # Replace empty strings with NaN
    df = df.replace('', np.nan)
    
    # Convert numeric columns to numeric, coercing errors to NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinity values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    required_features = schema["numeric_features"] + schema["categorical_features"]
    missing_features = set(required_features) - set(df.columns)
    
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        for feature in missing_features:
            df[feature] = np.nan
    
    available_features = [f for f in required_features if f in df.columns]
    df = df[available_features]
    
    return df


def predict(pipeline, X, threshold=0.5, return_proba=True):
    """Make predictions using the trained model."""
    
    # Get probability for class 2 (exoplanet)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]  # Class 2 is at index 1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred = y_pred + 1  # Convert 0,1 to 1,2
    
    if return_proba:
        return y_pred, y_pred_proba
    else:
        return y_pred


def main(args):
    pipeline, schema = load_model_and_schema(args.model, args.schema)
    
    threshold = args.threshold
    if args.threshold_file:
        threshold = load_threshold(args.threshold_file)
        print(f"Using threshold from file: {threshold}")
    
    print(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    X = prepare_data(df, schema)
    print(f"Data shape: {X.shape}")
    
    print("Making predictions...")
    y_pred, y_pred_proba = predict(pipeline, X, threshold, return_proba=True)
    
    results_df = df.copy()
    results_df['prediction'] = y_pred
    results_df['probability'] = y_pred_proba
    results_df['prediction_label'] = results_df['prediction'].map({1: 'Non-Exoplanet', 2: 'Exoplanet'})
    
    print(f"\nPrediction Summary:")
    print(f"Total samples: {len(results_df)}")
    print(f"Predicted exoplanets: {(y_pred == 2).sum()}")
    print(f"Predicted non-exoplanets: {(y_pred == 1).sum()}")
    print(f"Average probability: {y_pred_proba.mean():.4f}")
    
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    
    if args.verbose:
        print("\nDetailed predictions:")
        for i, (idx, row) in enumerate(results_df.iterrows()):
            if i < 10:
                print(f"Sample {idx}: {row['prediction_label']} (prob: {row['probability']:.4f})")
            else:
                break
    
    if args.probabilities_only:
        prob_df = pd.DataFrame({
            'probability': y_pred_proba,
            'prediction': y_pred
        })
        prob_output = args.output.replace('.csv', '_probabilities.csv') if args.output else 'probabilities.csv'
        prob_df.to_csv(prob_output, index=False)
        print(f"Probabilities saved to {prob_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using trained model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--schema", required=True, help="Path to feature schema")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", help="Path to save predictions")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--threshold-file", help="Path to threshold file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed predictions")
    parser.add_argument("--probabilities-only", action="store_true", help="Save only probabilities")
    
    args = parser.parse_args()
    main(args)
