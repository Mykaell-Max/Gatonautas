#!/usr/bin/env python3

import argparse
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data(model_path, data_path, target_col):
    """Load the trained model and test data."""
    print(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    
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
    
    # Handle NaN values - fill with median for numeric columns, or 0 if all values are NaN
    print(f"Handling NaN values in the data...")
    print(f"NaN values before handling: {X.isnull().sum().sum()}")
    
    # Fill NaN values with median for numeric columns
    numeric_columns = X.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            # If median is NaN (all values are NaN), use 0 instead
            if pd.isna(median_val):
                median_val = 0.0
                print(f"Column '{col}' has all NaN values, using 0 as fill value")
            X[col] = X[col].fillna(median_val)
            print(f"Filled {X[col].isnull().sum()} NaN values in column '{col}' with value: {median_val}")
    
    print(f"NaN values after handling: {X.isnull().sum().sum()}")
    
    # If this is a stacking model, we need to get predictions from base models first
    if "meta_learner" in model_path:
        print("Detected stacking model - getting predictions from base models...")
        X = get_stacking_features(X)
    
    return pipeline, X, y


def get_stacking_features(X):
    """Get predictions from base models for stacking."""
    base_models = {
        'gb': './artifacts/gb_model_pipeline.joblib',
        'lgbm': './artifacts/lgbm_model_pipeline.joblib', 
        'rf': './artifacts/rf_model_pipeline.joblib'
    }
    
    stacking_features = []
    
    for model_name, model_path in base_models.items():
        try:
            print(f"Loading {model_name} model...")
            base_model = joblib.load(model_path)
            
            # Get prediction probabilities for class 2 (exoplanet)
            pred_proba = base_model.predict_proba(X)[:, 1]
            stacking_features.append(pred_proba)
            print(f"Got {model_name} predictions: {len(pred_proba)} samples")
            
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
            # If we can't load a base model, fill with zeros
            stacking_features.append(np.zeros(len(X)))
    
    # Create DataFrame with base model predictions
    stacking_df = pd.DataFrame({
        'gb_prediction': stacking_features[0],
        'lgbm_prediction': stacking_features[1], 
        'rf_prediction': stacking_features[2]
    })
    
    print(f"Created stacking features with shape: {stacking_df.shape}")
    return stacking_df


def get_exoplanet_confidence(pipeline, X, threshold=0.5):
    """
    Get the percentage confidence that the model has for exoplanet predictions.
    
    Args:
        pipeline: Trained model pipeline
        X: Input features
        threshold: Classification threshold (default 0.5)
    
    Returns:
        dict: Contains confidence percentages and predictions
    """
    # Get probability for class 2 (exoplanet)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]  # Class 2 is at index 1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred = y_pred + 1  # Convert 0,1 to 1,2
    
    # Calculate confidence percentages
    exoplanet_confidence = y_pred_proba * 100  # Convert to percentage
    non_exoplanet_confidence = (1 - y_pred_proba) * 100  # Convert to percentage
    
    results = []
    for i in range(len(X)):
        result = {
            "sample_index": i,
            "prediction": int(y_pred[i]),
            "prediction_label": "Exoplanet" if y_pred[i] == 2 else "Non-Exoplanet",
            "exoplanet_confidence": float(exoplanet_confidence[i]),
            "non_exoplanet_confidence": float(non_exoplanet_confidence[i]),
            "threshold_used": float(threshold)
        }
        results.append(result)
    
    return results


def evaluate_model(pipeline, X, y, threshold=0.5):
    """Evaluate the model and return comprehensive metrics."""
    
    # Get probability for class 2 (exoplanet)
    y_pred_proba = pipeline.predict_proba(X)[:, 1]  # Class 2 is at index 1
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred = y_pred + 1  # Convert 0,1 to 1,2
    
    # Convert y to binary for ROC-AUC (1->0, 2->1)
    y_binary = (y == 2).astype(int)
    
    metrics = {
        "accuracy": float((y_pred == y).mean()),
        "precision": float(precision_score(y, y_pred, pos_label=2, zero_division=0)),
        "recall": float(recall_score(y, y_pred, pos_label=2, zero_division=0)),
        "f1": float(f1_score(y, y_pred, pos_label=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_binary, y_pred_proba)),
        "threshold": float(threshold)
    }
    
    return metrics, y_pred_proba, y_pred


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Exoplanet', 'Exoplanet'],
                yticklabels=['Non-Exoplanet', 'Exoplanet'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot and save ROC curve."""
    # Convert y_true to binary for ROC curve
    y_true_binary = (y_true == 2).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
    roc_auc = roc_auc_score(y_true_binary, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """Plot and save Precision-Recall curve."""
    # Convert y_true to binary for precision-recall curve
    y_true_binary = (y_true == 2).astype(int)
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    
    plt.show()


def evaluate_model_api(model_path, data_path, target_col="label", threshold=0.5, threshold_file=None, include_confidence=False):
    """
    API-friendly function to evaluate model and return structured results.
    
    Args:
        model_path (str): Path to the trained model
        data_path (str): Path to the test data
        target_col (str): Target column name
        threshold (float): Classification threshold
        threshold_file (str): Path to threshold file (optional)
        include_confidence (bool): Whether to include confidence analysis
    
    Returns:
        dict: Structured evaluation results
    """
    pipeline, X, y = load_model_and_data(model_path, data_path, target_col)
    
    # Use threshold from file if provided
    if threshold_file:
        with open(threshold_file, 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data["threshold"]
    
    # Get evaluation metrics
    metrics, y_pred_proba, y_pred = evaluate_model(pipeline, X, y, threshold)
    
    # Get classification report as dict
    class_report = classification_report(y, y_pred, 
                                       target_names=['Non-Exoplanet', 'Exoplanet'],
                                       output_dict=True)
    
    # Prepare results
    results = {
        "data_info": {
            "shape": X.shape,
            "target_distribution": y.value_counts().to_dict(),
            "threshold_used": threshold
        },
        "metrics": metrics,
        "classification_report": class_report,
        "predictions": {
            "probabilities": y_pred_proba.tolist(),
            "predictions": y_pred.tolist(),
            "true_labels": y.tolist()
        }
    }
    
    # Add confidence analysis if requested
    if include_confidence:
        confidence_results = get_exoplanet_confidence(pipeline, X, threshold)
        
        # Calculate confidence summary statistics
        exoplanet_confidences = [r["exoplanet_confidence"] for r in confidence_results]
        non_exoplanet_confidences = [r["non_exoplanet_confidence"] for r in confidence_results]
        
        confidence_summary = {
            "average_exoplanet_confidence": float(np.mean(exoplanet_confidences)),
            "average_non_exoplanet_confidence": float(np.mean(non_exoplanet_confidences)),
            "min_exoplanet_confidence": float(np.min(exoplanet_confidences)),
            "max_exoplanet_confidence": float(np.max(exoplanet_confidences)),
            "total_samples": len(confidence_results)
        }
        
        results["confidence_analysis"] = {
            "summary": confidence_summary,
            "detailed_results": confidence_results
        }
    
    return results


def predict_single_sample_api(model_path, sample_data, threshold=0.5):
    """
    API-friendly function to predict a single sample and return confidence.
    
    Args:
        model_path (str): Path to the trained model
        sample_data (dict or pd.Series): Single data sample
        threshold (float): Classification threshold
    
    Returns:
        dict: Prediction results with confidence
    """
    # Load model
    pipeline = joblib.load(model_path)
    
    # Convert to DataFrame if needed
    if isinstance(sample_data, dict):
        df = pd.DataFrame([sample_data])
    elif isinstance(sample_data, pd.Series):
        df = pd.DataFrame([sample_data])
    else:
        raise ValueError("sample_data must be a dictionary or pandas Series")
    
    # Handle NaN values
    df = df.replace('', np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
    
    # If this is a stacking model, get predictions from base models
    if "meta_learner" in model_path:
        df = get_stacking_features(df)
    
    # Get prediction
    y_pred_proba = pipeline.predict_proba(df)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    y_pred = y_pred + 1  # Convert 0,1 to 1,2
    
    # Calculate confidence percentages
    exoplanet_confidence = y_pred_proba[0] * 100
    non_exoplanet_confidence = (1 - y_pred_proba[0]) * 100
    
    result = {
        "prediction": int(y_pred[0]),
        "prediction_label": "Exoplanet" if y_pred[0] == 2 else "Non-Exoplanet",
        "exoplanet_confidence": float(exoplanet_confidence),
        "non_exoplanet_confidence": float(non_exoplanet_confidence),
        "threshold_used": float(threshold),
        "probability": float(y_pred_proba[0])
    }
    
    return result


def main(args):
    """Main function for command line usage (keeps printing for CLI)."""
    pipeline, X, y = load_model_and_data(args.model, args.data, args.target)
    
    print(f"Data shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    threshold = args.threshold
    if args.threshold_file:
        with open(args.threshold_file, 'r') as f:
            threshold_data = json.load(f)
            threshold = threshold_data["threshold"]
            print(f"Using threshold from file: {threshold}")
    
    print(f"Evaluating model with threshold: {threshold}")
    metrics, y_pred_proba, y_pred = evaluate_model(pipeline, X, y, threshold)
    
    print("\n=== Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n=== Classification Report ===")
    print(classification_report(y, y_pred, 
                              target_names=['Non-Exoplanet', 'Exoplanet']))
    
    # Generate confidence analysis if requested
    if args.confidence:
        print("\n=== Confidence Analysis ===")
        confidence_results = get_exoplanet_confidence(pipeline, X, threshold)
        
        # Show summary statistics
        exoplanet_confidences = [r["exoplanet_confidence"] for r in confidence_results]
        non_exoplanet_confidences = [r["non_exoplanet_confidence"] for r in confidence_results]
        
        print(f"Average exoplanet confidence: {np.mean(exoplanet_confidences):.2f}%")
        print(f"Average non-exoplanet confidence: {np.mean(non_exoplanet_confidences):.2f}%")
        print(f"Min exoplanet confidence: {np.min(exoplanet_confidences):.2f}%")
        print(f"Max exoplanet confidence: {np.max(exoplanet_confidences):.2f}%")
        
        # Show detailed results for first few samples
        print(f"\nDetailed confidence for first {min(5, len(confidence_results))} samples:")
        for i, result in enumerate(confidence_results[:5]):
            print(f"Sample {i}: {result['prediction_label']} "
                  f"(Exoplanet: {result['exoplanet_confidence']:.2f}%, "
                  f"Non-Exoplanet: {result['non_exoplanet_confidence']:.2f}%)")
    
    if args.plots:
        print("\nGenerating plots...")
        
        import os
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_confusion_matrix(y, y_pred, 
                            save_path=f"{plots_dir}/confusion_matrix.png")
        
        plot_roc_curve(y, y_pred_proba, 
                      save_path=f"{plots_dir}/roc_curve.png")
        
        plot_precision_recall_curve(y, y_pred_proba, 
                                  save_path=f"{plots_dir}/precision_recall_curve.png")
    
    if args.output:
        results = {
            "metrics": metrics,
            "predictions": {
                "probabilities": y_pred_proba.tolist(),
                "predictions": y_pred.tolist(),
                "true_labels": y.tolist()
            }
        }
        
        # Include confidence results if requested
        if args.confidence:
            results["confidence_analysis"] = get_exoplanet_confidence(pipeline, X, threshold)
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model (defaults to stacking model)")
    parser.add_argument("--model", default="./artifacts/meta_learner.joblib", 
                       help="Path to trained model (default: stacking model)")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--threshold-file", help="Path to threshold file")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--confidence", action="store_true", help="Show confidence analysis for exoplanet predictions")
    parser.add_argument("--output", help="Path to save detailed results")
    
    args = parser.parse_args()
    
    # Print information about the model being used
    if args.model == "./artifacts/meta_learner.joblib":
        print("Using stacking model (meta_learner.joblib) - the main model that combines multiple base models")
    else:
        print(f"Using custom model: {args.model}")
    
    main(args)
