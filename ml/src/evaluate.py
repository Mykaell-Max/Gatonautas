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
    
    return pipeline, X, y


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


def main(args):
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
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Detailed results saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model", required=True, help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to test data")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--threshold-file", help="Path to threshold file")
    parser.add_argument("--plots", action="store_true", help="Generate evaluation plots")
    parser.add_argument("--output", help="Path to save detailed results")
    
    args = parser.parse_args()
    main(args)
