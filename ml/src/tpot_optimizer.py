
import argparse
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# Disable dask distributed computing
os.environ['TPOT_DASK_CLIENT'] = 'False'


def load_data(data_path, target_col):
    """Load and prepare the data."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Clean data
    df = df.replace('', np.nan)
    
    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=[object]).columns
    for col in numeric_cols:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    
    print(f"Data shape: {df.shape}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    return df


def evaluate_model(y_true, y_pred, y_prob=None):
    """Evaluate model performance with multiple metrics."""
    # Convert predictions to binary for metrics (1=non-exoplanet, 2=exoplanet)
    y_true_binary = (y_true == 2).astype(int)
    y_pred_binary = (y_pred == 2).astype(int)
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
        "recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
        "f1": float(f1_score(y_true_binary, y_pred_binary, zero_division=0))
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true_binary, y_prob))
    
    return metrics


def apply_resampling(X, y, method='none', random_state=42):
    """Apply resampling technique to handle class imbalance."""
    if method == 'none':
        return X, y
    
    # Handle missing values before resampling
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    if method == 'smote':
        print("Applying SMOTE resampling...")
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_imputed, y)
        print(f"After SMOTE: {X_resampled.shape[0]} samples")
        return X_resampled, y_resampled
    elif method == 'undersample':
        print("Applying random undersampling...")
        undersampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = undersampler.fit_resample(X_imputed, y)
        print(f"After undersampling: {X_resampled.shape[0]} samples")
        return X_resampled, y_resampled
    elif method == 'smoteenn':
        print("Applying SMOTEENN resampling...")
        smoteenn = SMOTEENN(random_state=random_state)
        X_resampled, y_resampled = smoteenn.fit_resample(X_imputed, y)
        print(f"After SMOTEENN: {X_resampled.shape[0]} samples")
        return X_resampled, y_resampled
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def run_tpot_optimization(X, y, n_runs=10, generations=5, population_size=20, cv=5, 
                         random_state=42, scoring='f1', resampling='none'):
    """Run TPOT optimization multiple times."""
    results = []
    
    print(f"Running TPOT optimization for {n_runs} iterations...")
    print(f"Parameters: generations={generations}, population_size={population_size}, cv={cv}")
    print(f"Scoring metric: {scoring}")
    print(f"Resampling method: {resampling}")
    print("="*80)
    
    for i in range(n_runs):
        print(f"\n--- Run {i+1}/{n_runs} ---")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.75, test_size=0.25, random_state=random_state + i
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Apply resampling to training data only
        X_train_resampled, y_train_resampled = apply_resampling(
            X_train, y_train, method=resampling, random_state=random_state + i
        )
        
        # Create custom config with balanced class weights
        custom_config = {
            'sklearn.ensemble.RandomForestClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 10, None],
                'class_weight': ['balanced'],
                'random_state': [random_state + i]
            },
            'sklearn.ensemble.ExtraTreesClassifier': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 10, None],
                'class_weight': ['balanced'],
                'random_state': [random_state + i]
            },
            'sklearn.linear_model.LogisticRegression': {
                'C': [0.1, 1.0, 10.0],
                'class_weight': ['balanced'],
                'random_state': [random_state + i]
            },
            'sklearn.svm.SVC': {
                'C': [0.1, 1.0, 10.0],
                'class_weight': ['balanced'],
                'random_state': [random_state + i]
            }
        }
        
        # Create TPOT classifier with custom config for balanced class weights
        pipeline_optimizer = TPOTClassifier(
            generations=generations,
            population_size=population_size,
            cv=cv,
            random_state=random_state + i,
            n_jobs=1,  # Use single job to avoid distributed computing issues
            search_space='linear-light',  # Use lighter search space for faster runs
            verbose=4
        )
        
        # Train the model on resampled data
        print("Training TPOT pipeline...")
        pipeline_optimizer.fit(X_train_resampled, y_train_resampled)
        
        # Handle missing values in test data the same way as training data
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_test_imputed = imputer.fit_transform(X_test)
        
        # Make predictions
        y_pred = pipeline_optimizer.predict(X_test_imputed)
        
        # Get probabilities if available
        try:
            y_prob = pipeline_optimizer.predict_proba(X_test_imputed)[:, 1]
        except:
            y_prob = None
        
        # Evaluate performance
        metrics = evaluate_model(y_test, y_pred, y_prob)
        
        # Store results
        result = {
            "run": i + 1,
            "pipeline": str(pipeline_optimizer.fitted_pipeline_),
            "metrics": metrics
        }
        results.append(result)
        
        # Print results
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"Pipeline {i+1}:")
        print(pipeline_optimizer.fitted_pipeline_)
        print("-" * 80)
    
    return results


def analyze_results(results):
    """Analyze and summarize the results."""
    print("\n" + "="*80)
    print("TPOT OPTIMIZATION RESULTS SUMMARY")
    print("="*80)
    
    # Extract metrics
    accuracies = [r['metrics']['accuracy'] for r in results]
    precisions = [r['metrics']['precision'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    f1_scores = [r['metrics']['f1'] for r in results]
    roc_aucs = [r['metrics'].get('roc_auc', 0) for r in results if 'roc_auc' in r['metrics']]
    
    # Calculate statistics
    metrics_stats = {
        'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies), 'max': np.max(accuracies)},
        'precision': {'mean': np.mean(precisions), 'std': np.std(precisions), 'max': np.max(precisions)},
        'recall': {'mean': np.mean(recalls), 'std': np.std(recalls), 'max': np.max(recalls)},
        'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'max': np.max(f1_scores)}
    }
    
    if roc_aucs:
        metrics_stats['roc_auc'] = {'mean': np.mean(roc_aucs), 'std': np.std(roc_aucs), 'max': np.max(roc_aucs)}
    
    # Print summary
    for metric, stats in metrics_stats.items():
        print(f"{metric.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
    
    # Find best runs
    best_f1_idx = np.argmax(f1_scores)
    best_accuracy_idx = np.argmax(accuracies)
    
    print(f"\nBest F1 Score: Run {best_f1_idx + 1} (F1 = {f1_scores[best_f1_idx]:.4f})")
    print(f"Best Accuracy: Run {best_accuracy_idx + 1} (Accuracy = {accuracies[best_accuracy_idx]:.4f})")
    
    # Show best pipelines
    print(f"\nBest F1 Pipeline:")
    print(results[best_f1_idx]['pipeline'])
    
    return metrics_stats, results[best_f1_idx], results[best_accuracy_idx]


def main(args):
    # Load data
    df = load_data(args.data, args.target)
    
    # Prepare features and target
    X = df.drop(columns=[args.target]).values
    y = df[args.target].values
    
    # Run TPOT optimization
    results = run_tpot_optimization(
        X, y, 
        n_runs=args.runs,
        generations=args.generations,
        population_size=args.population_size,
        cv=args.cv,
        random_state=args.random_state,
        scoring=args.scoring,
        resampling=args.resampling
    )
    
    # Analyze results
    metrics_stats, best_f1_run, best_accuracy_run = analyze_results(results)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        with open(output_path, 'w') as f:
            json.dump({
                'summary': metrics_stats,
                'best_f1_run': best_f1_run,
                'best_accuracy_run': best_accuracy_run,
                'all_results': results
            }, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    # Save best pipeline
    if args.save_best:
        best_pipeline_path = Path(args.save_best)
        best_pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the best F1 pipeline
        with open(best_pipeline_path, 'w') as f:
            json.dump({
                'pipeline': best_f1_run['pipeline'],
                'metrics': best_f1_run['metrics']
            }, f, indent=2)
        
        print(f"Best pipeline saved to {best_pipeline_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TPOT optimization for exoplanet detection")
    parser.add_argument("--data", default="data/processed/kepler_lc/train.csv", help="Path to training data")
    parser.add_argument("--target", default="label", help="Target column name")
    parser.add_argument("--runs", type=int, default=5, help="Number of TPOT runs")
    parser.add_argument("--generations", type=int, default=5, help="Number of generations")
    parser.add_argument("--population-size", type=int, default=20, help="Population size")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--scoring", default="f1", choices=["accuracy", "f1", "precision", "recall", "roc_auc"], 
                       help="Scoring metric for TPOT optimization")
    parser.add_argument("--resampling", default="none", choices=["none", "smote", "undersample", "smoteenn"],
                       help="Resampling method to handle class imbalance")
    parser.add_argument("--output", help="Path to save detailed results (JSON)")
    parser.add_argument("--save-best", help="Path to save best pipeline (JSON)")
    
    args = parser.parse_args()
    main(args)
