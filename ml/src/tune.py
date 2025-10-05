#!/usr/bin/env python3

import argparse
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from train import build_pipeline, load_data
import warnings
warnings.filterwarnings('ignore')


def get_searcher(method, estimator, param_space, scoring, cv, n_iter, n_jobs):
    """Get the appropriate search object based on method."""
    if method == "grid":
        return GridSearchCV(
            estimator, param_space, 
            scoring=scoring, cv=cv, n_jobs=n_jobs, 
            refit=True, verbose=1
        )
    elif method == "random":
        return RandomizedSearchCV(
            estimator, param_space, 
            n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=n_jobs, 
            refit=True, verbose=1, random_state=42
        )
    elif method == "halving_grid":
        try:
            from sklearn.experimental import enable_halving_search_cv  # noqa
            from sklearn.model_selection import HalvingGridSearchCV
            return HalvingGridSearchCV(
                estimator, param_space, 
                scoring=scoring, cv=cv, factor=2, 
                refit=True, verbose=1
            )
        except ImportError:
            print("Warning: HalvingGridSearchCV not available, falling back to GridSearchCV")
            return GridSearchCV(
                estimator, param_space, 
                scoring=scoring, cv=cv, n_jobs=n_jobs, 
                refit=True, verbose=1
            )
    else:
        raise ValueError(f"Unknown search method: {method}")


def strip_prefix(d, prefix):
    """Remove prefix from dictionary keys."""
    out = {}
    for k, v in d.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    # Remove any keys that still have double underscores (safety check)
    return {k: v for k, v in out.items() if not k.startswith("__")}


def main(args):
    # Load configuration
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("Loading data for hyperparameter tuning...")
    X, y = load_data(cfg["data"]["train_path"], cfg["data"]["target"])
    
    # Build the same pipeline as training
    pipeline = build_pipeline(
        cfg["features"]["numeric"],
        cfg["features"]["categorical"],
        cfg["model"]["params"],
        cfg["model"]["name"]
    )
    
    # Get search configuration
    search_cfg = cfg["model"]["search"]
    method = search_cfg.get("method", "grid")
    cv = search_cfg.get("cv", 5)
    n_jobs = search_cfg.get("n_jobs", -1)
    n_iter = search_cfg.get("n_iter", 30)
    param_space = search_cfg["param_grid"]
    
    # Set up scoring
    scoring_name = search_cfg.get("scoring", "roc_auc")
    if scoring_name == "f1_pos2":
        scoring = make_scorer(f1_score, pos_label=2)
    elif scoring_name == "pr_auc":
        scoring = "average_precision"  # sklearn uses "average_precision" for PR-AUC
    elif scoring_name == "precision_pos2":
        from sklearn.metrics import precision_score
        scoring = make_scorer(precision_score, pos_label=2, zero_division=0)
    elif scoring_name == "recall_pos2":
        from sklearn.metrics import recall_score
        scoring = make_scorer(recall_score, pos_label=2, zero_division=0)
    else:
        scoring = scoring_name  # e.g., "roc_auc", "average_precision"
    
    print(f"Using scoring: {scoring_name}")
    print(f"Search method: {method}")
    print(f"CV folds: {cv}")
    print(f"Parameter combinations to try: {len(param_space) if method == 'grid' else n_iter}")
    
    # Set up cross-validation
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Create searcher
    searcher = get_searcher(method, pipeline, param_space, scoring, cv_obj, n_iter, n_jobs)
    
    print("Starting hyperparameter search...")
    searcher.fit(X, y)
    
    print(f"\nBest cross-validation score: {searcher.best_score_:.4f}")
    print("Best parameters:")
    for param, value in searcher.best_params_.items():
        print(f"  {param}: {value}")
    
    # Create artifacts directory
    artifacts_dir = Path(cfg["artifacts"]["dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model name for file naming
    model_name = cfg["model"]["name"]
    
    # Save best estimator
    best_estimator_path = f"{artifacts_dir}/{model_name}_best_estimator.joblib"
    joblib.dump(searcher.best_estimator_, best_estimator_path)
    print(f"Best estimator saved to {best_estimator_path}")
    
    # Save tuning results
    tuning_results = {
        "best_score": float(searcher.best_score_),
        "best_params": searcher.best_params_,
        "scoring": scoring_name,
        "cv": cv,
        "method": method,
        "n_iter": n_iter if method == "random" else None,
        "param_grid": param_space
    }
    
    tuning_results_path = f"{artifacts_dir}/{model_name}_tuning_results.json"
    with open(tuning_results_path, "w") as f:
        json.dump(tuning_results, f, indent=2)
    print(f"Tuning results saved to {tuning_results_path}")
    
    # Create tuned config (merge best params with original config)
    tuned_cfg = dict(cfg)
    tuned_cfg["model"] = dict(cfg["model"])
    
    # Strip classifier__ prefix and merge with original params
    best_params_clean = strip_prefix(searcher.best_params_, "classifier__")
    tuned_cfg["model"]["params"] = {**cfg["model"]["params"], **best_params_clean}
    
    # Remove the search section from tuned config
    if "search" in tuned_cfg["model"]:
        del tuned_cfg["model"]["search"]
    
    # Save tuned config
    tuned_config_path = f"{artifacts_dir}/{model_name}_tuned.yaml"
    with open(tuned_config_path, "w") as f:
        yaml.safe_dump(tuned_cfg, f, default_flow_style=False)
    print(f"Tuned config saved to {tuned_config_path}")
    
    # Show summary
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("="*50)
    print(f"Best CV Score ({scoring_name}): {searcher.best_score_:.4f}")
    print("\nTo retrain with best parameters:")
    print(f"  python src/train.py --config {tuned_config_path}")
    print("\nTo evaluate the best estimator:")
    print(f"  python src/evaluate.py --model {best_estimator_path} --data artifacts/test_set.csv --threshold-file artifacts/{model_name}_threshold.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ML models")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    main(args)
