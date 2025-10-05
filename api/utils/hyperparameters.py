#!/usr/bin/env python3
"""
Hyperparameter management utilities for the Exoplanet Detection API.
Handles loading, saving, and updating hyperparameters for ML models.
"""

import os
import yaml
from typing import Dict, Any, Optional
import copy

# Default hyperparameters for each model
DEFAULT_HYPERPARAMETERS = {
    "rf": {
        "bootstrap": True,
        "class_weight": None,
        "max_depth": 20,
        "max_features": 0.5,
        "max_samples": 0.8,
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "n_estimators": 400,
        "n_jobs": -1,
        "random_state": 42
    },
    "gb": {
        "learning_rate": 0.15,
        "max_depth": 3,
        "max_features": 0.8,
        "min_samples_leaf": 10,
        "min_samples_split": 10,
        "n_estimators": 350,
        "n_iter_no_change": 40,
        "random_state": 42,
        "subsample": 0.75,
        "tol": 0.0001,
        "validation_fraction": 0.2
    },
    "lgbm": {
        "random_state": 42,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.3,
        "reg_alpha": 0.0,
        "reg_lambda": 0.5,
        "class_weight": None,
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbose": -1
    }
}

class HyperparameterManager:
    """Manages hyperparameters for ML models."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the hyperparameter manager.
        
        Args:
            config_path: Path to the YAML file storing hyperparameters
        """
        if config_path is None:
            # Default to api directory
            api_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(api_dir, "hyperparameters.yaml")
        
        self.config_path = config_path
        self._hyperparameters = self._load_hyperparameters()
    
    def _load_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
        """Load hyperparameters from YAML file or create default."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Could not load hyperparameters from {self.config_path}: {e}")
                return {}
        else:
            # Create default file
            self._save_hyperparameters(DEFAULT_HYPERPARAMETERS)
            return copy.deepcopy(DEFAULT_HYPERPARAMETERS)
    
    def _save_hyperparameters(self, hyperparameters: Dict[str, Dict[str, Any]]) -> None:
        """Save hyperparameters to YAML file."""
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(hyperparameters, f, default_flow_style=False, indent=2)
        except Exception as e:
            print(f"Error saving hyperparameters: {e}")
            raise
    
    def get_hyperparameters(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get hyperparameters for a specific model or all models.
        
        Args:
            model_name: Name of the model ('rf', 'gb', 'lgbm') or None for all
            
        Returns:
            Dictionary of hyperparameters
        """
        if model_name is None:
            return copy.deepcopy(self._hyperparameters)
        
        if model_name not in self._hyperparameters:
            if model_name in DEFAULT_HYPERPARAMETERS:
                # Initialize with defaults
                self._hyperparameters[model_name] = copy.deepcopy(DEFAULT_HYPERPARAMETERS[model_name])
                self._save_hyperparameters(self._hyperparameters)
            else:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_HYPERPARAMETERS.keys())}")
        
        return copy.deepcopy(self._hyperparameters[model_name])
    
    def update_hyperparameters(self, model_name: str, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model ('rf', 'gb', 'lgbm')
            new_params: Dictionary of new hyperparameters to update
            
        Returns:
            Updated hyperparameters for the model
        """
        if model_name not in DEFAULT_HYPERPARAMETERS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_HYPERPARAMETERS.keys())}")
        
        # Initialize model if not exists
        if model_name not in self._hyperparameters:
            self._hyperparameters[model_name] = copy.deepcopy(DEFAULT_HYPERPARAMETERS[model_name])
        
        # Update with new parameters
        self._hyperparameters[model_name].update(new_params)
        
        # Save to file
        self._save_hyperparameters(self._hyperparameters)
        
        return copy.deepcopy(self._hyperparameters[model_name])
    
    def reset_hyperparameters(self, model_name: str = None) -> Dict[str, Any]:
        """
        Reset hyperparameters to defaults.
        
        Args:
            model_name: Name of the model to reset, or None for all models
            
        Returns:
            Reset hyperparameters
        """
        if model_name is None:
            self._hyperparameters = copy.deepcopy(DEFAULT_HYPERPARAMETERS)
        else:
            if model_name not in DEFAULT_HYPERPARAMETERS:
                raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_HYPERPARAMETERS.keys())}")
            self._hyperparameters[model_name] = copy.deepcopy(DEFAULT_HYPERPARAMETERS[model_name])
        
        self._save_hyperparameters(self._hyperparameters)
        return self.get_hyperparameters(model_name)
    
    def get_available_models(self) -> list:
        """Get list of available model names."""
        return list(DEFAULT_HYPERPARAMETERS.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model's hyperparameters.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with model info including current and default hyperparameters
        """
        if model_name not in DEFAULT_HYPERPARAMETERS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(DEFAULT_HYPERPARAMETERS.keys())}")
        
        current_params = self.get_hyperparameters(model_name)
        default_params = DEFAULT_HYPERPARAMETERS[model_name]
        
        return {
            "model_name": model_name,
            "current_hyperparameters": current_params,
            "default_hyperparameters": default_params,
            "available_parameters": list(default_params.keys())
        }

# Global instance
hyperparameter_manager = HyperparameterManager()
