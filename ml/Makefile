.PHONY: help install train evaluate predict clean

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

train-gb: ## Train the Gradient Boosting model
	python src/train.py --config configs/gb.yaml

train-lgbm: ## Train the LightGBM model
	python src/train.py --config configs/lgbm.yaml

train-rf: ## Train the Random Forest model
	python src/train.py --config configs/rf.yaml

tune-gb: ## Hyperparameter tuning for Gradient Boosting
	python src/tune.py --config configs/gb.yaml

tune-lgbm: ## Hyperparameter tuning for LightGBM
	python src/tune.py --config configs/lgbm.yaml

tune-rf: ## Hyperparameter tuning for Random Forest
	python src/tune.py --config configs/rf.yaml

train-tuned-gb: ## Train with best parameters from tuning (GB)
	python src/train.py --config artifacts/gb_tuned.yaml

train-tuned-lgbm: ## Train with best parameters from tuning (LightGBM)
	python src/train.py --config artifacts/lgbm_tuned.yaml

train-tuned-rf: ## Train with best parameters from tuning (Random Forest)
	python src/train.py --config artifacts/rf_tuned.yaml

stack: ## Train stacking model with meta-learner
	python src/stack.py --config configs/stack.yaml

evaluate-gb: ## Evaluate the trained Gradient Boosting model
	python src/evaluate.py \
		--model artifacts/gb_model_pipeline.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/gb_threshold.json \
		--plots

evaluate-lgbm: ## Evaluate the trained LightGBM model
	python src/evaluate.py \
		--model artifacts/lgbm_model_pipeline.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/lgbm_threshold.json \
		--plots

evaluate-rf: ## Evaluate the trained Random Forest model
	python src/evaluate.py \
		--model artifacts/rf_model_pipeline.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/rf_threshold.json \
		--plots

evaluate-tuned-gb: ## Evaluate the best estimator from tuning (GB)
	python src/evaluate.py \
		--model artifacts/gb_best_estimator.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/gb_threshold.json \
		--plots

evaluate-tuned-lgbm: ## Evaluate the best estimator from tuning (LightGBM)
	python src/evaluate.py \
		--model artifacts/lgbm_best_estimator.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/lgbm_threshold.json \
		--plots

evaluate-tuned-rf: ## Evaluate the best estimator from tuning (Random Forest)
	python src/evaluate.py \
		--model artifacts/rf_best_estimator.joblib \
		--data artifacts/test_set.csv \
		--threshold-file artifacts/rf_threshold.json \
		--plots

predict-gb: ## Make predictions on test data (Gradient Boosting)
	python src/predict.py \
		--model artifacts/gb_model_pipeline.joblib \
		--schema artifacts/gb_feature_schema.json \
		--input data/processed/kepler_lc/test.csv \
		--output predictions_gb.csv \
		--threshold-file artifacts/gb_threshold.json

predict-lgbm: ## Make predictions on test data (LightGBM)
	python src/predict.py \
		--model artifacts/lgbm_model_pipeline.joblib \
		--schema artifacts/lgbm_feature_schema.json \
		--input data/processed/kepler_lc/test.csv \
		--output predictions_lgbm.csv \
		--threshold-file artifacts/lgbm_threshold.json

predict-rf: ## Make predictions on test data (Random Forest)
	python src/predict.py \
		--model artifacts/rf_model_pipeline.joblib \
		--schema artifacts/rf_feature_schema.json \
		--input data/processed/kepler_lc/test.csv \
		--output predictions_rf.csv \
		--threshold-file artifacts/rf_threshold.json

predict-custom-gb: ## Make predictions on custom data (usage: make predict-custom-gb INPUT=path/to/file.csv)
	python src/predict.py \
		--model artifacts/gb_model_pipeline.joblib \
		--schema artifacts/gb_feature_schema.json \
		--input $(INPUT) \
		--output predictions_gb.csv \
		--threshold-file artifacts/gb_threshold.json

predict-stack: ## Make predictions using stacking model
	python src/predict_stack.py \
		--meta-learner artifacts/meta_learner.joblib \
		--meta-info artifacts/meta_learner_info.json \
		--schema artifacts/gb_feature_schema.json \
		--input data/processed/kepler_lc/test.csv \
		--output predictions_stack.csv

compare: ## Compare all models on test data
	python src/compare_models.py --data data/processed/kepler_lc/test.csv

tpot-optimize: ## Run TPOT optimization to find best pipeline
	python src/tpot_optimizer.py \
		--data data/processed/kepler_lc/train.csv \
		--runs 5 \
		--generations 5 \
		--population-size 20 \
		--scoring f1 \
		--resampling smote \
		--output artifacts/tpot_results.json \
		--save-best artifacts/tpot_best_pipeline.json

tpot-optimize-balanced: ## Run TPOT optimization with balanced class weights and F1 scoring
	python src/tpot_optimizer.py \
		--data data/processed/kepler_lc/train.csv \
		--runs 3 \
		--generations 10 \
		--population-size 30 \
		--scoring f1 \
		--resampling smote \
		--output artifacts/tpot_balanced_results.json \
		--save-best artifacts/tpot_balanced_best_pipeline.json

tpot-optimize-smote: ## Run TPOT optimization with SMOTE resampling
	python src/tpot_optimizer.py \
		--data data/processed/kepler_lc/train.csv \
		--runs 3 \
		--generations 8 \
		--population-size 25 \
		--scoring f1 \
		--resampling smote \
		--output artifacts/tpot_smote_results.json \
		--save-best artifacts/tpot_smote_best_pipeline.json

clean: ## Clean generated files
	rm -rf artifacts/
	rm -rf plots/
	rm -f *.png
	rm -f predictions*.csv
	rm -f *_probabilities.csv

full-tuning: tune-gb tune-lgbm tune-rf

full-pipeline-gb: train-gb evaluate-gb predict-gb ## Run the complete ML pipeline for Gradient Boosting
full-pipeline-lgbm: train-lgbm evaluate-lgbm predict-lgbm ## Run the complete ML pipeline for LightGBM
full-pipeline-rf: train-rf evaluate-rf predict-rf ## Run the complete ML pipeline for Random Forest
full-pipeline: full-pipeline-gb full-pipeline-lgbm full-pipeline-rf stack compare
