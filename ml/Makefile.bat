@echo off
setlocal enabledelayedexpansion

REM Windows batch file equivalent of Makefile
REM Usage: Makefile.bat [command]
REM Example: Makefile.bat train-gb

if "%1"=="" goto :help
if "%1"=="help" goto :help
if "%1"=="install" goto :install
if "%1"=="train-gb" goto :train-gb
if "%1"=="train-lgbm" goto :train-lgbm
if "%1"=="train-rf" goto :train-rf
if "%1"=="tune-gb" goto :tune-gb
if "%1"=="tune-lgbm" goto :tune-lgbm
if "%1"=="tune-rf" goto :tune-rf
if "%1"=="train-tuned-gb" goto :train-tuned-gb
if "%1"=="train-tuned-lgbm" goto :train-tuned-lgbm
if "%1"=="train-tuned-rf" goto :train-tuned-rf
if "%1"=="stack" goto :stack
if "%1"=="evaluate-gb" goto :evaluate-gb
if "%1"=="evaluate-lgbm" goto :evaluate-lgbm
if "%1"=="evaluate-rf" goto :evaluate-rf
if "%1"=="evaluate-tuned-gb" goto :evaluate-tuned-gb
if "%1"=="evaluate-tuned-lgbm" goto :evaluate-tuned-lgbm
if "%1"=="evaluate-tuned-rf" goto :evaluate-tuned-rf
if "%1"=="predict-gb" goto :predict-gb
if "%1"=="predict-lgbm" goto :predict-lgbm
if "%1"=="predict-rf" goto :predict-rf
if "%1"=="predict-custom-gb" goto :predict-custom-gb
if "%1"=="predict-stack" goto :predict-stack
if "%1"=="compare" goto :compare
if "%1"=="tpot-optimize" goto :tpot-optimize
if "%1"=="tpot-optimize-balanced" goto :tpot-optimize-balanced
if "%1"=="tpot-optimize-smote" goto :tpot-optimize-smote
if "%1"=="clean" goto :clean
if "%1"=="full-tuning" goto :full-tuning
if "%1"=="full-pipeline-gb" goto :full-pipeline-gb
if "%1"=="full-pipeline-lgbm" goto :full-pipeline-lgbm
if "%1"=="full-pipeline-rf" goto :full-pipeline-rf
if "%1"=="full-pipeline" goto :full-pipeline

echo Unknown command: %1
goto :help

:help
echo Available commands:
echo.
echo   install                    Install dependencies
echo   train-gb                   Train the Gradient Boosting model
echo   train-lgbm                 Train the LightGBM model
echo   train-rf                   Train the Random Forest model
echo   tune-gb                    Hyperparameter tuning for Gradient Boosting
echo   tune-lgbm                  Hyperparameter tuning for LightGBM
echo   tune-rf                    Hyperparameter tuning for Random Forest
echo   train-tuned-gb             Train with best parameters from tuning (GB)
echo   train-tuned-lgbm           Train with best parameters from tuning (LightGBM)
echo   train-tuned-rf             Train with best parameters from tuning (Random Forest)
echo   stack                      Train stacking model with meta-learner
echo   evaluate-gb                Evaluate the trained Gradient Boosting model
echo   evaluate-lgbm              Evaluate the trained LightGBM model
echo   evaluate-rf                Evaluate the trained Random Forest model
echo   evaluate-tuned-gb          Evaluate the best estimator from tuning (GB)
echo   evaluate-tuned-lgbm        Evaluate the best estimator from tuning (LightGBM)
echo   evaluate-tuned-rf          Evaluate the best estimator from tuning (Random Forest)
echo   predict-gb                 Make predictions on test data (Gradient Boosting)
echo   predict-lgbm               Make predictions on test data (LightGBM)
echo   predict-rf                 Make predictions on test data (Random Forest)
echo   predict-custom-gb          Make predictions on custom data (usage: Makefile.bat predict-custom-gb INPUT=path/to/file.csv)
echo   predict-stack              Make predictions using stacking model
echo   compare                    Compare all models on test data
echo   tpot-optimize              Run TPOT optimization to find best pipeline
echo   tpot-optimize-balanced     Run TPOT optimization with balanced class weights and F1 scoring
echo   tpot-optimize-smote        Run TPOT optimization with SMOTE resampling
echo   clean                      Clean generated files
echo   full-tuning                Run all tuning commands
echo   full-pipeline-gb           Run the complete ML pipeline for Gradient Boosting
echo   full-pipeline-lgbm         Run the complete ML pipeline for LightGBM
echo   full-pipeline-rf           Run the complete ML pipeline for Random Forest
echo   full-pipeline              Run all complete ML pipelines
echo.
goto :end

:install
echo Installing dependencies...
pip install -r requirements.txt
goto :end

:train-gb
echo Training Gradient Boosting model...
python src/train.py --config configs/gb.yaml
goto :end

:train-lgbm
echo Training LightGBM model...
python src/train.py --config configs/lgbm.yaml
goto :end

:train-rf
echo Training Random Forest model...
python src/train.py --config configs/rf.yaml
goto :end

:tune-gb
echo Hyperparameter tuning for Gradient Boosting...
python src/tune.py --config configs/gb.yaml
goto :end

:tune-lgbm
echo Hyperparameter tuning for LightGBM...
python src/tune.py --config configs/lgbm.yaml
goto :end

:tune-rf
echo Hyperparameter tuning for Random Forest...
python src/tune.py --config configs/rf.yaml
goto :end

:train-tuned-gb
echo Training with best parameters from tuning (GB)...
python src/train.py --config artifacts/gb_tuned.yaml
goto :end

:train-tuned-lgbm
echo Training with best parameters from tuning (LightGBM)...
python src/train.py --config artifacts/lgbm_tuned.yaml
goto :end

:train-tuned-rf
echo Training with best parameters from tuning (Random Forest)...
python src/train.py --config artifacts/rf_tuned.yaml
goto :end

:stack
echo Training stacking model with meta-learner...
python src/stack.py --config configs/stack.yaml
goto :end

:evaluate-gb
echo Evaluating the trained Gradient Boosting model...
python src/evaluate.py --model artifacts/gb_model_pipeline.joblib --data artifacts/test_set.csv --threshold-file artifacts/gb_threshold.json --plots
goto :end

:evaluate-lgbm
echo Evaluating the trained LightGBM model...
python src/evaluate.py --model artifacts/lgbm_model_pipeline.joblib --data artifacts/test_set.csv --threshold-file artifacts/lgbm_threshold.json --plots
goto :end

:evaluate-rf
echo Evaluating the trained Random Forest model...
python src/evaluate.py --model artifacts/rf_model_pipeline.joblib --data artifacts/test_set.csv --threshold-file artifacts/rf_threshold.json --plots
goto :end

:evaluate-tuned-gb
echo Evaluating the best estimator from tuning (GB)...
python src/evaluate.py --model artifacts/gb_best_estimator.joblib --data artifacts/test_set.csv --threshold-file artifacts/gb_threshold.json --plots
goto :end

:evaluate-tuned-lgbm
echo Evaluating the best estimator from tuning (LightGBM)...
python src/evaluate.py --model artifacts/lgbm_best_estimator.joblib --data artifacts/test_set.csv --threshold-file artifacts/lgbm_threshold.json --plots
goto :end

:evaluate-tuned-rf
echo Evaluating the best estimator from tuning (Random Forest)...
python src/evaluate.py --model artifacts/rf_best_estimator.joblib --data artifacts/test_set.csv --threshold-file artifacts/rf_threshold.json --plots
goto :end

:predict-gb
echo Making predictions on test data (Gradient Boosting)...
python src/predict.py --model artifacts/gb_model_pipeline.joblib --schema artifacts/gb_feature_schema.json --input data/processed/kepler_lc/test.csv --output predictions_gb.csv --threshold-file artifacts/gb_threshold.json
goto :end

:predict-lgbm
echo Making predictions on test data (LightGBM)...
python src/predict.py --model artifacts/lgbm_model_pipeline.joblib --schema artifacts/lgbm_feature_schema.json --input data/processed/kepler_lc/test.csv --output predictions_lgbm.csv --threshold-file artifacts/lgbm_threshold.json
goto :end

:predict-rf
echo Making predictions on test data (Random Forest)...
python src/predict.py --model artifacts/rf_model_pipeline.joblib --schema artifacts/rf_feature_schema.json --input data/processed/kepler_lc/test.csv --output predictions_rf.csv --threshold-file artifacts/rf_threshold.json
goto :end

:predict-custom-gb
if "%INPUT%"=="" (
    echo Error: INPUT parameter not provided
    echo Usage: Makefile.bat predict-custom-gb INPUT=path/to/file.csv
    goto :end
)
echo Making predictions on custom data (Gradient Boosting)...
python src/predict.py --model artifacts/gb_model_pipeline.joblib --schema artifacts/gb_feature_schema.json --input %INPUT% --output predictions_gb.csv --threshold-file artifacts/gb_threshold.json
goto :end

:predict-stack
echo Making predictions using stacking model...
python src/predict_stack.py --meta-learner artifacts/meta_learner.joblib --meta-info artifacts/meta_learner_info.json --schema artifacts/gb_feature_schema.json --input data/processed/kepler_lc/test.csv --output predictions_stack.csv
goto :end

:compare
echo Comparing all models on test data...
python src/compare_models.py --data data/processed/kepler_lc/test.csv
goto :end

:tpot-optimize
echo Running TPOT optimization to find best pipeline...
python src/tpot_optimizer.py --data data/processed/kepler_lc/train.csv --runs 5 --generations 5 --population-size 20 --scoring f1 --resampling smote --output artifacts/tpot_results.json --save-best artifacts/tpot_best_pipeline.json
goto :end

:tpot-optimize-balanced
echo Running TPOT optimization with balanced class weights and F1 scoring...
python src/tpot_optimizer.py --data data/processed/kepler_lc/train.csv --runs 3 --generations 10 --population-size 30 --scoring f1 --resampling smote --output artifacts/tpot_balanced_results.json --save-best artifacts/tpot_balanced_best_pipeline.json
goto :end

:tpot-optimize-smote
echo Running TPOT optimization with SMOTE resampling...
python src/tpot_optimizer.py --data data/processed/kepler_lc/train.csv --runs 3 --generations 8 --population-size 25 --scoring f1 --resampling smote --output artifacts/tpot_smote_results.json --save-best artifacts/tpot_smote_best_pipeline.json
goto :end

:clean
echo Cleaning generated files...
if exist artifacts rmdir /s /q artifacts
if exist plots rmdir /s /q plots
if exist *.png del *.png
if exist predictions*.csv del predictions*.csv
if exist *_probabilities.csv del *_probabilities.csv
echo Clean completed.
goto :end

:full-tuning
echo Running full tuning pipeline...
call :tune-gb
call :tune-lgbm
call :tune-rf
goto :end

:full-pipeline-gb
echo Running complete ML pipeline for Gradient Boosting...
call :train-gb
call :evaluate-gb
call :predict-gb
goto :end

:full-pipeline-lgbm
echo Running complete ML pipeline for LightGBM...
call :train-lgbm
call :evaluate-lgbm
call :predict-lgbm
goto :end

:full-pipeline-rf
echo Running complete ML pipeline for Random Forest...
call :train-rf
call :evaluate-rf
call :predict-rf
goto :end

:full-pipeline
echo Running all complete ML pipelines...
call :full-pipeline-gb
call :full-pipeline-lgbm
call :full-pipeline-rf
call :stack
call :compare
goto :end

:end
endlocal
