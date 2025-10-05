# Exoplanet Detection ML Pipeline

## Quick Start

#### Lazy way (does literally everything, except tuning)

```bash
# Install dependencies
make install

# Train all models and compare performance
make full-pipeline
```

#### Step-by-step way (does only training):

```bash
# Install dependencies
make install

# Train Gradient Boosting
make train-gb

# Train LGBM
make train-lgbm

# Train meta-learner
make stack

# Compare results
make compare
```

## Project Structure

```
ml/
├── configs/           # Model configurations
│   ├── gb.yaml       # Gradient Boosting config
│   ├── lgbm.yaml     # LightGBM config
│   └── stack.yaml    # Stacking config
├── src/              # Source code
│   ├── train.py      # Training script
│   ├── tune.py       # Hyperparameter tuning
│   ├── stack.py      # Stacking implementation
│   ├── evaluate.py   # Model evaluation
│   ├── predict.py    # Single model predictions
│   ├── predict_stack.py # Stacking predictions
│   └── compare_models.py # Model comparison
├── artifacts/        # Trained models and results
├── data/            # Dataset
└── Makefile         # Command automation
```

## Available Commands

### Training

```bash
make train-gb          # Train Gradient Boosting
make train-lgbm        # Train LightGBM
make stack             # Train stacking ensemble
```

### Hyperparameter Tuning

```bash
make tune-gb           # Tune Gradient Boosting
make tune-lgbm         # Tune LightGBM
```

### Evaluation

```bash
make evaluate-gb       # Evaluate GB model
make evaluate-lgbm     # Evaluate LightGBM
make compare           # Compare all models
```

### Predictions

```bash
make predict-gb        # Predict with GB
make predict-lgbm      # Predict with LightGBM
make predict-stack     # Predict with stacking
make predict-custom-gb INPUT=path/to/file.csv  # Custom input
```

### Full Pipelines

```bash
make full-pipeline-gb     # Complete GB pipeline
make full-pipeline-lgbm   # Complete LightGBM pipeline
```

## Hyperparameter Tuning

### Configuration Files

Each model has its own configuration file in `configs/`:

- **`gb.yaml`** - Gradient Boosting parameters
- **`lgbm.yaml`** - LightGBM parameters
- **`rf.yaml`** - Random Forest parameters
- **`stack.yaml`** - Stacking meta-learner parameters
