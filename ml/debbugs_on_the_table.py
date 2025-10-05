import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer

breast = load_breast_cancer()

x = pd.DataFrame(breast.data, columns=breast.feature_names)
y = pd.Series(breast.target, name='type')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=68, stratify=y)

smote = SMOTE(sampling_strategy=0.75, random_state=68)
under = RandomUnderSampler(sampling_strategy=1.0, random_state=68)


pipe = Pipeline([
    ('lgbm', LGBMClassifier(random_state=68))
])

param_dist = {
    'lgbm__learning_rate': [0.05, 0.1],
    'lgbm__num_leaves': [31, 63],
    'lgbm__max_depth': [-1, 10],
    'lgbm__min_child_samples': [5, 10],
    'lgbm__subsample': [0.8, 1.0],
    'lgbm__colsample_bytree': [0.8, 1.0],
    'lgbm__boosting_type': ['gbdt'],  # remove 'dart' temporariamente
    'lgbm__n_estimators': [500]
}




search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    n_iter=50, 
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=3, 
    error_score='raise'
)

search.fit(x_train, y_train)
y_pred = search.predict(x_test)
print(classification_report(y_test, y_pred))










