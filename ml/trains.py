import pandas as pd
import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from tqdm import tqdm

RDS = 68

def smoted(percent):
    return SMOTE(sampling_strategy=percent, random_state=RDS)

def underSampled(percent):
    return RandomUnderSampler(sampling_strategy=percent, random_state=RDS)

def charge(path):
    df = pd.read_csv(path)
    y = df['label']
    x = df.drop(['label', 'row_index'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RDS, stratify=y)
    
    return [x_train, x_test, y_train, y_test]


def rSearch(pipe, param_dist, n_iter, cv, scoring, n_jobs):
    return RandomizedSearchCV(
        estimator=pipe, 
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose = 3
    )

def gSearch(pipe, param_grid, cv, scoring, n_jobs):
    return GridSearchCV(
        estimator=pipe, 
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose = 3
    )

def tSearch(pipe, param_grid, x_train, x_test, y_train, y_test): #Não usável
    param_list = list(ParameterGrid(param_grid))
    results = []
    for params in tqdm(param_list, desc="Hyperparameters test"):
        pipe.set_params(**params)
        pipe.fit(x_train, y_train)
        y_proba = pipe.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results.append({**params, 'roc_auc': auc})

    results_df = pd.DataFrame(results)
    print(results_df.sort_values('roc_auc', ascending=False))


def showResults(search):
    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best scores: {search.best_score_}")

def fImportance(search, model):
    df = pd.read_csv("../data/batch_results.csv")
    feature_names = df.drop(['label', 'row_index'], axis=1).columns.tolist()
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': search.best_estimator_.named_steps[model].feature_importances_
    }).sort_values(by='importance', ascending=False)
    print()
    print(search.best_estimator_.named_steps[model].feature_importances_)
    print()
    print(f"Feature Importance:\n{feature_importance}")


def train_rf(tunning):
    x_train, x_test, y_train, y_test = charge(".../data/batch_results.csv")
    

    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=RDS)),
        ('under', RandomUnderSampler(random_state=RDS)),
        ('rf', RandomForestClassifier(random_state=RDS))
    ])

    # param_dist = {
    #     'smote__sampling_strategy':[0.25],
    #     'smote__k_neighbors': [3, 5],
    #     'under__sampling_strategy':[0.75],
    #     'rf__n_estimators':[300, 350, 400],
    #     'rf__max_depth': [10, 15, 20],
    #     'rf__min_samples_split':[2, 5, 10],
    #     'rf__min_samples_leaf': [1, 2, 5],
    #     'rf__max_features': ["sqrt", "log2", 0.5, 0.7, None],
    #     'rf__class_weight': [None, "balanced"]
    # }


    param_dist = {
        'smote__sampling_strategy':[0.2, 0.25, 0.3],
        'smote__k_neighbors': [3, 5],
        'under__sampling_strategy':[0.7, 0.75, 0.8],
        'rf__n_estimators':[100, 200, 300, 350, 400],
        'rf__max_depth': [None, 5, 10, 15, 20],
        'rf__min_samples_split':[2, 5, 10, 20, 50],
        'rf__min_samples_leaf': [1, 2, 5, 10, 20],
        'rf__max_features': ["sqrt", "log2", 0.5, 0.7, None],
        'rf__class_weight': [None, "balanced", "balanced_subsample"]
    }


    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RDS)

    if tunning == 1:
        search = rSearch(pipe, param_dist, 150, cv, "average_precision", -1)
    elif tunning == 2:
        search = gSearch(pipe, param_dist, 3, "average_precision", -1)
    else:
        tSearch(pipe, param_dist, x_train, x_test, y_train, y_test)
        return
        
    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    print(f"Classification score: \n{classification_report(y_test, y_pred, zero_division=0)}")
    showResults(search)
    fImportance(search, 'rf')


def train_gb(tunning):
    x_train, x_test, y_train, y_test = charge("../data/batch_results.csv")

    # over = smoted(0.25)                  #0.2 and 0.75 | 0.25 and 0.75
    # under = underSampled(0.8)
    
    pipe = ImbPipeline([
        ('smote', SMOTE(random_state=RDS)),
        ('under', RandomUnderSampler(random_state=RDS)),
        ('gb', GradientBoostingClassifier(random_state=RDS))
    ])


    param_dist = {
        'smote__sampling_strategy': [0.25],
        'smote__k_neighbors': [3, 5],
        'under__sampling_strategy': [0.75], 
        'gb__n_estimators': [200, 300, 350, 450],       
        'gb__learning_rate': [0.05, 0.1, 0.15,  0.2],     
        'gb__max_depth': [1, 3, 5,  7],     
        'gb__min_samples_split': [2, 4, 6, 10, 20],    
        'gb__min_samples_leaf': [5, 7, 10],  
        'gb__subsample': [0.7, 0.8, 1.0],
        'gb__max_features': [None, 'sqrt', 'log2', 0.5, 0.8], 
        'gb__loss': ['log_loss'], 
        'gb__validation_fraction': [0.1, 0.15, 0.2],
        'gb__n_iter_no_change': [10, 20, 30, 40, 50]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RDS)

    # param_dist = {
    #     'gb__n_estimators': [100, 200, 300, 400, 500],       
    #     'gb__learning_rate': [0.01, 0.05, 0.1, 0.15,  0.2],     
    #     'gb__max_depth': [3, 4, 5, 6, 8],     
    #     'gb__min_samples_split': [2, 5, 10, 20],    
    #     'gb__min_samples_leaf': [1, 2, 5, 10],  
    #     'gb__subsample': [0.6, 0.8, 1.0],
    #     'gb__max_features': [None, 'sqrt', 'log2', 0.5], 
    #     'gb__loss': ['log_loss', 'exponential']         
    # }

    if tunning == 1:
        search = rSearch(pipe, param_dist, 150, cv, "average_precision", -1)
    elif tunning == 2:
        search = gSearch(pipe, param_dist, 3, "average_precision", -1)
    else:
        tSearch(pipe, param_dist, x_train, x_test, y_train, y_test)    

    search.fit(x_train, y_train)
    y_pred = search.predict(x_test)
    print(f"Classification score: \n{classification_report(y_test, y_pred, zero_division=0)}")
    showResults(search)
    fImportance(search, 'gb')



def train_lgbm():
    x_train, x_test, y_train, y_test =  charge("../data/batch_results.csv")

    # x_train_full, x_test, y_train_full, y_test =  charge("../data/batch_results.csv")

    # x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=RDS, stratify=y_train_full)

    over = smoted(0.25)              #0.25 and 0.75 for 260 itens, basic for anothers
    under = underSampled(0.75)

    pipe = ImbPipeline([
        ('over', over),
        ('under', under),
        ('lgbm', LGBMClassifier(
            random_state=RDS,      
            n_estimators=100,      
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31, 
            min_child_samples=20,
            class_weight='balanced',
            # objective='binary', 
            # metric='auc',
            boosting_type='gbdt',
            # verbose=-1,
            subsample=0.5,
            colsample_bytree=1,
            reg_alpha=0,
            reg_lambda=0,
        ))
    ])

    # def objective(trial): #INÚTIL
    #     lr = trial.suggest_float('learning_rate', 0.01, 0.1)
    #     num_leaves = trial.suggest_int('num_leaves', 31, 127)
    #     max_depth = trial.suggest_int('max_depth', 3, 20)
    #     min_child_samples = trial.suggest_int('min_child_samples', 10, 30)
    #     subsample = trial.suggest_float('subsample', 0.7, 1.0)
    #     colsample_bytree = trial.suggest_float('colsample_bytree', 0.7, 1.0)
    #     n_estimators = trial.suggest_int('n_estimators', 100, 1000)

    #     model = LGBMClassifier(
    #         learning_rate=lr,
    #         num_leaves=num_leaves,
    #         max_depth=max_depth,
    #         min_child_samples=min_child_samples,
    #         subsample=subsample,
    #         colsample_bytree=colsample_bytree,
    #         boosting_type='gbdt',
    #         n_estimators=n_estimators
    #     )

    #     model.fit(x_train, y_train)
    #     y_pred = model.predict_proba(x_val)[:, 1]
    #     auc = roc_auc_score(y_val, y_pred)
    #     return auc

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=30, show_progress_bar=True)

    # print(f"Best hyperparameters: {study.best_params}")
    # print(f"Best auc from validation: {study.best_value}")

    # best_params = study.best_params
    # final_model = LGBMClassifier(
    #     learning_rate=best_params['learning_rate'],
    #     num_leaves=best_params['num_leaves'],
    #     max_depth=best_params['max_depth'],
    #     min_child_samples=best_params['min_child_samples'],
    #     subsample=best_params['subsample'],
    #     colsample_bytree=best_params['colsample_bytree'],
    #     n_estimators=best_params['n_estima'], 
    #     random_state=RDS,
    #     boosting_type='gbdt'
    # )
    # final_model.fit(x_train_full, y_train_full)

    # y_pred = final_model.predict(x_test)
    # y_pred_proba = final_model.predict_proba(x_test)[:, 1]
    # print("\nRelatório de classificação no teste:")
    # print(classification_report(y_test, y_pred))
    # test_auc = roc_auc_score(y_test, y_pred_proba)
    # print(f"AUC-ROC no teste: {test_auc:.4f}")
    # showResults(search)
    # fImportance(search, 'lgbm')

    pipe.fit(x_train, y_train)
    y_pred = pipe.predict(x_test)
    y_pred_proba = pipe.predict_proba(x_test)[:, 1]
    print(f"Classification score: \n{classification_report(y_test, y_pred, zero_division=0)}")
    print(roc_auc_score(y_test, y_pred_proba))

train_rf(1)