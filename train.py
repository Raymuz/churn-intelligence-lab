"""Reproducible bank + telecom churn training; no customer row data is published."""
from __future__ import annotations
import argparse
import sys
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score,
    fbeta_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
SPECS = {
    'bank': {
        'filename': 'bank_customer_churn.csv', 'target': 'Exited',
        'numeric': ['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary'],
        'categorical': ['Geography'], 'excluded': ['CustomerId','Surname'],
        'display': 'Bank customer churn',
    },
    'telecom': {
        'filename': 'customer_churn_data.csv', 'target': 'Churn',
        'numeric': ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges'],
        'categorical': ['Partner','Dependents','PhoneService','InternetService','Contract','PaymentMethod'],
        'excluded': ['customerID'], 'display': 'Telecom customer churn',
    },
}

def pyvalue(v):
    if isinstance(v, np.generic):
        return v.item()
    return v

def load_dataset(path, spec):
    if not path.is_file():
        raise FileNotFoundError(f'Required dataset missing: {path}. Upload the original CSV in Colab.')
    raw = pd.read_csv(path)
    required = spec['numeric'] + spec['categorical'] + [spec['target']]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f'{path.name}: missing columns {missing}. Found {list(raw.columns)}')
    original_rows = len(raw)
    frame = raw.drop_duplicates().copy()
    # Never use identifiers, surnames, or the label as predictors.
    X = frame[spec['numeric'] + spec['categorical']].copy()
    for col in spec['numeric']:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in spec['categorical']:
        X[col] = X[col].astype('string').str.strip().replace('', pd.NA)
        X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
    if spec['target'] == 'Churn':
        y = frame[spec['target']].astype('string').str.strip().str.lower().map({'yes': 1, 'no': 0})
    else:
        y = pd.to_numeric(frame[spec['target']], errors='coerce')
    if y.isna().any() or not set(y.unique()).issubset({0, 1}):
        raise ValueError(f'{path.name}: target must be fully labeled binary 0/1 or Yes/No')
    y = y.astype('int64')
    if y.value_counts().min() < 20:
        raise ValueError(f'{path.name}: insufficient positive/negative examples')
    audit = {
        'original_rows': int(original_rows),
        'duplicate_rows_removed': int(original_rows-len(frame)),
        'rows_after_deduplication': int(len(frame)),
        'positive_rate': round(float(y.mean()), 5),
        'feature_missing_counts': {col: int(X[col].isna().sum()) for col in X.columns},
        'sha256_first12': hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    }
    return X, y, audit

def preprocessing(spec):
    numerical = Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    return ColumnTransformer([('numeric', numerical, spec['numeric']),
                              ('categorical', categorical, spec['categorical'])],
                             remainder='drop')

def score_proba(model, X):
    return model.predict_proba(X)[:, 1]

def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        'roc_auc': round(float(roc_auc_score(y, p)), 5),
        'pr_auc': round(float(average_precision_score(y, p)), 5),
        'balanced_accuracy': round(float(balanced_accuracy_score(y, pred)), 5),
        'precision': round(float(precision_score(y, pred, zero_division=0)), 5),
        'recall': round(float(recall_score(y, pred, zero_division=0)), 5),
        'f1': round(float(f1_score(y, pred, zero_division=0)), 5),
        'f2': round(float(fbeta_score(y, pred, beta=2, zero_division=0)), 5),
        'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(y, pred, labels=[0, 1],
                                                     output_dict=True, zero_division=0),
    }

def tune_threshold(y, p):
    thresholds = np.round(np.arange(0.10, 0.901, 0.025), 3)
    values = [fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(values))])

def plot_diagnostics(project, key, ytest, ptest, threshold, importance_df):
    report_dir = project / 'reports'
    pred = (ptest >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(ytest, pred, display_labels=['Retained', 'Churned'],
                                            cmap='Blues', colorbar=False, ax=ax)
    ax.set_title(f'{key.title()} — untouched test data')
    fig.tight_layout()
    fig.savefig(report_dir / f'{key}_confusion_matrix.png', dpi=140)
    plt.close(fig)
    fpr, tpr, _ = roc_curve(ytest, ptest)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(ytest, ptest):.3f}')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Chance')
    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=f'{key.title()} — test ROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(report_dir / f'{key}_roc.png', dpi=140)
    plt.close(fig)
    top = importance_df.head(10).sort_values('importance_mean')
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top['feature'], top['importance_mean'])
    ax.set(xlabel='Change in validation PR AUC after shuffling', title=f'{key.title()} — permutation importance')
    fig.tight_layout()
    fig.savefig(report_dir / f'{key}_feature_importance.png', dpi=140)
    plt.close(fig)

def train_one(project, source_dir, key, tuning_iterations):
    spec = SPECS[key]
    path = source_dir / spec['filename']
    X, y, audit = load_dataset(path, spec)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.20,
                                                       random_state=SEED, stratify=y_dev)
    minority_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    models = {
        'Logistic regression': LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED),
        'Random forest': RandomForestClassifier(n_estimators=170, min_samples_leaf=3,
                                                class_weight='balanced_subsample', n_jobs=-1, random_state=SEED),
    }
    candidates = {}
    leaderboard = []
    for name, estimator in models.items():
        pipeline = Pipeline([('preprocessor', preprocessing(spec)), ('model', estimator)])
        pipeline.fit(X_train, y_train)
        candidates[name] = pipeline
        row = metrics(y_val, score_proba(pipeline, X_val), 0.5)
        leaderboard.append({'dataset': key, 'model': name, 'validation_pr_auc': row['pr_auc'],
                            'validation_roc_auc': row['roc_auc'], 'tuning': 'none'})
    xgb_pipe = Pipeline([('preprocessor', preprocessing(spec)),
        ('model', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=minority_weight,
            eval_metric='logloss', tree_method='hist', random_state=SEED, n_jobs=1))])
    if tuning_iterations:
        params = {
            'model__n_estimators': [100, 180, 260],
            'model__max_depth': [2, 3, 4],
            'model__learning_rate': [0.025, 0.05, 0.1],
            'model__subsample': [0.8, 1.0],
            'model__colsample_bytree': [0.8, 1.0],
            'model__min_child_weight': [1, 3],
        }
        search = RandomizedSearchCV(xgb_pipe, params, n_iter=tuning_iterations,
              scoring='average_precision', cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
              random_state=SEED, n_jobs=2, refit=True, verbose=0, error_score='raise')
        search.fit(X_train, y_train)
        xgb_pipe = search.best_estimator_
        print(f'[{key}] XGBoost best 3-fold training CV PR AUC: {search.best_score_:.4f}')
    else:
        xgb_pipe.fit(X_train, y_train)
    candidates['XGBoost'] = xgb_pipe
    val_xgb = metrics(y_val, score_proba(xgb_pipe, X_val), 0.5)
    leaderboard.append({'dataset': key, 'model': 'XGBoost',
        'validation_pr_auc': val_xgb['pr_auc'], 'validation_roc_auc': val_xgb['roc_auc'],
        'tuning': f'{tuning_iterations}x3 CV' if tuning_iterations else 'off'})
    leaderboard.sort(key=lambda r: (r['validation_pr_auc'], r['validation_roc_auc']), reverse=True)
    winner = leaderboard[0]['model']
    champion = candidates[winner]
    threshold = tune_threshold(y_val, score_proba(champion, X_val))
    # The test partition has never been used to choose model, hyperparameters, or threshold.
    test_scores = score_proba(champion, X_test)
    test = metrics(y_test, test_scores, threshold)
    # Deliberate launch-quality gate: near-chance classifiers remain research-only.
    # Hold-out data are used for final model assessment, not training/selection.
    reliable = (test['roc_auc'] >= 0.60 and test['pr_auc'] >= float(y_test.mean()) + 0.03)
    deployment_status = 'demonstration' if reliable else 'research_only_low_signal'
    if not reliable:
        print(f'[{key}] WARNING: near-chance model; API will not issue actionable churn labels.')
    val = metrics(y_val, score_proba(champion, X_val), threshold)
    defaults = {c: float(X_train[c].median()) for c in spec['numeric']}
    categories = {}
    for col in spec['categorical']:
        vals = X_train[col].dropna().astype(str)
        categories[col] = sorted(vals.unique().tolist())
        defaults[col] = str(vals.mode().iloc[0])
    importance = permutation_importance(champion, X_val, y_val, scoring='average_precision',
                                        n_repeats=3, random_state=SEED, n_jobs=1)
    importance_df = pd.DataFrame({'feature': spec['numeric'] + spec['categorical'],
        'importance_mean': importance.importances_mean, 'importance_std': importance.importances_std})
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    importance_df.to_csv(project / 'reports' / f'{key}_feature_importance.csv', index=False)
    plot_diagnostics(project, key, y_test, test_scores, threshold, importance_df)
    bundle = {'pipeline': champion, 'threshold': threshold,
              'numeric': spec['numeric'], 'categorical': spec['categorical'], 'features': spec['numeric'] + spec['categorical']}
    path_model = project / 'models' / f'{key}_churn.joblib'
    joblib.dump(bundle, path_model, compress=3)
    reloaded = joblib.load(path_model)
    assert np.allclose(reloaded['pipeline'].predict_proba(X_test.iloc[:5])[:,1], test_scores[:5])
    metadata = {
       'label': spec['display'], 'selected_model': winner,
       'numeric': spec['numeric'], 'categorical': spec['categorical'],
       'features': spec['numeric'] + spec['categorical'],
       'excluded_columns': spec['excluded'], 'categorical_choices': categories,
       'defaults': defaults, 'threshold': threshold,
       'training_rows': len(X_train), 'validation_rows': len(X_val), 'test_rows': len(X_test),
       'audit': audit, 'validation_metrics': val, 'test_metrics': test,
       'deployment_status': deployment_status,
       'model_size_mb': round(path_model.stat().st_size / 1024**2, 3),
       'validation_model_comparison': leaderboard,
    }
    print(f'[{key}] Winner: {winner}, threshold: {threshold:.3f}, '
          f'test PR AUC: {test["pr_auc"]:.4f}, ROC AUC: {test["roc_auc"]:.4f}, recall: {test["recall"]:.4f}')
    return metadata, leaderboard

def main(data_dir='.', project_dir='.', tuning_iterations=6):
    project = Path(project_dir).resolve()
    data_dir = Path(data_dir).resolve()
    (project / 'models').mkdir(parents=True, exist_ok=True)
    (project / 'reports').mkdir(parents=True, exist_ok=True)
    results, leaderboard = {}, []
    for key in SPECS:
        print(f'\n=== TRAINING {key.upper()} ===')
        results[key], comparison = train_one(project, data_dir, key, tuning_iterations)
        leaderboard.extend(comparison)
    summary = {'generated_utc': datetime.now(timezone.utc).isoformat(),
      'random_seed': SEED, 'sklearn_version': sklearn.__version__,
      'xgboost_version': xgboost.__version__, 'python_version': sys.version.split()[0], 'datasets': results,
      'decision_rule': 'maximize validation F2 (recall-weighted), after selecting by validation PR AUC',
      'probability_warning': 'Scores are uncalibrated model probabilities; not validated as true churn probabilities.'}
    (project / 'model_metadata.json').write_text(json.dumps(summary, indent=2, default=pyvalue), encoding='utf-8')
    pd.DataFrame(leaderboard).to_csv(project / 'reports' / 'model_comparison.csv', index=False)
    print('\nSaved trained pipelines, evaluation, charts, and audit metadata to', project)
    return summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='.')
    parser.add_argument('--project-dir', default='.')
    parser.add_argument('--tuning-iterations', type=int, default=6)
    args = parser.parse_args()
    main(args.data_dir, args.project_dir, args.tuning_iterations)
