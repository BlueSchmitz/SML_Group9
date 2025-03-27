# %% import all necessary packages
import numpy as np
import pandas as pd
import imblearn
import sklearn
import time
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score, mean_squared_error
from imblearn.combine import SMOTETomek
import random
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
import seaborn as sns
import shap

# %% load data and select model
data = pd.read_csv('/content/aggr_sol_dataset.csv')
data = data.drop(2478, axis='index')
data = data[data['TMHMM'] != 1]

# model 1 (surface features)
'''
model_data = data[['thsa_netsurfp2', 'rhsa_netsurfp2', 'Gravy', 'LHPpred', 'C_exposed']].copy()
model_data['positive_surface'] = data[['R_exposed', 'H_exposed', 'K_exposed']].sum(axis=1)
model_data['negative_surface'] = data[['D_exposed', 'E_exposed']].sum(axis=1)
model_data['nonpolar_surface'] = data[['A_exposed', 'V_exposed', 'I_exposed', 'L_exposed', 'M_exposed', 'F_exposed', 'Y_exposed', 'W_exposed', 'P_exposed', 'G_exposed']].sum(axis=1)
model_data['polar_surface'] = data[['S_exposed', 'T_exposed', 'N_exposed', 'Q_exposed']].sum(axis=1)
model_data['LHP_norm'] = data['LHPpred']/data['molecular_weight']
# target
model_data['solubility'] = data['solubility']
print(model_data.head())
'''


# model 2 (stability)
'''
model_data = data[['length', 'helix', 'sheet', 'disorder', 'Instability_index', 'charge_at_5', 'charge_at_7']].copy()
model_data['charge_change'] = data['charge_at_7'] - data['charge_at_5']
#target
model_data['solubility'] = data['solubility']
print(model_data.head())
'''

# model 3 (Surface features + Stability)

model_data = data[['thsa_netsurfp2', 'rhsa_netsurfp2', 'Gravy', 'LHPpred', 'C_exposed', 'length', 'helix', 'sheet', 'disorder', 'Instability_index','charge_at_5', 'charge_at_7']].copy()
model_data['positive_surface'] = data[['R_exposed', 'H_exposed', 'K_exposed']].sum(axis=1)
model_data['negative_surface'] = data[['D_exposed', 'E_exposed']].sum(axis=1)
model_data['nonpolar_surface'] = data[['A_exposed', 'V_exposed', 'I_exposed', 'L_exposed', 'M_exposed', 'F_exposed', 'Y_exposed', 'W_exposed', 'P_exposed', 'G_exposed']].sum(axis=1)
model_data['polar_surface'] = data[['S_exposed', 'T_exposed', 'N_exposed', 'Q_exposed']].sum(axis=1)
model_data['LHP_norm'] = data['LHPpred']/data['molecular_weight']
model_data['charge_change'] = data['charge_at_7'] - data['charge_at_5']
#target
model_data['solubility'] = data['solubility']
print(model_data.head())

np.random.seed(42)
random.seed(42)

# %% Define model, parameters and scoring metrics 
# Define models
models = {
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    }
}

# Define scoring metrics
scoring = {
    'mae': make_scorer(mean_absolute_error),
    'r2': make_scorer(r2_score),
    'mse': make_scorer(mean_squared_error)
}

# %% Separate train and test data using binning
num_bins = 10
model_data['y_binned'] = pd.qcut(model_data['solubility'], q=num_bins, duplicates='drop')
train, test = train_test_split(
    model_data,
    test_size=1000,
    stratify=model_data['y_binned'],
    random_state=42
)

# Plot train-test solubility distributions
plt.figure(figsize=(12, 5))
sns.histplot(model_data['solubility'], bins=30, kde=True, color='blue', label='Original', stat='density')
sns.histplot(train['solubility'], bins=30, kde=True, color='green', label='Train', stat='density')
sns.histplot(test['solubility'], bins=30, kde=True, color='red', label='Test', stat='density')
plt.xlabel('Solubility')
plt.ylabel('Density')
plt.title('Comparison of Solubility Distributions')
plt.legend()
plt.show()

# Remove binned labels
train = train.drop(columns=['y_binned'])
test = test.drop(columns=['y_binned'])
X_train = train.drop(columns=['solubility'])
y_train = train['solubility']

# %% Function to evaluate models with hyperparameter tuning and CV
def evaluate_resampling_and_hyperparameters(X, y):
    results = {}
    num_bins = 10
    y_binned = pd.qcut(y, q=num_bins, duplicates="drop").cat.codes  # Convert to numeric codes
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model_info in models.items():
        model = model_info['model']
        model_param_grid = model_info['param_grid']

        best_score = np.inf
        best_params = None
        best_scores = None

        fold_idx = 0  # Track fold index

        for model_params in ParameterGrid(model_param_grid):
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing Model: {model_name} | Model Params: {model_params}")
            model_instance = model.__class__(random_state=42, **model_params)

            fold_scores = {metric: [] for metric in scoring.keys()}

            for train_idx, valid_idx in cv.split(X, y_binned):
                X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
                y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]

                # Plot solubility distribution for each fold
                plt.figure(figsize=(12, 5))
                sns.histplot(y_train_fold, bins=30, kde=True, color='blue', label='Train', stat='density')
                sns.histplot(y_valid_fold, bins=30, kde=True, color='red', label='Validation', stat='density')
                plt.xlabel('Solubility')
                plt.ylabel('Density')
                plt.title(f'Solubility Distribution in Fold {fold_idx+1}')
                plt.legend()
                plt.show()

                fold_idx += 1  # Increment fold index

                model_instance.fit(X_train_fold, y_train_fold)
                y_pred = model_instance.predict(X_valid_fold)

                for metric, scorer in scoring.items():
                    fold_scores[metric].append(scorer._score_func(y_valid_fold, y_pred))

            # Select best hyperparameters based on lowest MSE
            mean_score = np.mean(fold_scores['mse'])
            if mean_score < best_score:
                best_score = mean_score
                best_params = {'model_params': model_params}
                best_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}

        results[model_name] = {
            'best_params': best_params,
            'scores': best_scores
        }

    return pd.DataFrame({k: v['scores'] for k, v in results.items()}).T, results

# %% Run function
results_df, detailed_results = evaluate_resampling_and_hyperparameters(X_train, y_train)

# Print best parameters and performance
for model_name, result in detailed_results.items():
    print(f"\n Best Hyperparameters for {model_name}:")
    for param, value in result['best_params']['model_params'].items():
        print(f"  {param}: {value}")

    print("\n Performance Metrics:")
    for metric, score in result['scores'].items():
        print(f"  {metric.upper()}: {score:.4f}")

    print("-" * 50)

# %% Final model training and evaluation on test set 
# Separate train and test data using binning
num_bins = 10
model_data['y_binned'] = pd.qcut(model_data['solubility'], q=num_bins, duplicates='drop')
train, test = train_test_split(
    model_data,
    test_size=1000,
    stratify=model_data['y_binned'],
    random_state=42
)

# Prepare data
train = train.drop(columns=['y_binned'])  # Remove binning column
test = test.drop(columns=['y_binned'])

X_train = train.drop(columns=['solubility'])
y_train = train['solubility']

X_test = test.drop(columns=['solubility'])
y_test = test['solubility']

# Plot train-test solubility distributions
plt.figure(figsize=(12, 5))
sns.histplot(model_data['solubility'], bins=30, kde=True, color='blue', label='Original', stat='density')
sns.histplot(train['solubility'], bins=30, kde=True, color='green', label='Train', stat='density')
sns.histplot(test['solubility'], bins=30, kde=True, color='red', label='Test', stat='density')
plt.xlabel('Solubility')
plt.ylabel('Density')
plt.title('Comparison of Solubility Distributions')
plt.legend()
plt.show()

# Define model and parameters
best_params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 2
}
rf_final = RandomForestRegressor(**best_params)
rf_final.fit(X_train, y_train)

# Evaluate on Test Set
y_pred = rf_final.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Final Model Performance on Test Set:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"R²: {r2:.4f}")

# feature importance
importances = rf_final.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest (Best Model)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# SHAP Values
explainer = shap.TreeExplainer(rf_final)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(10, 5))
shap.summary_plot(shap_values, X_test)
plt.show()