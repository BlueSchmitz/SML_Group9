# %% Import packages
import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from datetime import datetime
from imblearn.pipeline import Pipeline
import shap

# %% load data and select model
data = pd.read_csv('/aggr_sol_dataset.csv')
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
# Encode labels: soluble -> 0, aggregator -> 1
model_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})
print(model_data.head())
'''


# model 2 (stability)
'''
model_data = data[['length', 'helix', 'sheet', 'disorder', 'Instability_index', 'charge_at_5', 'charge_at_7']].copy()
model_data['charge_change'] = data['charge_at_7'] - data['charge_at_5']
model_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})
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
model_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})
print(model_data.head())

# %%
np.random.seed(42)
random.seed(42)

# %% Define model to use (Random Forest)
models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    }
}

# %% Separate train and test data while preserving class distribution
train, test = train_test_split(
    model_data,
    test_size=1000,  # 500 instances in the test set (can change this)
    stratify=model_data['set'],  # Preserve class distribution
    random_state=42  # For reproducibility
)
# exclude Y for training
X_train = train.drop(columns=['set'])  # Drop the target column to get features + solubility
y_train = train['set']  # Target column

# Check class distribution
print("Original class distribution:\n", model_data['set'].value_counts(normalize=True))
print("\nTest set class distribution:\n", test['set'].value_counts(normalize=True))
print("\nTrain set size:", len(train))
print("Test set size:", len(test))

# %% Define resampling methods to try and hyperparameters to tune
resampling_methods = {
    'None': (None, [{}]),
    'RandomOverSampler': (RandomOverSampler, [{'sampling_strategy': [0.5, 0.75, 1.0]}]),
    'SMOTE': (SMOTE, [{'sampling_strategy': [0.5, 0.75, 1.0], 'k_neighbors': [3, 5, 7]}]),
    'RandomUnderSampler': (RandomUnderSampler, [{'sampling_strategy': [0.5, 0.75, 1.0]}]),
    'SMOTE-Tomek': (SMOTETomek, [{'sampling_strategy': [0.5, 0.75, 1.0]}]),
    'ADASYN': (ADASYN, [{'sampling_strategy': [0.5, 0.75, 1.0], 'n_neighbors': [3, 5, 7]}]),
}

models = {
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    }
}

# Define scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, pos_label=1, average='binary', zero_division=0),  # Score for the aggregator class (1)
    'recall': make_scorer(recall_score, pos_label=1, average='binary', zero_division=0),  # Score for the aggregator class (1)
    'f1': make_scorer(f1_score, pos_label=1, average='binary', zero_division=0),  # Score for the aggregator class (1)
    'roc_auc': make_scorer(roc_auc_score, average='weighted')  # No change needed for ROC AUC
}

# %% Define the function to evaluate resampling methods with hyperparameter grid search
def evaluate_resampling_and_hyperparameters(X, y):
    results = {}  # initialize results
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # StratifiedKFold ensures the same distribution in splits, as in original data

    for model_name, model_info in models.items():  # Loop through models (Random Forest only)
        model = model_info['model']
        model_param_grid = model_info['param_grid']

        for name, (sampler_class, param_grid) in resampling_methods.items():  # Loop through resampling methods and hyperparameters
            best_score = -np.inf
            best_params = None
            best_scores = None

            for params in ParameterGrid(param_grid):  # Iterate over all possible hyperparameter combinations for each sampler
                for model_params in ParameterGrid(model_param_grid):
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Testing Model: {model_name} | Resampling: {name} | Resampler Params: {params} | Model Params: {model_params}")
                    steps = []  # Initialize list to store pipeline steps
                    if sampler_class:  # Check if resampling method is defined
                        sampler = sampler_class(**params)  # Initialize resampler with current hyperparameters
                        steps.append(('sampler', sampler))  # Add the resampling steps to the pipeline
                    model = RandomForestClassifier(random_state=42, **model_params)  # Only RandomForestClassifier
                    steps.append(('classifier', model))  # Add the classifier to the pipeline

                    pipeline = Pipeline(steps)  # Create the pipeline from steps
                    fold_scores = {metric: [] for metric in scoring.keys()}  # Initializes empty dictionary

                    for train_idx, valid_idx in cv.split(X, y):
                        X_train_fold, X_valid_fold = X.iloc[train_idx], X.iloc[valid_idx]
                        y_train_fold, y_valid_fold = y.iloc[train_idx], y.iloc[valid_idx]
                        pipeline.fit(X_train_fold, y_train_fold)
                        y_pred = pipeline.predict(X_valid_fold)

                        # Check if there are no positive predictions
                        if np.all(y_pred == 0) or np.all(y_pred == 1):  # Only one class predicted
                            print(f"\nWarning: Model predicted only one class! | Sampler: {sampler_class.__name__ if sampler_class else 'None'} | "
                            f"Sampler Params: {params} | Model: {model_name} | Model Params: {model_params} | "
                            f"Train Data Size: {X_train_fold.shape} | Validation Data Size: {X_valid_fold.shape} | "
                            f"Unique Predictions: {dict(zip(*np.unique(y_pred, return_counts=True)))}\n")

                        for metric, scorer in scoring.items():
                            fold_scores[metric].append(scorer._score_func(y_valid_fold, y_pred))

                    # For each resampling method, hyperparameters yielding the highest mean score across folds are selected
                    mean_score = np.mean(fold_scores['f1'])
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'sampler_params': params, 'model_params': model_params}
                        best_scores = {metric: np.mean(scores) for metric, scores in fold_scores.items()}

            results[(model_name, name)] = {  # Store best results for each resampler and model
                'best_params': best_params,
                'scores': best_scores
            }

    return pd.DataFrame({k: v['scores'] for k, v in results.items()}).T, results  # Returns a summary dataframe showing the best scores for each resampling method and a dict

# %% Run the evaluation pipeline on the training data
results_df, detailed_results = evaluate_resampling_and_hyperparameters(X_train, y_train)

# Print out the summary of results
print("Summary of Resampling Strategies and Their Scores:\n")
print(results_df)

# Print best hyperparameters for each method
print("\nBest Hyperparameters for Each Resampling Method and Model:\n")
for (model_name, method), info in detailed_results.items():
    print(f"{model_name} - {method}: {info['best_params']}")

# Print best model hyperparameters separately
print("\nBest Model Hyperparameters Across All Resampling Methods:\n")
best_overall = max(detailed_results.items(), key=lambda x: x[1]['scores']['f1'])
print(f"Model: {best_overall[0][0]}\nResampling Method: {best_overall[0][1]}\nBest Hyperparameters: {best_overall[1]['best_params']['model_params']}\nBest F1 Score: {best_overall[1]['scores']['f1']}")

# %% Final step: Extract the best resampling method and model parameters
best_resampling_method = best_overall[0][1]  # Best resampling method found
best_model_params = best_overall[1]['best_params']['model_params']  # Best model parameters found
best_sampler_params = best_overall[1]['best_params'].get('sampler_params', {})  # Best sampler params, default to {} if None

# Initialize Random Forest model with best parameters
rf_final = RandomForestClassifier(random_state=42, **best_model_params)

# Initialize the resampling method (if any)
if best_resampling_method != 'None':  # If resampling is selected
    resampler_classes = {
        'RandomOverSampler': RandomOverSampler,
        'SMOTE': SMOTE,
        'RandomUnderSampler': RandomUnderSampler,
        'SMOTE-Tomek': SMOTETomek,
        'ADASYN': ADASYN
    }

    # Get the correct resampler class
    resampler_class = resampler_classes.get(best_resampling_method)

    # Initialize the resampler with best sampler params
    resampler = resampler_class(**best_sampler_params)

    # Apply resampling if necessary
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
else:
    X_resampled, y_resampled = X_train, y_train  # No resampling, use original data

# Train the Random Forest model on the (possibly resampled) data
rf_final.fit(X_resampled, y_resampled)

# Extract feature importances
importances = rf_final.feature_importances_

# Sort importances in descending order
indices = np.argsort(importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importance - Random Forest (Best Model)")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

# Optionally, extract and display the feature importance values in a DataFrame
importance_df = pd.DataFrame(list(zip(X_train.columns, importances[indices])),
                             columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:\n", importance_df)

# %% SHAP feature importance
# SHAP value calculation
explainer = shap.TreeExplainer(rf_final)  # Create SHAP explainer for Random Forest

# Calculate SHAP values
shap_values = explainer.shap_values(X_resampled)  # Calculate SHAP values for the resampled training data

# Check the shapes again
print(f"Shape of SHAP values (class 0): {shap_values[0].shape}")
print(f"Shape of SHAP values (class 1): {shap_values[1].shape}")
print(f"Shape of X_resampled: {X_resampled.shape}")

# Visualize the feature importance using SHAP (for the positive class, i.e., "aggregator")
# Ensure that shap_values[1] matches the shape of X_resampled
shap.summary_plot(shap_values[:,:,1], X_resampled)  # The `[1]` is for the positive class (aggregator) if you're dealing with binary classification

# Feature importance for the Random Forest model using SHAP
# If you want to just extract feature importance values from SHAP
importance_df = pd.DataFrame(list(zip(X_resampled.columns, np.mean(np.abs(shap_values[1]), axis=0))),
                             columns=['Feature', 'Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print feature importance
print("SHAP Feature Importance:\n", importance_df)

# %% Test evaluation
# Defie test data lol
X_test = test.drop(columns=['set'])  # Drop target column to get features
y_test = test['set']  # Target column
# Evaluate the model on the test set (specifically for aggregator class)
y_pred = rf_final.predict(X_test)  # Get predictions on the test set
y_pred_proba = rf_final.predict_proba(X_test)[:, 1]  # Get predicted probabilities for ROC AUC

# Performance Metrics based on the aggregator class (positive class = 1)
test_accuracy = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)  # Aggregator class (1)
test_recall = recall_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)  # Aggregator class (1)
test_f1 = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)  # Aggregator class (1)
test_roc_auc = roc_auc_score(y_test, y_pred_proba)  # ROC AUC for the positive class

# Print the performance metrics
print("\nTest Set Evaluation Metrics for Aggregator Class:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"ROC AUC: {test_roc_auc:.4f}")

# Confusion Matrix (based on aggregator class)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Soluble", "Aggregator"], yticklabels=["Soluble", "Aggregator"])
plt.title("Confusion Matrix - Test Set (Aggregator Class)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve Plot for the Aggregator Class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % test_roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve (Aggregator Class)')
plt.legend(loc="lower right")
plt.show()

# %% Estimate of spread in performance
## SPECIFY EACH OF THE MODELS
# model 1 (surface features)
model1_data = data[['thsa_netsurfp2', 'rhsa_netsurfp2', 'Gravy', 'LHPpred', 'C_exposed']].copy()
model1_data['positive_surface'] = data[['R_exposed', 'H_exposed', 'K_exposed']].sum(axis=1)
model1_data['negative_surface'] = data[['D_exposed', 'E_exposed']].sum(axis=1)
model1_data['nonpolar_surface'] = data[['A_exposed', 'V_exposed', 'I_exposed', 'L_exposed', 'M_exposed', 'F_exposed', 'Y_exposed', 'W_exposed', 'P_exposed', 'G_exposed']].sum(axis=1)
model1_data['polar_surface'] = data[['S_exposed', 'T_exposed', 'N_exposed', 'Q_exposed']].sum(axis=1)
model1_data['LHP_norm'] = data['LHPpred']/data['molecular_weight']
# Encode labels: soluble -> 0, aggregator -> 1
model1_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})

# model 2 (stability)
model2_data = data[['length', 'helix', 'sheet', 'disorder', 'Instability_index', 'charge_at_5', 'charge_at_7']].copy()
model2_data['charge_change'] = data['charge_at_7'] - data['charge_at_5']
model2_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})

# model 3 (Surface features + Stability)
model3_data = data[['thsa_netsurfp2', 'rhsa_netsurfp2', 'Gravy', 'LHPpred', 'C_exposed', 'length', 'helix', 'sheet', 'disorder', 'Instability_index','charge_at_5', 'charge_at_7']].copy()
model3_data['positive_surface'] = data[['R_exposed', 'H_exposed', 'K_exposed']].sum(axis=1)
model3_data['negative_surface'] = data[['D_exposed', 'E_exposed']].sum(axis=1)
model3_data['nonpolar_surface'] = data[['A_exposed', 'V_exposed', 'I_exposed', 'L_exposed', 'M_exposed', 'F_exposed', 'Y_exposed', 'W_exposed', 'P_exposed', 'G_exposed']].sum(axis=1)
model3_data['polar_surface'] = data[['S_exposed', 'T_exposed', 'N_exposed', 'Q_exposed']].sum(axis=1)
model3_data['LHP_norm'] = data['LHPpred']/data['molecular_weight']
model3_data['charge_change'] = data['charge_at_7'] - data['charge_at_5']
model3_data['set'] = data['set'].map({'soluble': 0, 'aggregator': 1})

## MAKE THE TEST TRAIN SPLIT FOR EACH MODEL
train1, test1 = train_test_split(
    model1_data,
    test_size=1000,  # 500 instances in the test set (can change this)
    stratify=model1_data['set'],  # Preserve class distribution
    random_state=42  # For reproducibility
)
# exclude Y for training
X_train1 = train1.drop(columns=['set'])  # Drop the target column to get features + solubility
y_train1 = train1['set']  # Target column
X_test1 = test1.drop(columns=['set'])  # Drop target column to get features
y_test1 = test1['set']

train2, test2 = train_test_split(
    model2_data,
    test_size=1000,  # 500 instances in the test set (can change this)
    stratify=model2_data['set'],  # Preserve class distribution
    random_state=42  # For reproducibility
)

# exclude Y for training
X_train2 = train2.drop(columns=['set'])  # Drop the target column to get features + solubility
y_train2 = train2['set']  # Target column
X_test2 = test2.drop(columns=['set'])  # Drop target column to get features
y_test2 = test2['set']

train3, test3 = train_test_split(
    model3_data,
    test_size=1000,  # 500 instances in the test set (can change this)
    stratify=model3_data['set'],  # Preserve class distribution
    random_state=42  # For reproducibility
)

# exclude Y for training
X_train3 = train3.drop(columns=['set'])  # Drop the target column to get features + solubility
y_train3 = train3['set']  # Target column
X_test3 = test3.drop(columns=['set'])  # Drop target column to get features
y_test3 = test3['set']

## SET RANDOM FOREST FOR EACH MODEL (BASED ON HYPERPARAMETER FILE FROM MARLENY)
rf_model1 = RandomForestClassifier(max_depth=10, min_samples_split=5, n_estimators=100)
rf_model2 = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200)
rf_model3 = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200)

## PERFORM THE RESAMPLING FOR THE TRAINING SET
X_resampled1, y_resampled1 = RandomOverSampler(sampling_strategy=1.0).fit_resample(X_train1, y_train1)
X_resampled2, y_resampled2 = RandomUnderSampler(sampling_strategy=0.5).fit_resample(X_train2, y_train2)
X_resampled3, y_resampled3 = RandomUnderSampler(sampling_strategy=0.5).fit_resample(X_train3, y_train3)

N = 5
F1_scores = {'Model1': {}, 'Model2': {}, 'Model3': {}}
Precision = {'Model1': {}, 'Model2': {}, 'Model3': {}}
Recall = {'Model1': {}, 'Model2': {}, 'Model3': {}}


for i in range(0, N):
  rf_model1.fit(X_resampled1, y_resampled1)
  y_pred1 = rf_model1.predict(X_test1)
  test_f1_1 = f1_score(y_test1, y_pred1, pos_label=1, average='binary', zero_division=0)
  test_precision1 = precision_score(y_test1, y_pred1, pos_label=1, average='binary', zero_division=0)
  test_recall1 = recall_score(y_test1, y_pred1, pos_label=1, average='binary', zero_division=0)
  F1_scores['Model1'][i]=test_f1_1
  Precision['Model1'][i]=test_precision1
  Recall['Model1'][i]=test_recall1

  rf_model2.fit(X_resampled2, y_resampled2)
  y_pred2 = rf_model2.predict(X_test2)
  test_f1_2 = f1_score(y_test2, y_pred2, pos_label=1, average='binary', zero_division=0)
  test_precision2 = precision_score(y_test2, y_pred2, pos_label=1, average='binary', zero_division=0)
  test_recall2 = recall_score(y_test2, y_pred2, pos_label=1, average='binary', zero_division=0)
  F1_scores['Model2'][i]=test_f1_2
  Precision['Model2'][i]=test_precision2
  Recall['Model2'][i]=test_recall2

  rf_model3.fit(X_resampled3, y_resampled3)
  y_pred3 = rf_model3.predict(X_test3)
  test_f1_3 = f1_score(y_test3, y_pred3, pos_label=1, average='binary', zero_division=0)
  test_precision3 = precision_score(y_test3, y_pred3, pos_label=1, average='binary', zero_division=0)
  test_recall3 = recall_score(y_test3, y_pred3, pos_label=1, average='binary', zero_division=0)
  F1_scores['Model3'][i]=test_f1_3
  Precision['Model3'][i]=test_precision3
  Recall['Model3'][i]=test_recall3

  explainer = shap.TreeExplainer(rf_model3)
  shap_values = explainer.shap_values(X_resampled3)
  shap.summary_plot(shap_values[:, :, 1], X_resampled3) # check if feature importance changes

print('F1_scores', F1_scores)
print('Precision', Precision)
print('Recall', Recall)

# %% Additional plots for feature importance interpretation.
explainer = shap.TreeExplainer(rf_model3)
shap_values = explainer.shap_values(X_test3)

#shap.dependence_plot('disorder', shap_values[:, :, 1], X_test3)
shap.summary_plot(shap_values[:, :, 1], X_test3, show=False)
plt.savefig("shap_summary.png",dpi=700, bbox_inches="tight")
explanation = explainer(X_test3)
shap.plots.scatter(explanation[:, "disorder", 1], show=False)
plt.savefig("shap_disorder.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here
shap.plots.scatter(explanation[:, "charge_at_5", 1], show=False)
plt.savefig("shap_charge5.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here
shap.plots.scatter(explanation[:, "thsa_netsurfp2", 1], show=False)
plt.savefig("shap_thsa.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here
shap.plots.scatter(explanation[:, "Gravy", 1], show=False)
plt.savefig("shap_gravy.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here

# %% Difference plot for features misclassified examples
# Identify False Negatives (actual 1, predicted 0) and True Positives (actual 1, predicted 1)
FN_indices = (y_test3.values == 1) & (y_pred3 == 0)
TP_indices = (y_test3.values == 1) & (y_pred3 == 1)
FP_indices = (y_test3.values == 0) & (y_pred3 == 1)
TN_indices = (y_test3.values == 0) & (y_pred3 == 0)

X_test_FNs = X_test3[FN_indices]
X_test_TPs = X_test3[TP_indices]
X_test_FPs = X_test3[FP_indices]
X_test_TNs = X_test3[TN_indices]

## Compute mean difference for selected features: FNs vs TPs

Xt_FNs = X_test_FNs[['disorder', 'charge_at_5', 'Gravy', 'thsa_netsurfp2', 'Instability_index']].copy()
Xt_TPs = X_test_TPs[['disorder', 'charge_at_5', 'Gravy', 'thsa_netsurfp2', 'Instability_index']].copy()

fn_mean = Xt_FNs.mean()
tp_mean = Xt_TPs.mean()

comparison_df = pd.DataFrame({'FN_mean': fn_mean, 'TP_mean': tp_mean})
comparison_df['Difference'] = comparison_df['FN_mean'] - comparison_df['TP_mean']

plt.figure(figsize=(10, 6))
comparison_df['Difference'].head(10).plot(kind='barh', color=['red' if x < 0 else 'blue' for x in comparison_df['Difference'].head(10)])
plt.xlabel('Difference in Mean Value')
plt.ylabel('Feature')
plt.title('Top Features Differing Between False Negatives and True Positives')
plt.gca().invert_yaxis()
plt.savefig("dif_fn_minus_tp.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here

print(comparison_df['Difference'])
print(comparison_df)

## Compute mean difference for selected features: FPs vs TNs

Xt_FPs = X_test_FPs[['disorder', 'charge_at_5', 'Gravy', 'thsa_netsurfp2', 'Instability_index']].copy()
Xt_TNs = X_test_TNs[['disorder', 'charge_at_5', 'Gravy', 'thsa_netsurfp2', 'Instability_index']].copy()

fp_mean = Xt_FPs.mean()
tn_mean = Xt_TNs.mean()

comparison_df = pd.DataFrame({'FP_mean': fp_mean, 'TN_mean': tn_mean})
comparison_df['Difference'] = comparison_df['FP_mean'] - comparison_df['TN_mean']

plt.figure(figsize=(10, 6))
comparison_df['Difference'].head(10).plot(kind='barh', color=['red' if x < 0 else 'blue' for x in comparison_df['Difference'].head(10)])
plt.xlabel('Difference in Mean Value')
plt.ylabel('Feature')
plt.title('Top Features Differing Between False Positives and True Negatives')
plt.gca().invert_yaxis()
plt.savefig("dif_fp_minus_tn.png",dpi=700, bbox_inches="tight") #.png,.pdf will also support here

print(comparison_df['Difference'])
print(comparison_df)