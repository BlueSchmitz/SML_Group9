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
from sklearn.linear_model import LinearRegression, LogisticRegression

# %% Linear regression
# load the data
data = pd.read_csv('/aggr_sol_dataset.csv')
data = data.drop(2478, axis='index')
data = data[data['TMHMM'] != 1]
data = data.drop(columns = ['Gene', 'Gene_name', 'Uniprot_ID', 'fasta_sequence'])
data.shape
np.random.seed(42)
random.seed(42)

# Separate train and test data using binning
num_bins = 10
data['y_binned'] = pd.qcut(data['solubility'], q=num_bins, duplicates='drop')
train, test = train_test_split(
    data,
    test_size=1000,
    stratify=data['y_binned'],
    random_state=42
)

# Prepare data
train = train.drop(columns=['y_binned'])  # Remove binning column
test = test.drop(columns=['y_binned'])

X_train = train.drop(columns=['solubility', 'set'])
y_train = train['solubility']

X_test = test.drop(columns=['solubility', 'set'])
y_test = test['solubility']

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R squared
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE is the square root of MSE
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f'Mean Squared Error (MSE): {mse:.6f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.6f}')
print(f'Mean Absolute Error (MAE): {mae:.6f}')
print(f'R² Score: {r2:.6f}')

# %% Logistic regression
# import data
data = pd.read_csv('/aggr_sol_dataset.csv')
data = data.drop(2478, axis='index')
data = data[data['TMHMM'] != 1]
data = data.drop(columns = ['Gene', 'Gene_name', 'Uniprot_ID', 'fasta_sequence'])
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import seaborn as sns

# Separate train and test data
train, test = train_test_split(
    data,
    test_size=1000,  # 500 instances in the test set (can change this)
    stratify=data['set'],  # Preserve class distribution
    random_state=42  # For reproducibility
)
# exclude Y for training
X_train = train.drop(columns=['set', 'solubility'])  # Drop the target column to get features + solubility
y_train = train['set']  # Target column

X_test = test.drop(columns=['solubility', 'set'])
y_test = test['set']

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print full classification report (includes precision, recall, F1 for all classes)
print("Classification Report:")
print(classification_report(y_test, y_pred))
# Get the classification report as a dictionary
report = classification_report(y_test, y_pred, output_dict=True)

# Extract the F1 score for the "Aggregator" class (assuming it's labeled 'Aggregator')
aggregator_f1 = report['aggregator']['f1-score']
aggregator_precision = report['aggregator']['precision']
aggregator_recall = report['aggregator']['recall']
# Print the F1 score with more decimal places
print(f"F1 Score for Aggregator: {aggregator_f1:.6f}")
print(f"precision Score for Aggregator: {aggregator_precision:.6f}")
print(f"recall Score for Aggregator: {aggregator_recall:.6f}")
# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_swapped = cm[::-1, ::-1]  # Reverse the rows and columns of the cm

plt.figure(figsize=(6, 5))
sns.heatmap(cm_swapped, annot=True, fmt='d', cmap='Blues', xticklabels=['Soluble', 'Aggregator'], yticklabels=['Soluble', 'Aggregator'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()