import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, ConfusionMatrixDisplay, roc_curve)

import matplotlib
matplotlib.use("Qt5Agg")  # Forces non-GUI backend that doesn’t need tkinter
# Load the dataset
data_path = r'c:\Program Files\diabetes.csv'
try:
    data = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: The file at {data_path} was not found.")
    exit()

# Displays the first few rows of the dataset
print(data.head())

# Displays the summary of the dataset
print(data.describe())

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Check if 'Outcome' column exists
if 'Outcome' not in data.columns:
    print("Error: 'Outcome' column not found in the dataset.")
    exit()

# Standardize the data
x = data.drop('Outcome', axis=1)
y = data['Outcome']

# Check for empty DataFrame
if x.empty or y.empty:
    print("Error: Feature set or target variable is empty.")
    exit()

scaler = StandardScaler()
x_scaled = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)  # Ensure this is a DataFrame

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Initialize the model
model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence

# Train the model
model.fit(x_train, y_train)

# Make the predictions
y_pred = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Confusion matrix
print('Confusion Matrix:')
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Check if grid search was successful
if grid_search.best_estimator_ is None:
    print("Error: GridSearchCV did not find a suitable model.")
    exit()

# Best parameter and best score
print('Best Parameters:', grid_search.best_params_)
print('Best Cross-Validation Score:', grid_search.best_score_)

# Using the best estimator from the grid search 
best_model = grid_search.best_estimator_

# Predictions using the best model
y_pred_best = best_model.predict(x_test)

# Final accuracy
final_accuracy = accuracy_score(y_test, y_pred_best)
print(f'Final Accuracy: {final_accuracy * 100:.2f}%')

# Calculate ROC-AUC Score
roc_auc = roc_auc_score(y_test, best_model.predict_proba(x_test)[:, 1])
print(f'ROC-AUC Score: {roc_auc:.2f}')

# Confusion Matrix Visualization
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
plt.title('Confusion Matrix Visualization')
plt.show()


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:, 1])
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC Curve')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Feature Importance Analysis
importance = best_model.coef_[0]
feature_importance = pd.DataFrame({'Feature': x.columns, 'Importance': importance})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()


# Correlation heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# Pair plot
sns.pairplot(data, hue='Outcome', diag_kind='kde')
plt.title('Pair Plot of Features')
plt.show()


# Save the model
joblib.dump(best_model, 'diabetes_logistic_model.pkl')
print("Model saved as 'diabetes_logistic_model.pkl'.")

# Save the scaler
joblib.dump(scaler, 'diabetes_scaler.pkl')
print("Scaler saved as 'diabetes_scaler.pkl'.")

# Load the model for future use
loaded_model = joblib.load('diabetes_logistic_model.pkl')

# Load the scaler
loaded_scaler = joblib.load('diabetes_scaler.pkl')

# Test loading the scaler
# Create a sample DataFrame similar to your original features
sample_data = pd.DataFrame([[2, 120, 70, 30, 85, 33.6, 0.627, 50]], 
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 
                                     'SkinThickness', 'Insulin', 'BMI', 
                                     'DiabetesPedigreeFunction', 'Age'])

# Scale the sample data using the loaded scaler
scaled_sample_data = loaded_scaler.transform(sample_data)

# Convert back to DataFrame for readability
scaled_sample_data_df = pd.DataFrame(scaled_sample_data, columns=sample_data.columns)

# Display the scaled sample data
print("Scaled Sample Data:")
print(scaled_sample_data_df)

# Make predictions on new data using the loaded model
new_data = pd.DataFrame([[2, 120, 70, 30, 85, 33.6, 0.627, 50]], columns=x.columns)  # Example new data as DataFrame
new_data_scaled = loaded_scaler.transform(new_data)  # Scale the new data
new_prediction = loaded_model.predict(new_data_scaled)
print(f"Prediction for the new sample: {'Diabetic' if new_prediction[0] == 1 else 'Non-Diabetic'}")

import os
print("Current Working Directory:", os.getcwd())
