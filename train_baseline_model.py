import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load cleaned dataset
df = pd.read_csv("C:/Users/angelica.ginige/Desktop/Script/predictive-maintenance/predictive_maintenance_cleaned.csv")

# Display dataset info
print(df.info())
df.head()

#Define features and target
# Select feature columns (excluding non-relevant ones)
X = df.drop(columns=['UDI', 'Product ID', 'Type', 'Failure Type', 'failure_label'])

# Define target variable
y = df['failure_label']

# Handle missing values
X.fillna(X.median(), inplace=True)  # Fill NaNs with median values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Train baseline models1
# Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=500)
log_reg.fit(X_train, y_train)
# Predictions
y_pred_log = log_reg.predict(X_test)
# Evaluate Performance
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))


#Train baseline models2(strongest baseline)
# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Predictions
y_pred_rf = rf_model.predict(X_test)
# Evaluate Performance
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#Confusion Matrix(visualize the performance of the model)
plt.figure(figsize=(5,5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.show()


#Generate Probabilities for ROC & PR Curves
# Get predicted probabilities
y_prob_log = log_reg.predict_proba(X_test)[:, 1]  # Logistic Regression
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]  # Random Forest

# Compute ROC curves
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# Compute Precision-Recall curves
prec_log, recall_log, _ = precision_recall_curve(y_test, y_prob_log)
prec_rf, recall_rf, _ = precision_recall_curve(y_test, y_prob_rf)

# Plot ROC & Precision-Recall Curves
plt.figure(figsize=(12,5))

# ROC Curve
plt.subplot(1,2,1)
plt.plot(fpr_log, tpr_log, label="Logistic Regression (AUC = {:.2f})".format(auc(fpr_log, tpr_log)))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(auc(fpr_rf, tpr_rf)))
plt.plot([0,1], [0,1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Precision-Recall Curve
plt.subplot(1,2,2)
plt.plot(recall_log, prec_log, label="Logistic Regression")
plt.plot(recall_rf, prec_rf, label="Random Forest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()


#Hyperparameter Tuning for Random Forest
# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#Perform Grid Search
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Print best parameters
print("Best Parameters:", grid_search.best_params_)

#Train with Best Parameters

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate Best Model
print("Best Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print(classification_report(y_test, y_pred_best_rf))




# Save the trained model
model_path = "C:/Users/angelica.ginige/Desktop/Script/predictive-maintenance/best_model.pkl"
with open(model_path, "wb") as model_file:
    pickle.dump(best_rf, model_file)

print("✅ Model saved successfully!")

print("Model expects these features:", list(X.columns))


