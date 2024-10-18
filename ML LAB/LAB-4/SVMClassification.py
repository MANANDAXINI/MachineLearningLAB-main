# SVM Classification with Grid Search on Parkinson's Disease Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Load the dataset
dataset = pd.read_csv('parkinsons_new.csv')

# Display the first and last few rows of the dataset
print("First 5 rows of the dataset:")
print(dataset.head())
print("\nLast 5 rows of the dataset:")
print(dataset.tail())

# Dataset shape
print("\nShape of the dataset:")
print(dataset.shape)

# Dataset columns and info
print("\nDataset columns:")
print(dataset.columns)
print("\nDataset info:")
print(dataset.info())

# Statistical description of the dataset
print("\nDataset description:")
print(dataset.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(pd.isnull(dataset).sum())

# Distribution of classes in the 'status' column
classes = dataset['status'].value_counts()
print("\nDistribution of status (class labels):")
print(classes)

# Plot the class distribution
classes.plot.bar()
plt.title('Class Distribution (Parkinsons Status)')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Convert categorical columns to numerical if necessary
dataset = pd.get_dummies(dataset, prefix_sep='sex')

# Correlation matrix
corr_var = dataset.corr()
print("\nCorrelation matrix:")
print(corr_var)

# Heatmap of the correlation matrix
plt.figure(figsize=(20, 17.5))
sns.heatmap(corr_var, annot=True, cmap='BuPu')
plt.title('Correlation Heatmap')
plt.show()

# Separate features and target variable
X = dataset.loc[:, dataset.columns != "status"]
Y = dataset["status"]

# Display feature set and target
print("\nFeatures (X):")
print(X.head())
print("\nTarget (Y):")
print(Y.head())

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print("\nDimensions of training and test sets:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("Y_train:", Y_train.shape)
print("Y_test:", Y_test.shape)

# Standardize the features
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# Train the SVM classifier with linear kernel
cl = svm.SVC(kernel='linear', C=0.01)
cl.fit(X_train, Y_train)

# Predict using the model
Y_pred_train = cl.predict(X_train)
Y_pred_test = cl.predict(X_test)

# Display predictions on test data
print("\nPredictions on test data:")
print(Y_pred_test)

# Confusion matrix and accuracy score
cm = confusion_matrix(Y_test, Y_pred_test)
print("\nConfusion Matrix:")
print(cm)

acc = accuracy_score(Y_test, Y_pred_test)
print(f"\nAccuracy Score: {acc:.4f}")

# Perform grid search to optimize hyperparameters
parameters = {
    'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000],
    'degree': [2, 3, 4, 5],
    'gamma': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['rbf', 'poly']
}
cl = svm.SVC()
grid = GridSearchCV(cl, parameters, cv=10)
grid.fit(X_train, Y_train)

# Print the best parameters and best estimator
print("\nBest parameters found by GridSearch:")
print(grid.best_params_)
print("\nBest estimator found by GridSearch:")
print(grid.best_estimator_)

# Predict using the best model
grid_prediction = grid.predict(X_test)

# Print the classification report for the grid search model
print("\nClassification Report:")
print(classification_report(Y_test, grid_prediction))

