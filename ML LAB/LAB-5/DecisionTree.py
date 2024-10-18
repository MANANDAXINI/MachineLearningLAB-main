# Decision Tree Classification on Car Evaluation Dataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import category_encoders as ce
from sklearn import tree
import graphviz

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv('car_evaluation.csv')

# Display basic information about the dataset
print("Dataset shape:", data.shape)
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Define column names
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
data.columns = col_names

# Display column names and dataset info
print("\nColumn names:", col_names)
print("\nDataset info:")
data.info()

# Display value counts for each column
for col in col_names:
    print(f"\nValue counts for {col}:")
    print(data[col].value_counts())

# Check for missing values
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Split data into features and target variable
X = data.drop(['class'], axis=1)
Y = data['class']

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Display shapes of the training and test sets
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Check the data types of the training set
print("\nData types in training set:")
print(X_train.dtypes)

# Encoding categorical variables
encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

# Display encoded training and test sets
print("\nEncoded training set:")
print(X_train.head())
print("\nEncoded test set:")
print(X_test.head())

# Train Decision Tree Classifier with Gini index
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, Y_train)

# Make predictions and calculate accuracy
Y_pred_gini = clf_gini.predict(X_test)
print('Model accuracy score with criterion Gini index: {0:0.4f}'.format(accuracy_score(Y_test, Y_pred_gini)))

# Predictions on training set
Y_pred_train_gini = clf_gini.predict(X_train)
print('Training set accuracy score: {0:0.4f}'.format(accuracy_score(Y_train, Y_pred_train_gini)))

# Training and test set scores
print('Training set score: {:.4f}'.format(clf_gini.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(clf_gini.score(X_test, Y_test)))

# Visualizing the Decision Tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_gini.fit(X_train, Y_train))
plt.show()

# Exporting the decision tree to Graphviz format
dot_data = tree.export_graphviz(clf_gini, out_file=None, feature_names=X_train.columns, class_names=Y_train.unique(),
                                 filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("gini_tree")  # Save tree visualization

# Train Decision Tree Classifier with Entropy
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, Y_train)

# Predictions and accuracy with Entropy
Y_pred_en = clf_en.predict(X_test)
print('Model accuracy score with criterion Entropy: {0:0.4f}'.format(accuracy_score(Y_test, Y_pred_en)))

# Predictions on training set with Entropy
Y_pred_train_en = clf_en.predict(X_train)
print('Training set accuracy score: {0:0.4f}'.format(accuracy_score(Y_train, Y_pred_train_en)))

# Training and test set scores for Entropy
print('Training set score: {:.4f}'.format(clf_en.score(X_train, Y_train)))
print('Test set score: {:.4f}'.format(clf_en.score(X_test, Y_test)))

# Visualizing the Decision Tree for Entropy
plt.figure(figsize=(12, 8))
tree.plot_tree(clf_en.fit(X_train, Y_train))
plt.show()

# Exporting the decision tree to Graphviz format for Entropy
dot_data = tree.export_graphviz(clf_en, out_file=None, feature_names=X_train.columns, class_names=Y_train.unique(),
                                 filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("entropy_tree")  # Save tree visualization

# Confusion matrix and classification report
cm = confusion_matrix(Y_test, Y_pred_en)
print('Confusion Matrix:\n', cm)
print('\nClassification Report:\n', classification_report(Y_test, Y_pred_en))
