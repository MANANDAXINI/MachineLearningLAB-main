# Support Vector Regression (SVR) on Position Salaries Dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
dataset = pd.read_csv("Position_Salaries.csv")

# Display basic information about the dataset
print("First 5 rows of the dataset:")
print(dataset.head())
print("\nLast 5 rows of the dataset:")
print(dataset.tail())

# Dataset columns and shape
print("\nColumns in the dataset:")
print(dataset.columns)
print("\nShape of the dataset:")
print(dataset.shape)

# Statistical description of the dataset
print("\nDataset description:")
print(dataset.describe())

# Check for missing values
print("\nAny missing data or NaN in the dataset:")
print(dataset.isnull().values.any())

# Dataset info
print("\nDataset Info:")
print(dataset.info())

# Extracting features and target variable
X = dataset.iloc[:, 1:2].values  # Position level (independent variable)
Y = dataset.iloc[:, 2].values    # Salary (dependent variable)

# Reshaping Y to fit the StandardScaler
Y = Y.reshape(-1, 1)

# Feature Scaling using StandardScaler
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)

# Training the SVR model with RBF kernel
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, Y.ravel())  # Using ravel to flatten Y into 1D array

# Predicting a new result with SVR for position level 6.5
Y_pred = regressor.predict(sc_X.transform([[6.5]]))  # Transforming input data before prediction
Y_pred = Y_pred.reshape(-1, 1)
Y_pred = sc_Y.inverse_transform(Y_pred)  # Inverse transform to get the actual salary
print(f"\nPredicted Salary for Position Level 6.5: {Y_pred[0][0]}")

# Visualizing the SVR results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the SVR results with a higher resolution for smoother curve
X_grid = np.arange(min(X), max(X), 0.01)  # Smoother range for X values
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or Bluff (SVR) - High Resolution')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
