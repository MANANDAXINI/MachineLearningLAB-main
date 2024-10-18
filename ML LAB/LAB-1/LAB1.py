import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('ex1data1.csv', names=['population', 'profit'])
print("First few rows of the dataset:")
print(data.head())

# Plotting the data
data.plot(kind='scatter', x='population', y='profit', figsize=(8, 8))
plt.xlabel('Population (in 10,000s)')
plt.ylabel('Profit (in $10,000s)')
plt.title('Population vs. Profit')
plt.show()

# Adding a column of ones for the intercept term
data.insert(0, 'ones', 1)

# Prepare the data for regression
columns = data.shape[1]
X = data.iloc[:, 0:columns - 1]
y = data.iloc[:, columns - 1:columns]

# Convert data to numpy matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0, 0]))

# Cost function
def compute_cost(X, y, theta):
    error = np.power((X * theta.T - y), 2)
    return np.sum(error) / (2 * len(X))

# Gradient Descent function
def gradient_descent(X, y, theta, alpha, iterations):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost_history[i] = compute_cost(X, y, theta)

    return theta, cost_history

# Parameters for Gradient Descent
alpha = 0.01
iterations = 1500

# Compute the cost before running gradient descent
initial_cost = compute_cost(X, y, theta)
print(f"Initial cost: {initial_cost}")

# Running gradient descent
theta_final, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# Output the final theta values and the final cost
print(f"Theta after gradient descent: {theta_final}")
print(f"Final cost: {cost_history[-1]}")

# Plot the linear fit on the data
x_vals = np.linspace(data.population.min(), data.population.max(), 100)
y_vals = theta_final[0, 0] + (theta_final[0, 1] * x_vals)

plt.figure(figsize=(8, 8))
plt.plot(data.population, data.profit, 'rx', label='Training data')
plt.plot(x_vals, y_vals, 'b-', label='Linear regression fit')
plt.xlabel('Population (in 10,000s)')
plt.ylabel('Profit (in $10,000s)')
plt.title('Linear Regression Fit')
plt.legend(loc='best')
plt.show()

# Plotting the cost function over the iterations
plt.plot(np.arange(iterations), cost_history, 'r')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.show()
