# Logistic regression classification with ROC curve plotting

import pandas as pd
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Generate a synthetic dataset for binary classification
x, y = make_classification(
    n_samples=300,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)

# Create a scatter plot of the generated dataset
plt.scatter(x, y, c=y, cmap='rainbow')
plt.title('Scatter plot of Logistic Regression dataset')
plt.show()

# Split the dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

# Train a logistic regression model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

# Print the model coefficients and intercept
print(f"Model Coefficients: {log_reg.coef_}")
print(f"Model Intercept: {log_reg.intercept_}")

# Predict the test set
y_pred = log_reg.predict(x_test)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Create a DataFrame showing actual vs predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nActual vs Predicted:")
print(df)

# Create another DataFrame for the training data
df_train = pd.DataFrame({'x_train': x_train.flatten(), 'y_train': y_train})
print("\nTraining Data:")
print(df_train)

# Generate ROC curve data
y_pred_proba = log_reg.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
