# Multiclass Logistic Regression with Iris Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import seaborn as sn
from matplotlib import pyplot as plt

# Load the Iris dataset
data = pd.read_csv('Iris.csv')

# Display the first few rows of the dataset
print("Iris Dataset Head:")
print(data.head())

# Display the unique species
print("\nUnique Species in the dataset:")
print(data['Species'].unique())

# Replace the species names with numeric values for classification
data['Species'] = data['Species'].replace({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3})

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, Y_train, Y_test = train_test_split(
    data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']], 
    data['Species'], 
    train_size=0.2
)

# Print training and test sets
print("\nTraining Features (X_train):")
print(X_train)
print("\nTraining Labels (Y_train):")
print(Y_train)

# Initialize and train the logistic regression model
mymodel = linear_model.LogisticRegression(max_iter=120)
mymodel.fit(X_train, Y_train)

# Predict the labels for the test set
predicted_output = mymodel.predict(X_test)

# Print the test set and the predictions
print("\nTest Features (X_test):")
print(X_test)
print("\nPredicted Output:")
print(predicted_output)

# Evaluate the model's accuracy
score = mymodel.score(X_test, Y_test)
print(f"\nModel Accuracy: {score * 100:.2f}%")

# Generate a confusion matrix
cm = confusion_matrix(Y_test, predicted_output)

# Print the confusion matrix
print("\nConfusion Matrix:")
print(cm)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(5, 4))
sn.heatmap(cm, annot=True, cmap="Blues", fmt='g')
plt.xlabel('Predicted Value')
plt.ylabel('Truth or Actual Value')
plt.title('Confusion Matrix for Multiclass Logistic Regression')
plt.show()
