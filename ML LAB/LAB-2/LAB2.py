import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('homeprices.csv')
print("Home Prices Dataset:")
print(df.head())

# Statistical description of the data
print("\nData Description:")
print(df.describe())

# Separate independent variable (area) and dependent variable (price)
x = df.drop('price', axis='columns')  # Independent variable (area)
y = df.price  # Dependent variable (price)

# Train the Linear Regression model
model = linear_model.LinearRegression()
model.fit(x, y)

# Make a prediction for a specific area (e.g., 3350 square feet)
predicted_price = model.predict([[3350]])
print(f"\nPredicted price for a 3350 sq.ft area: {predicted_price[0]}")

# Display model coefficients and intercept
print(f"Model Coefficient (Slope): {model.coef_[0]}")
print(f"Model Intercept: {model.intercept_}")

# Load a new dataset containing areas to predict their prices
area_df = pd.read_csv("areas.csv")
print("\nNew Areas Data:")
print(area_df.head())

# Predict prices for the new areas
p = model.predict(area_df)
print("\nPredicted Prices for New Areas:")
print(p)

# Add the predicted prices to the dataframe
area_df['predicted_price'] = p
print("\nAreas Data with Predicted Prices:")
print(area_df)

# Save the predictions to a new CSV file
area_df.to_csv("prediction.csv", index=False)
print("\nPredictions saved to 'prediction.csv'.")

# Plot the original data and the regression line
plt.xlabel('Area (sq.ft)')
plt.ylabel('Price (in $)')
plt.scatter(df.area, df.price, color='blue', marker='+')
plt.plot(df.area, model.predict(df[['area']]), color='red')

# Show the plot
plt.show()
