# Building-a-smarter-AI-powered-spam-classifier
Building a smarter AI-powered spam classifie
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (you would replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv('your_dataset.csv')

# Define the features (X) and target variable (y)
X = data[['feature1', 'feature2', 'feature3']]  # Replace with your actual features
y = data['target']  # Replace with your actual target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# You can now use the trained model to make predictions on new data
