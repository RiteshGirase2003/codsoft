import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:100]  # Only the first 100 records
y = iris.target[:100]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the K-nearest neighbors classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Display the confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sample prediction on a new data point
# Replace the values with your own data to predict a class


new_data_points = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 3.0, 4.8, 1.8],
    [5.8, 2.7, 3.9, 1.2],
    [7.2, 3.2, 1.0, 0.3],
    [5.5, 2.8, 4.5, 1.3],
    [6.3, 2.9, 5.6, 1.8],
    [6.7, 3.1, 4.7, 1.5],
    [6.1, 2.9, 4.7, 1.4],
    [6.5, 3.0, 5.2, 2.0],
    [6.4, 2.7, 5.3, 1.9],
    [5.7, 2.6, 3.5, 1.0]
])

# Loop through each new data point and predict its class
for i, new_data_point in enumerate(new_data_points):
    new_data_point_scaled = scaler.transform([new_data_point])
    predicted_class = knn.predict(new_data_point_scaled)
    print(f"Predicted class for new data point {i + 1}: {iris.target_names[predicted_class][0]}")

print("--------------------------------")
new_data_points = pd.read_csv("iris_new_data.csv").values

# Scale the new data points
new_data_points_scaled = scaler.transform(new_data_points)

# Predict the classes for the new data points
predicted_classes = knn.predict(new_data_points_scaled)

# Display the predicted classes for the new data points
for i, predicted_class in enumerate(predicted_classes):
    print(f"Predicted class for new data point {i + 1}: {predicted_class}")
