import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from prettytable import PrettyTable

# Read Iris dataset
iris = pd.read_csv('IRIS.csv')

# features and target variable
X = iris.drop("species", axis=1)
y = iris["species"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the K-nearest neighbors classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# predictions
y_pred = knn.predict(X_test)

# Printing confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# New csv file is create to predict data 
print("\n\n Prediction for new data \n")
new_data = pd.read_csv("iris_new_data.csv").values

# Scale the new data 
new_data_scaled = scaler.transform(new_data)

# Predict the classes for the new data points
predicted_classes = knn.predict(new_data_scaled)




table = PrettyTable()

# defining table
table.field_names = ["Data Point", "Predicted Class"]
table.align["Data Point"] = "l"
table.align["Predicted Class"] = "l"

# adding data(records ) to table
for i, predicted_class in enumerate(predicted_classes):
    table.add_row([f"New Data Point {i + 1}", predicted_class])

# Printing the table
print(table)
