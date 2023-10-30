# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Load the Titanic dataset
# titanic_data=pd.read_csv('titanic.csv')

# # Display the first few rows of the dataset
# print(titanic_data.head())


# # Features and target variable
# X = titanic_data.drop('survived', axis=1)
# y = titanic_data['survived']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # Predict the target variable on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# print('Accuracy:', accuracy)
# print('Classification Report:\n', classification_report(y_test, y_pred))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset into a pandas DataFrame
df = pd.read_csv('tested.csv')

# Data Preprocessing
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Impute missing values in 'Age'
imputer = SimpleImputer(strategy='mean')
df["Age"] = imputer.fit_transform(df[["Age"]])

# Split the dataset into features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Feature Selection (You may choose different features based on your analysis)
selected_features = X.columns

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X[selected_features], y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---

unknown_data = pd.read_csv('new_titanic.csv')  # Replace with the appropriate file path

# Drop unnecessary columns and preprocess 'Sex' and 'Embarked'
unknown_data.drop(["Name", "Ticket", "Cabin"], axis=1, inplace=True)
unknown_data["Sex"] = unknown_data["Sex"].map({"male": 0, "female": 1})
unknown_data = pd.get_dummies(unknown_data, columns=["Embarked"], drop_first=True)

# Impute missing values in 'Age' for the unknown data
imputer = SimpleImputer(strategy='mean')
unknown_data["Age"] = imputer.fit_transform(unknown_data[["Age"]])

# Feature Selection (You may choose different features based on your analysis)
selected_features_unknown = unknown_data.columns

# Predict survival for the unknown data using the trained model
unknown_data_predictions = model.predict(unknown_data[selected_features_unknown])

# Add the predictions to the unknown_data DataFrame
unknown_data["Predicted_Survival"] = unknown_data_predictions

# Print the DataFrame with predictions
print(unknown_data)


# Accuracy:

# Accuracy is the ratio of correctly predicted instances to the total instances in the dataset.
# Formula: (TP + TN) / (TP + TN + FP + FN)
# In the given output, the accuracy is 0.4, indicating that 40% of the predictions were correct.
# Confusion Matrix:

# A confusion matrix is a table that describes the performance of a classification model.
# It presents a summary of correct and incorrect predictions, broken down by each class (in this case, classes 0 and 1).
# The elements of the confusion matrix are:
# True Positive (TP): Actual class is 1, and the model predicted 1.
# True Negative (TN): Actual class is 0, and the model predicted 0.
# False Positive (FP): Actual class is 0, but the model predicted 1 (Type I error).
# False Negative (FN): Actual class is 1, but the model predicted 0 (Type II error).
# In the given confusion matrix:
# [1 0] indicates 1 true positive and 0 false positives for class 0.
# [3 1] indicates 1 true positive and 3 false negatives for class 1.
# Classification Report:

# The classification report provides precision, recall, f1-score, and support for each class.
# Precision: The ratio of true positives to the sum of true positives and false positives.
# Recall: The ratio of true positives to the sum of true positives and false negatives.
# F1-score: The weighted average of precision and recall. It's a good way to show that a classifer has a good value for both recall and precision.
# Support: The number of actual occurrences of the class in the specified dataset.
# Macro Avg: The average of precision, recall, and f1-score for all classes.
# Weighted Avg: The weighted average of precision, recall, and f1-score based on the number of samples for each class.


# Precision:

# For class 0: 25% of the things we predicted as class 0 were actually class 0.
# For class 1: If we predicted something as class 1, we were correct 100% of the time.
# Recall:

# For class 0: We found all the actual class 0 cases, so it's 100%.
# For class 1: We only found 25% of the actual class 1 cases.
# F1-Score:

# It's like an average of precision and recall. It's 40% for both classes.
# Accuracy:

# Overall, we got 40% of the predictions correct.
# Macro Avg:

# It's the average of the precision, recall, and f1-score for both classes.
# It's 62% on average for precision, recall, and f1-score.
# Weighted Avg:

# It's another type of average considering the number of samples for each class.
# It's 85% for precision, 40% for recall, and 40% for f1-score, considering the weights based on the number of samples.