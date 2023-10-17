import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from prettytable import PrettyTable


# reading dataset
df = pd.read_csv('tested.csv')

# Data Preprocessing
df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Impute the missing values in 'Age'
imputer = SimpleImputer(strategy='mean')
df["Age"] = imputer.fit_transform(df[["Age"]])

# Impute the  missing values in other columns if any
# Filling missing values with their mean for numerical columns

df.fillna(df.mean(), inplace=True)  

# Split the dataset into features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]


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

new_df = pd.read_csv('new_titanic.csv')
temp_df=new_df.copy()
# Data Preprocessing for the new dataset
# Drop unnecessary columns 
new_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1, inplace=True)
new_df["Sex"] = new_df["Sex"].map({"male": 0, "female": 1})
new_df = pd.get_dummies(new_df, columns=["Embarked"], drop_first=True)

# Impute missing values in 'Age' and other numerical columns
imputer = SimpleImputer(strategy='mean')
new_df["Age"] = imputer.fit_transform(new_df[["Age"]])
new_df.fillna(new_df.mean(), inplace=True)  
#Above -  Filling missing values with mean for numerical columns

# Making predictions using the trained model
X_new = new_df[selected_features]  
predictions = model.predict(X_new)

# Print the predictions
print("Predicted Survived for the new dataset (survived - yes / can't survive - no ) : ")
print(predictions)


print("------------------------")
names = temp_df["Name"]
PID = temp_df["PassengerId"]


# Create a PrettyTable instance obj.
table = PrettyTable()


table.field_names = ["PassengerId", "Name", "Prediction"]
table.align["PassengerId"] = "l"
table.align["Name"] = "l"
table.align["Prediction"] = "l"

# adding records to table
for i in range(len(predictions)):
    prediction = "yes" if predictions[i] == 1 else "no"
    table.add_row([PID[i], names[i], prediction])

# Print the table
print(table)

# normal way
# for i in range(len(predictions)):
#     if predictions[i] == 1:
#         print(f" {PID[i]} {names[i]} - \t yes")
#     else:
#         print(f" {PID[i]} {names[i]} - \t no")
# print("-------using zip------")
# for name, prediction in zip(names, predictions):
#     if prediction == 1:
#         print(f"{name}: Survived")
#     else:
#         print(f"{name}: Can't survive")