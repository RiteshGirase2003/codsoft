import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.impute import SimpleImputer

# reading dataset
data = pd.read_csv('movie_dataset_new.csv')

# Cleaning the data
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Duration'] = data['Duration'].str.extract('(\d+)').astype(float)
data['Votes'] = data['Votes'].str.replace(',', '').astype(float)
data.dropna(subset=['Year', 'Duration', 'Votes', 'Rating'], inplace=True)

# Selecting ->  features and target
features = ['Year', 'Duration', 'Votes']
X = data[features]
y = data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# count -> no. of samples 
print('Training completed.\n Number of training samples:', len(X_train))

# Print -> coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Print mean squared error on the test-set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error on Test Set:', mse)



from prettytable import PrettyTable

# reading the new dataset -> to do prediction 
new_data = pd.read_csv('movie_testing.csv')  

# Selecting features
new_features = ['Year', 'Duration', 'Votes']

# ignoring records with missing data
new_data = new_data.dropna(subset=new_features)

# Use SimpleImputer to fill missing values with the mean
imputer = SimpleImputer(strategy='mean')
new_data[new_features] = imputer.fit_transform(new_data[new_features])

# Predict movie ratings for the new dataset
predicted_ratings = model.predict(new_data[new_features])

# Cliping predicted ratings to be within the range of 0 to 10
predicted_ratings_clipped = np.clip(predicted_ratings, 0, 10)

# Add the predicted ratings to the dataset
new_data['Predicted_Rating'] = predicted_ratings_clipped

# Create a PrettyTable instance obj.
rating_table = PrettyTable()

# giving -> field names to table
rating_table.field_names = ["Movie", "Year", "Duration", "Votes", "Predicted Rating"]

# adding records to rating table
for index, row in new_data.iterrows():
    rating_table.add_row([row['Movie'], row['Year'], row['Duration'], row['Votes'], row['Predicted_Rating']])

# Print the table
print(rating_table)
