from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
import numpy as np

# load cleaned and encoded dataset
data = pd.read_csv(r"C:\Users\Srikanth Tata\Downloads\house_price_prediction\cleaned_encoded_house_price_dataset.csv")

# Define features and target variable
X = data.drop('Price', axis=1)
Y = data['Price']

# Split the dataset inti training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Intialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Make prediction on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
accuracy = model.score(X_test, Y_test)
print(f"Linear Regression R^2 Score: {r2}")
print(f"Linear Regression Model Accuracy: {accuracy * 100:.2f}%")
print("--------------------------------")

# Train the random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# make predictio on the test set
rf_model_pred = rf_model.predict(X_test)

# Evaluate the random forest model
rf_mse = mean_squared_error(Y_test, rf_model_pred)
rf_r2 = r2_score(Y_test, rf_model_pred)
rf_accuracy = rf_model.score(X_test, Y_test)
print(f"Random Forest R^2 Score: {rf_r2}")
print(f"Random Forest Model Accuracy: {rf_accuracy * 100:.2f}%")
print("--------------------------------")

# Train the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, Y_train) 

# Make prediction on the test set
xgb_model_pred = xgb_model.predict(X_test)

# Evaluate the XGBoost model                
xgb_mse = mean_squared_error(Y_test, xgb_model_pred)
xgb_r2 = r2_score(Y_test, xgb_model_pred)
xgb_accuracy = xgb_model.score(X_test, Y_test)
print(f"XGBoost R^2 Score: {xgb_r2}")
print(f"XGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")
print("--------------------------------")

# save the best model
best_model = None
if xgb_r2 >= rf_r2 and xgb_r2 >= r2:
    best_model = xgb_model
    model_name = "XGBoost"
elif rf_r2 >= xgb_r2 and rf_r2 >= r2:   
    best_model = rf_model
    model_name = "Random Forest"
else:
    best_model = model
    model_name = "Linear Regression"
joblib.dump(best_model, f"{model_name}_house_price_model.pkl")
print(f"The best model is {model_name} and has been saved as {model_name}_house_price_model.pkl")


