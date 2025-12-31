import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from preprocessing import Datapreprocessing

# --------------------------
# Load data
# --------------------------
data = pd.read_csv('enhanced_house_price_dataset.csv')

preprocessing = Datapreprocessing()

# Check for nulls / duplicates
print("Nulls:\n", preprocessing.null_check(data))
print("Duplicates:", preprocessing.dupli(data))


df = preprocessing.feature_engineering(data)
df = preprocessing.binary_encoding(df)


X = df.drop(columns=['Price'])
y = df['Price']

X_scaled = preprocessing.normalization(X)


df_final = pd.concat([X_scaled.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
model_columns = X_scaled.columns.tolist()

# Save model columns
with open('model_columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)


x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


class ModelBuilding:
    def __init__(self):
        self.lr_model = LinearRegression()
        self.rf_model = RandomForestRegressor(n_estimators=300, max_features='sqrt', random_state=42)
        self.dt_model = DecisionTreeRegressor(random_state=42)
        self.svr_model = SVR(C=100, epsilon=0.01, kernel='linear')


    def lr(self, x_train, y_train, x_test, y_test):
        y_train_log = np.log1p(y_train)
        self.lr_model.fit(x_train, y_train_log)
        log_pred = self.lr_model.predict(x_test)
        y_pred = np.expm1(log_pred)  # inverse log-transform
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Save model
        with open('linear_regression_model.pkl', 'wb') as f:
            pickle.dump(self.lr_model, f)
        return f'Linear Regression - MSE: {mse}, R2: {r2}'

    # ----------------------
    # Random Forest
    # ----------------------
    def rf(self, x_train, y_train, x_test, y_test):
        self.rf_model.fit(x_train, y_train)
        y_pred = self.rf_model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        with open('random_forest_model.pkl', 'wb') as f:
            pickle.dump(self.rf_model, f)
        return f'Random Forest Regressor - MSE: {mse}, R2: {r2}'

    # ----------------------
    # Decision Tree
    # ----------------------
    def dt(self, x_train, y_train, x_test, y_test):
        self.dt_model.fit(x_train, y_train)
        y_pred = self.dt_model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        with open('decision_tree_model.pkl', 'wb') as f:
            pickle.dump(self.dt_model, f)
        return f'Decision Tree Regressor - MSE: {mse}, R2: {r2}'

    # ----------------------
    # Support Vector Regressor
    # ----------------------
    def svr(self, x_train, y_train, x_test, y_test):
        self.svr_model.fit(x_train, y_train)
        y_pred = self.svr_model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump(self.svr_model, f)
        return f'Support Vector Regressor - MSE: {mse}, R2: {r2}'

# --------------------------
# Train and evaluate all models
# --------------------------
builder = ModelBuilding()
print(builder.lr(x_train, y_train, x_test, y_test))
print(builder.rf(x_train, y_train, x_test, y_test))
print(builder.dt(x_train, y_train, x_test, y_test))
print(builder.svr(x_train, y_train, x_test, y_test))
