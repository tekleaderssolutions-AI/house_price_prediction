import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

df=pd.read_csv('preprocessed_house_price_dataset.csv')

x=df.drop(columns=['Price'])
y=df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)

mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print(f'Linear Regression - MSE: {mse_lr}, R2: {r2_lr}')

# Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)

rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)    
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest Regressor - MSE: {mse_rf}, R2: {r2_rf}')

'''

\
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_
print(grid_search.best_params_)
'''


# Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
print(f'Decision Tree Regressor - MSE: {mse_dt}, R2: {r2_dt}')

# Support Vector Regressor
svr_model = SVR(C=100, epsilon=0.01, kernel='linear')
svr_model.fit(x_train, y_train)
y_pred_svr = svr_model.predict(x_test)
mse_svr = mean_squared_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)
print(f'Support Vector Regressor - MSE: {mse_svr}, R2: {r2_svr}')

'''
param_grid = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.1, 0.5],
    'kernel': ['rbf', 'linear', 'poly']
}

svr = SVR()
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='r2')
grid_search.fit(x_train, y_train)

best_svr = grid_search.best_estimator_
print(grid_search.best_params_)
'''

import pickle
rf_model_filename = 'random_forest_model.pkl'
svm_model_filename = 'svm_model.pkl'
dt_model_filename = 'decision_tree_model.pkl'
lr_model_filename = 'linear_regression_model.pkl'

with open(rf_model_filename, 'wb') as file:
    pickle.dump(rf_model, file) 

with open(svm_model_filename, 'wb') as file:
    pickle.dump(svr_model, file)

with open(dt_model_filename, 'wb') as file:
    pickle.dump(dt_model, file)

with open(lr_model_filename, 'wb') as file:
    pickle.dump(lr_model, file)
