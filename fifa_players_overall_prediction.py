# -*- coding: utf-8 -*-
"""Fifa Players Overall Prediction

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13DnHX7TeCYrp6lmxYgun8aJeFlYhpZGj
"""

from google.colab import drive
drive.mount('/content/drive')

import os
import csv
import sklearn
import pandas as pd
import numpy as np
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from numpy import array
from sklearn import tree, metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from google.colab import drive
drive.mount('/content/drive')

# importing the datasets
players_21_dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Mid-semester Project Work/players_21.csv")
players_22_dataframe = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Mid-semester Project Work/players_22.csv")

players_21_dataframe.info()

# finding the null values
null_columns_21 = players_21_dataframe.columns[players_21_dataframe.isnull().any()]

# removing objects from the dataset
numeric_dataset_21 = players_21_dataframe.select_dtypes(exclude = ["object"])

# finding the mean of each column in the dataset
columns_mean_21 = numeric_dataset_21.mean()

# replacing the null values with the mean of their respective columns
numeric_dataset_21.fillna(columns_mean_21, inplace = True)

# removing numeric columns from the dataset
categorical_dataset_21 = players_21_dataframe.select_dtypes(include=["object"])

from pandas.core.arrays import categorical

# encoding each cateorical column
label_encoder = LabelEncoder()
for column in categorical_dataset_21:
    if categorical_dataset_21[column].dtype == 'object':
        categorical_dataset_21[column] = label_encoder.fit_transform(categorical_dataset_21[column])

# combinig both the numeric and encoded categorical datasets
combined_dataset_21 = pd.concat([numeric_dataset_21, categorical_dataset_21], axis=1)

# choosing the attribute to predict
target_column_21 = "overall"

# finding the correlation between the columns and what we want to predict
correlations_21 = combined_dataset_21.corrwith(players_21_dataframe[target_column_21])
sorted_correlations = correlations_21.sort_values(ascending = True)

# choosing columns with the highest correlations
desired_columns_21 = ["potential", "value_eur", "wage_eur", "age", "release_clause_eur", "shooting", "passing", "dribbling",
                      "physic", "attacking_short_passing", 'skill_long_passing', "movement_reactions", "power_shot_power",
                      "mentality_vision", "mentality_composure", "mentality_positioning"]

# creating a datasset with the desired columns
selected_dataset_21 = numeric_dataset_21[desired_columns_21]

from sklearn.preprocessing import StandardScaler

# scaling our dataset
SS = StandardScaler()
selected_dataset_21 = SS.fit_transform(selected_dataset_21)

# saving the scaled model
import pickle
pickle_out = open("scaler.pkl", "wb")
pickle.dump(SS, pickle_out)
pickle_out.close()

# giving a new variable to our dataset
X_data = pd.DataFrame(selected_dataset_21)

# giving a new variable to our target column
Y_data = numeric_dataset_21["overall"]

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X_data, Y_data, test_size = 0.2, random_state = 42)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

"""### **RANDOM FOREST**"""

# training our dataset with a random forest model
random_forest_model = RandomForestRegressor(n_estimators = 300, random_state = 42)
random_forest_model.fit(Xtrain, Ytrain)

# testing the model
random_forest_prediction = random_forest_model.predict(Xtest)
random_forest_prediction = pd.DataFrame(random_forest_prediction)

# calculating the accuracy
random_forest_MAE = mean_absolute_error(Ytest, random_forest_prediction)
random_forest_MAE = mean_squared_error(Ytest, random_forest_prediction)
print(f"Mean Absolute Error: {random_forest_MAE:.4f}")
print(f"Mean Squared Error: {random_forest_MAE:.4f}")

from sklearn.ensemble import GradientBoostingRegressor

"""### **GRADIENT BOOST**"""

#raining our dataset with a gradient boost
gradient_boost_model = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)
gradient_boost_model.fit(Xtrain, Ytrain)

# testing the model
gradient_boost_prediction = gradient_boost_model.predict(Xtest)
gradient_boost_prediction = pd.DataFrame(gradient_boost_prediction)

# calculating the accuracy
gradient_boost_MAE = mean_absolute_error(Ytest, gradient_boost_prediction)
gradient_boost_MAE = mean_squared_error(Ytest, gradient_boost_prediction)
print(f"Mean Absolute Error: {gradient_boost_MAE:.4f}")
print(f"Mean Squared Error: {gradient_boost_MAE:.4f}")

"""### **XGB BOOST**"""

import xgboost as xgb
from xgboost import XGBRegressor

# training our dataset with an XG boost model
xgboost_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = 42)
xgboost_model.fit(Xtrain, Ytrain)

# testing the model
xgboost_prediction = xgboost_model.predict(Xtest)
xgboost_prediction = pd.DataFrame(xgboost_prediction)

# calculating the accuracy
xgboost_MAE = mean_absolute_error(Ytest, xgboost_prediction)
xgboost_MSE = mean_squared_error(Ytest, xgboost_prediction)
print(f"Mean Absolute Error: {xgboost_MAE:.4f}")
print(f"Mean Squared Error: {xgboost_MSE:.4f}")

from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

"""# **VOTING REGRESSOR**"""

# training our dataset with a voting regressor model
linear_regressor_model = LinearRegression()
svr_model = SVR(kernel = "linear")
random_forest_model = RandomForestRegressor(n_estimators = 300, random_state = 42)
gradient_boost_model = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.1, random_state = 42)
xgboost_model = xgb.XGBRegressor(objective = "reg:squarederror", random_state = 42)

# testing the model
voting_regressor_model = VotingRegressor(estimators = [
    ("linear regressor", linear_regressor_model),
    ("svr", svr_model),
    ("random forest", random_forest_model),
    ("gradient boost", gradient_boost_model),
    ("xgboost", xgboost_model)
])
voting_regressor_model

# calculating the accuracy
models = [linear_regressor_model, svr_model, random_forest_model, gradient_boost_model, xgboost_model, voting_regressor_model]
for model in models:
    model.fit(Xtrain, Ytrain)
    Y_prediction = model.predict(Xtest)
    Y_MAE = mean_absolute_error(Ytest, Y_prediction)
    Y_MSE = mean_absolute_error(Ytest, Y_prediction)
    Y_RMSE = np.sqrt(Y_MSE)
    r2 = r2_score(Ytest, Y_prediction)

    print(f"Model: {model.__class__.__name__}")
    print(f"Mean Absolute Error: {Y_MAE:.4f}")
    print(f"Mean Squared Error: {Y_MSE:.4f}")
    print(f"Root Mean Squared Error: {Y_RMSE:.4f}")
    print(f"R-squared (R^2): {r2:.4f}")
    print()

from sklearn.model_selection import GridSearchCV

# Find the best combination of hyperparameters
random_forest = RandomForestRegressor()
param_grid = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [None, 10, 20, 30],
}

# Performing hyperparameter tuning on the random regressor model
grid_search = GridSearchCV(estimator = random_forest, param_grid = param_grid, n_jobs = -1, cv = 3, scoring = "neg_mean_absolute_error")
grid_search.fit(X_data, Y_data)

# Selecting the best hyperparameters from the grid search
best_hyperparameters = grid_search.best_params_
better_random_forest = RandomForestRegressor(**best_hyperparameters)
better_random_forest.fit(Xtrain, Ytrain)

Y_prediction = better_random_forest.predict(Xtest)
Y_prediction = pd.DataFrame(Y_prediction)

Y_MAE = mean_absolute_error(Ytest, Y_prediction)
Y_MSE = mean_squared_error(Ytest, Y_prediction)
r2 = r2_score(Ytest, Y_prediction)
rmse = np.sqrt(Y_MSE)

# calculating the accuracy
print(f'Mean Absolute Error: {Y_MAE:.4f}')
print(f'Mean Squared Error (MSE): {Y_MSE:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')
print(f'R-squared (R^2): {r2:.4f}')

players_22_dataframe.info()

# finding the null values
null_columns_22 = players_22_dataframe.columns[players_22_dataframe.isnull().any()]

# removing objects from the dataset
numeric_dataset_22 = players_22_dataframe.select_dtypes(exclude = ["object"])

# finding the mean of each column in the dataset
columns_mean_22 = numeric_dataset_22.mean()

# replacing the null values with the mean of their respective columns
numeric_dataset_22.fillna(columns_mean_22, inplace = True)

# choosing the attribute to predict
target_column_22 = "overall"

# finding the correlation between the columns and what we want to predict
correlations_22 = numeric_dataset_22.corrwith(players_22_dataframe[target_column_22])
sorted_correlations_22 = correlations_22.sort_values(ascending = True)

# choosing columns with the highest correlations
desired_columns_22 = ["potential", "value_eur", "wage_eur", "age", "release_clause_eur", "shooting", "passing", "dribbling",
                      "physic", "attacking_short_passing", 'skill_long_passing', "movement_reactions", "power_shot_power",
                      "mentality_vision", "mentality_composure", "mentality_positioning"]

# creating a datasset with the desired columns
selected_dataset_22 = numeric_dataset_22[desired_columns_22]

from sklearn.preprocessing import StandardScaler

# scaling our dataset
SS = StandardScaler()
selected_dataset_22 = SS.fit_transform(selected_dataset_22)

X_data = pd.DataFrame(selected_dataset_22)

better_random_forest.predict(Xtest)

# saving the choosen model
import pickle
pickle_out = open("better_random_forest.pkl", "wb")
pickle.dump(better_random_forest, pickle_out)
pickle_out.close()